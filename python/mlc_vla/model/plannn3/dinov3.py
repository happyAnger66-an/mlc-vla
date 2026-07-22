"""plannn3 视觉 tokenizer stage（DINOv3 / EVA02 视觉塔，M1）。

忠实复刻 `laser_model_export/.../encoder/dinov3_encoder.py` 的 `Dinov3Encoder`：

    frame[B,3,H,W]
      → patchify   Conv2d(3,64,k2,s2)
      → downsample DownsampleBlock(64,256,k3,stride=(1,2))         # SiLU + conv1 + conv2 + shortcut
      → patch_embed Conv2d(256,384,k16,s16)  → [B,Hp,Wp,384] 展平 token
      → [cls(1) + reg(4)] 前缀 token + Eva blocks×12（dinov3 2D-RoPE, LayerScale, SwiGLU）
      → 丢前缀 → reshape[B,384,Hp,Wp]
      → expand     Conv2d(384,1024,k3,s1,p1)
      → flatten    → [B, Hp*Wp, 1024]

由于 `Dinov3Encoder` 用 `eva_cfg={}`（全默认）+ `patch_embed.proj` 替换成 Conv2d(256,384,16,16)，
EVA 结构即 timm 默认：embed_dim=384 / depth=12 / heads=6 / head_dim=64 / mlp_ratio=4(SwiGLU align8) /
qkv_bias=False / scale_norm=False / init_values=1e-5(LayerScale) / cls+4reg=5 前缀 / rope_type='dinov3'
rotate_half。

按 arch.md「独立 vit engine」：本 stage 只做单帧 DINOv3 backbone；多视角/时序拼接、view token、
PE 投影等编排（数据无关的静态 Python 循环）留在宿主侧。dinov3 2D-RoPE 的 sin/cos 按 patch 网格在
宿主 numpy 预算并 bake 为常量（对齐 `pos_embed_sincos.RotaryEmbeddingDinoV3`）。精度：视觉塔恒 fp32。
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


def _conv_out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


@dataclasses.dataclass
class Dinov3Config:
    """DINOv3 视觉塔配置（对齐 timm 默认 Eva + Dinov3Encoder 外壳）。"""

    # CNN stem
    patchify_out: int = 64
    downsample_out: int = 256
    downsample_stride: tuple = (1, 2)
    # EVA
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_prefix_tokens: int = 5  # cls(1) + reg(4)
    init_values: float = 1e-5  # LayerScale
    eps: float = 1e-5
    patch_embed_kernel: int = 16
    # 输出
    mid_channel: int = 384
    out_channel: int = 1024
    # dinov3 rope
    rope_temperature: float = 100.0
    # 输入图（单帧，编译期固定；不同相机尺寸编译不同 variant）
    image_h: int = 512
    image_w: int = 1536
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)

    def grid(self) -> tuple:
        """返回 patch 网格 (Hp, Wp)（依 stem 卷积输出公式推导）。"""
        h1, w1 = self.image_h // 2, self.image_w // 2  # patchify k2s2
        sh, sw = self.downsample_stride
        h2 = _conv_out(h1, 3, sh, 1)  # downsample conv1 k3 pad1
        w2 = _conv_out(w1, 3, sw, 1)
        k = self.patch_embed_kernel
        hp = _conv_out(h2, k, k, 0)  # patch_embed k16 s16
        wp = _conv_out(w2, k, k, 0)
        return hp, wp

    @classmethod
    def dummy(cls) -> "Dinov3Config":
        """小尺寸冒烟配置（秒级）。"""
        return cls(depth=2, image_h=64, image_w=128)


def _dinov3_rope_np(hp: int, wp: int, head_dim: int, temperature: float):
    """按 patch 网格预算 dinov3 2D-RoPE 的 sin/cos，形状 [1,1,Hp*Wp,head_dim]（fp32）。

    对齐 `pos_embed_sincos`：0.5-centered、separate 归一、映射到 [-1,1]；
    periods=temperature**(2*arange(head_dim//4)/(head_dim//2))；rotate_half（repeat 拼接）。
    """
    d4 = head_dim // 4
    coords_h = (np.arange(0.5, hp, dtype=np.float64)) / float(hp)
    coords_w = (np.arange(0.5, wp, dtype=np.float64)) / float(wp)
    gh, gw = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([gh, gw], axis=-1).reshape(-1, 2)  # (HW,2)
    coords = 2.0 * coords - 1.0
    exponents = 2.0 * np.arange(d4, dtype=np.float64) / (head_dim // 2)
    periods = temperature ** exponents  # (d4,)
    angles = 2.0 * math.pi * coords[:, :, None] / periods[None, None, :]  # (HW,2,d4)
    angles = angles.reshape(coords.shape[0], -1)  # (HW, head_dim//2)
    angles = np.concatenate([angles, angles], axis=-1)  # rotate_half -> (HW, head_dim)
    sin = np.sin(angles).reshape(1, 1, coords.shape[0], head_dim).astype("float32")
    cos = np.cos(angles).reshape(1, 1, coords.shape[0], head_dim).astype("float32")
    return sin, cos


def _rope_rotate_half(x: Tensor) -> Tensor:
    """[-x2, x1]：把最后一维对半切，后半取负放前面。"""
    x1, x2 = op.split(x, 2, axis=-1)
    return op.concat([x2 * (-1.0), x1], dim=-1)


def _apply_rope_dinov3(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """x: [B,heads,M,head_dim]；sin/cos: [1,1,M,head_dim]。返回 x*cos + rot_half(x)*sin。"""
    return x * cos + _rope_rotate_half(x) * sin


class DownsampleBlock(nn.Module):
    def __init__(self, cfg: Dinov3Config):
        sh, sw = cfg.downsample_stride
        ic, oc = cfg.patchify_out, cfg.downsample_out
        self.conv1 = nn.Conv2D(ic, oc, 3, stride=(sh, sw), padding=1)
        self.conv2 = nn.Conv2D(oc, oc, 3, stride=1, padding=1)
        self.conv_shortcut = nn.Conv2D(ic, oc, 3, stride=(sh, sw), padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(op.silu(x))
        h = self.conv2(op.silu(h))
        return self.conv_shortcut(x) + h


class EvaAttention(nn.Module):
    def __init__(self, cfg: Dinov3Config):
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.embed_dim = cfg.embed_dim
        self.num_prefix = cfg.num_prefix_tokens
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        b, n, _ = x.shape
        q, k, v = op.split(self.qkv(x), 3, axis=-1)  # each [b,n,embed]

        def to_heads(t):
            return op.permute_dims(op.reshape(t, (b, n, self.num_heads, self.head_dim)), [0, 2, 1, 3])

        q, k, v = to_heads(q), to_heads(k), to_heads(v)  # [b,heads,n,hd]
        # rope 只作用在 patch token（前缀 cls/reg 不加）
        q_pre, q_pat = op.split(q, [self.num_prefix], axis=2)
        k_pre, k_pat = op.split(k, [self.num_prefix], axis=2)
        q = op.concat([q_pre, _apply_rope_dinov3(q_pat, sin, cos)], dim=2)
        k = op.concat([k_pre, _apply_rope_dinov3(k_pat, sin, cos)], dim=2)

        kt = op.permute_dims(k, [0, 1, 3, 2])
        attn = op.matmul(q, kt, out_dtype="float32") * self.scale
        attn = op.softmax(attn, axis=-1)
        out = op.matmul(attn, v)  # [b,heads,n,hd]
        out = op.reshape(op.permute_dims(out, [0, 2, 1, 3]), (b, n, self.embed_dim))
        return self.proj(out)


class SwiGLU(nn.Module):
    """timm SwiGLU（fc1_g / fc1_x 分离，act=SiLU，无 norm）。"""

    def __init__(self, cfg: Dinov3Config):
        self.fc1_g = nn.Linear(cfg.embed_dim, cfg.mlp_hidden, bias=True)
        self.fc1_x = nn.Linear(cfg.embed_dim, cfg.mlp_hidden, bias=True)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(op.silu(self.fc1_g(x)) * self.fc1_x(x))


class EvaBlock(nn.Module):
    """EVA block w/ LayerScale：x = x + w1*attn(norm1(x)); x = x + w2*mlp(norm2(x))。"""

    def __init__(self, cfg: Dinov3Config):
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)
        self.attn = EvaAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)
        self.mlp = SwiGLU(cfg)
        self.weight_1 = nn.Parameter((cfg.embed_dim,))
        self.weight_2 = nn.Parameter((cfg.embed_dim,))

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        x = x + self.weight_1 * self.attn(self.norm1(x), sin, cos)
        x = x + self.weight_2 * self.mlp(self.norm2(x))
        return x


class Dinov3VisualEncoder(nn.Module):
    """单帧 DINOv3 视觉 tokenizer → [B, Hp*Wp, out_channel]。"""

    def __init__(self, cfg: Dinov3Config):
        self.config = cfg
        self.dtype = cfg.dtype
        hp, wp = cfg.grid()
        self.hp, self.wp = hp, wp

        self.patchify = nn.Conv2D(3, cfg.patchify_out, 2, stride=2, padding=0)
        self.downsample = DownsampleBlock(cfg)
        self.patch_embed = nn.Conv2D(cfg.downsample_out, cfg.embed_dim, cfg.patch_embed_kernel,
                                     stride=cfg.patch_embed_kernel, padding=0)
        self.cls_token = nn.Parameter((1, 1, cfg.embed_dim))
        self.reg_token = nn.Parameter((1, cfg.num_prefix_tokens - 1, cfg.embed_dim))
        self.blocks = nn.ModuleList([EvaBlock(cfg) for _ in range(cfg.depth)])
        self.expand = nn.Conv2D(cfg.mid_channel, cfg.out_channel, 3, stride=1, padding=1)

        sin, cos = _dinov3_rope_np(hp, wp, cfg.head_dim, cfg.rope_temperature)
        self._rope_sin = sin
        self._rope_cos = cos

    def embed_visual(self, image: Tensor) -> Tensor:
        cfg = self.config
        b = 1
        hp, wp = self.hp, self.wp
        x = self.patchify(image)          # [b,64,H1,W1]
        x = self.downsample(x)            # [b,256,H2,W2]
        x = self.patch_embed(x)           # [b,384,Hp,Wp]
        # NHWC 展平成 token（对齐 timm dynamic_img_size 的 view(B,HW,C)）
        x = op.reshape(op.permute_dims(x, [0, 2, 3, 1]), (b, hp * wp, cfg.embed_dim))
        x = op.concat([self.cls_token, self.reg_token, x], dim=1)  # [b, prefix+HW, 384]

        sin = nn.Tensor.from_const(self._rope_sin)
        cos = nn.Tensor.from_const(self._rope_cos)
        for blk in self.blocks:
            x = blk(x, sin, cos)

        _, x = op.split(x, [cfg.num_prefix_tokens], axis=1)  # 丢前缀 -> [b,HW,384]
        x = op.permute_dims(op.reshape(x, (b, hp, wp, cfg.embed_dim)), [0, 3, 1, 2])  # [b,384,Hp,Wp]
        x = self.expand(x)                # [b,1024,Hp,Wp]
        x = op.permute_dims(op.reshape(x, (b, cfg.out_channel, hp * wp)), [0, 2, 1])  # [b,HW,1024]
        return x

    def get_default_spec(self, functions=None):
        cfg = self.config
        mod_spec = {
            "embed_visual": {
                "image": nn.spec.Tensor([1, 3, cfg.image_h, cfg.image_w], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        }
        if functions is not None:
            mod_spec = {k: mod_spec[k] for k in functions}
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
