"""SigLIP So400m/14 视觉塔 + PaliGemma multi-modal projector。

对齐 HF ``SiglipVisionModel``（PaliGemma 默认视觉配置）+ ``multi_modal_projector``。
输出：每张图 ``num_patches`` 个 token，宽度投影到 Gemma width。

M0 实现说明：
- batch=1、固定 224x224 输入；patch embedding 用「展平 patch + Linear」等价实现
  （等价于 stride=patch 的非重叠卷积），避免对 conv2d 算子的依赖。
- 输入图像约定为 channels-last ``[B, H, W, 3]``（对齐 openpi inputs_spec）。
"""

from __future__ import annotations

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .pi0_config import SiglipConfig


class SiglipMLP(nn.Module):
    def __init__(self, cfg: SiglipConfig):
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(op.gelu(self.fc1(x), approximate="tanh"))


class SiglipAttention(nn.Module):
    def __init__(self, cfg: SiglipConfig):
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        n, hd = self.num_heads, self.head_dim
        q = op.permute_dims(op.reshape(self.q_proj(x), (b, t, n, hd)), [0, 2, 1, 3])
        k = op.permute_dims(op.reshape(self.k_proj(x), (b, t, n, hd)), [0, 2, 1, 3])
        v = op.permute_dims(op.reshape(self.v_proj(x), (b, t, n, hd)), [0, 2, 1, 3])
        kt = op.permute_dims(k, [0, 1, 3, 2])  # [B,N,H,T]
        logits = op.matmul((q * self.scale).astype("float32"), kt.astype("float32"))
        probs = op.softmax(logits, axis=-1).astype(x.dtype)
        ctx = op.matmul(probs, v)  # [B,N,T,H]
        ctx = op.reshape(op.permute_dims(ctx, [0, 2, 1, 3]), (b, t, n * hd))
        return self.out_proj(ctx)


class SiglipEncoderLayer(nn.Module):
    def __init__(self, cfg: SiglipConfig):
        self.self_attn = SiglipAttention(cfg)
        self.mlp = SiglipMLP(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipVisionTower(nn.Module):
    """SigLIP 视觉塔 + multi-modal projector。"""

    def __init__(self, cfg: SiglipConfig):
        self.cfg = cfg
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.num_channels
        self.patch_embedding = nn.Linear(self.patch_dim, cfg.hidden_size, bias=True)
        self.position_embedding = nn.Parameter((cfg.num_patches, cfg.hidden_size))
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        # PaliGemma multi-modal projector：vision hidden -> gemma width
        self.multi_modal_projector = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=True)

    def _patchify(self, image: Tensor) -> Tensor:
        """[B, H, W, 3] -> [B, num_patches, patch_dim]（非重叠 patch 展平）。"""
        cfg = self.cfg
        b = image.shape[0]
        side = cfg.image_size // cfg.patch_size
        p, c = cfg.patch_size, cfg.num_channels
        # [B, side, p, side, p, c]
        x = op.reshape(image, (b, side, p, side, p, c))
        # -> [B, side, side, p, p, c]
        x = op.permute_dims(x, [0, 1, 3, 2, 4, 5])
        # -> [B, num_patches, p*p*c]
        return op.reshape(x, (b, side * side, p * p * c))

    def forward(self, image: Tensor) -> Tensor:
        x = self._patchify(image)
        x = self.patch_embedding(x)
        x = x + op.reshape(self.position_embedding, (1, self.cfg.num_patches, self.cfg.hidden_size))
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return self.multi_modal_projector(x)  # [B, num_patches, projection_dim]
