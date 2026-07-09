"""π0.5 顶层模型（M0：单步前向 + 导出 spec）。

对齐 openpi ``pi0_pytorch.py``：``embed_prefix`` / ``embed_suffix`` / ``denoise_step``。

M0 把模型拆成三个可导出的函数，便于宿主侧 engine 编排与对拍：
- ``embed_image``     : 单张图 -> 图像 token
- ``embed_language``  : 语言 token id -> 语言 embedding（已乘 sqrt(width)）
- ``denoise_step``    : (prefix_embs, x_t, time_emb) -> v_t

说明：
- π0.5 的 state 进入离散语言 token，故 suffix 只含动作 token（见 openpi pi05 分支）。
- 时间正弦编码在宿主侧用 numpy 预计算后作为 ``time_emb`` 传入；in-graph 只跑 time MLP。
- RoPE cos/sin 与注意力 mask 在固定 shape 下作为常量 bake 进图。
"""

from __future__ import annotations

import math

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .gemma_dual import DualExpertGemma
from .pi0_config import Pi0Config
from .siglip import SiglipVisionTower


def _build_additive_mask(prefix_len: int, suffix_len: int, neg_inf: float, dtype: str) -> np.ndarray:
    """构造 π0.5 的加性注意力 mask [1,1,T,T]（对齐 openpi make_att_2d_masks）。

    - prefix（att=0）：只能看 prefix
    - suffix（首 token att=1，其余 0）：可看 prefix + 全部 suffix
    """
    total = prefix_len + suffix_len
    att = np.zeros(total, dtype=np.int32)
    att[prefix_len] = 1  # suffix 第一个 token 起新 block
    cumsum = np.cumsum(att)
    allow = cumsum[None, :] <= cumsum[:, None]  # [T,T], allow[i,j]: i 可看 j
    mask = np.where(allow, 0.0, neg_inf).astype(dtype)
    return mask.reshape(1, 1, total, total)


_ALL_STAGES = ("vision", "embed", "backbone")


def include_for(functions) -> set[str]:
    """按导出函数推断需要构建的子模块（分段编译：每个 engine 只携带自身权重）。"""
    if functions is None:
        return set(_ALL_STAGES)
    need: set[str] = set()
    for f in functions:
        if f == "embed_image":
            need.add("vision")
        elif f == "embed_language":
            need.add("embed")
        elif f in ("denoise_step", "prefill", "denoise_step_kv"):
            need.add("backbone")
    return need or set(_ALL_STAGES)


class Pi0Model(nn.Module):
    def __init__(self, config: Pi0Config, include: set[str] | None = None):
        """``include``：仅构建指定子模块（``vision`` / ``embed`` / ``backbone``）。

        默认构建全部（向后兼容）。分段编译时按需只建一部分，使 ``export_tvm`` 的 packed
        参数只含该 stage 自身权重（否则每个 engine 会打包整模 775 个参数）。
        """
        self.config = config
        vlm = config.vlm
        ae = config.action_expert
        include = set(_ALL_STAGES) if include is None else set(include)
        self._include = include

        self.dtype = config.dtype
        if "vision" in include:
            self.vision = SiglipVisionTower(config.siglip)
        if "embed" in include:
            self.embed_tokens = nn.Embedding(PALIGEMMA_VOCAB_SIZE := 257_152, vlm.width)
            self.vocab_size = PALIGEMMA_VOCAB_SIZE
        if "backbone" in include:
            self.backbone = DualExpertGemma(config)
            # 动作头
            self.action_in_proj = nn.Linear(config.action_dim, ae.width, bias=True)
            self.action_out_proj = nn.Linear(ae.width, config.action_dim, bias=True)
            # π0.5 time MLP（产生 adaRMS 条件）
            self.time_mlp_in = nn.Linear(ae.width, ae.width, bias=True)
            self.time_mlp_out = nn.Linear(ae.width, ae.width, bias=True)

        self._vlm_width = vlm.width
        self._ae_width = ae.width
        self._mask_np = _build_additive_mask(
            config.prefix_len, config.suffix_len, config.attn_neg_inf, config.dtype
        )

    # ---------- 子函数 ----------
    def embed_image(self, image: Tensor) -> Tensor:
        """[1,224,224,3] -> [1, num_patches, vlm_width]。"""
        return self.vision(image)

    def embed_language(self, input_ids: Tensor) -> Tensor:
        """[1, max_token_len] int32 -> [1, max_token_len, vlm_width]（乘 sqrt(width)）。"""
        emb = self.embed_tokens(input_ids)
        return emb * math.sqrt(self._vlm_width)

    def _time_cond(self, time_emb: Tensor) -> Tensor:
        x = op.silu(self.time_mlp_in(time_emb))
        x = op.silu(self.time_mlp_out(x))
        return x  # adaRMS 条件 [1, ae_width]

    def denoise_step(self, prefix_embs: Tensor, x_t: Tensor, time_emb: Tensor) -> Tensor:
        """单步去噪：返回 v_t [1, action_horizon, action_dim]。"""
        cfg = self.config
        total = cfg.prefix_len + cfg.suffix_len
        cos, sin = self.backbone.make_rope_tables(total)
        attn_mask = nn.Tensor.from_const(self._mask_np)

        suffix_emb = self.action_in_proj(x_t)  # [1,horizon,ae_width]
        adarms_cond = self._time_cond(time_emb)

        xs = [prefix_embs.astype(self.dtype), suffix_emb.astype(self.dtype)]
        outs = self.backbone(xs, cos, sin, attn_mask, [None, adarms_cond])
        # action_out_proj 在模型 dtype 下计算，最后统一回 float32（对 fp16 权重也自洽）
        v_t = self.action_out_proj(outs[1])
        return v_t.astype("float32")

    # ---------- M1：prefix 固化 + suffix-only 去噪 ----------
    def prefill(self, prefix_embs: Tensor, prefix_mask: Tensor):
        """expert-0 对 prefix 跑一次，返回逐层缓存 (keys, values)。

        - ``prefix_mask``：[1,1,1,prefix_len] 加性 mask（0=有效，-inf=padding），屏蔽 padded
          prefix 列，使有效 token 的 K/V 与 openpi（带 pad mask 的 prefill）一致。全 0 时退化为
          M1 无 padding 语义。
        - keys/values 形状 [depth, 1, kv, prefix_len, head_dim]（keys fp32，values 模型 dtype）。
        """
        cos, sin = self.backbone.make_rope_tables(self.config.prefix_len, offset=0)
        keys, values = self.backbone.prefill_prefix(
            prefix_embs.astype(self.dtype), cos, sin, add_mask=prefix_mask
        )
        return keys, values

    def denoise_step_kv(
        self, keys: Tensor, values: Tensor, x_t: Tensor, time_emb: Tensor,
        suffix_cos: Tensor, suffix_sin: Tensor, prefix_mask: Tensor,
    ) -> Tensor:
        """suffix-only 去噪：expert-1 用外部 prefix K/V，返回 v_t [1,horizon,action_dim]。

        - ``suffix_cos``/``suffix_sin``：[1,suffix_len,1,head_dim//2] fp32，宿主按 suffix 位置
          （openpi：``sum(prefix_pad)`` + arange(suffix_len)）预算的 RoPE 表。padding 会改变有效
          prefix 长度，故 suffix RoPE offset 由宿主传入而非编译期 bake。
        - ``prefix_mask``：[1,1,1,prefix_len] 加性 mask，屏蔽 padded prefix 列。
        """
        cfg = self.config
        suffix_emb = self.action_in_proj(x_t).astype(self.dtype)
        adarms_cond = self._time_cond(time_emb)
        # 拼出 suffix 对 [prefix; suffix] 的完整加性 mask：prefix 段用 pad mask，suffix 段全 0
        zeros_suffix = nn.Tensor.from_const(np.zeros((1, 1, 1, cfg.suffix_len), "float32"))
        full_mask = op.concat([prefix_mask, zeros_suffix], dim=-1)  # [1,1,1,prefix_len+suffix_len]
        out = self.backbone.decode_suffix(
            suffix_emb, keys, values, suffix_cos, suffix_sin, adarms_cond, add_mask=full_mask
        )
        v_t = self.action_out_proj(out)
        return v_t.astype("float32")

    # ---------- 导出 spec ----------
    def get_default_spec(self, functions=None):
        """构造导出 spec。``functions`` 可选，仅导出指定子函数（如 ["denoise_step"]）。

        注：SigLIP 的 ``embed_image`` 含 ``nn.LayerNorm``，而 TVM ``layer_norm``
        目前只支持 fp32/fp16；若要用 bf16 编译 backbone，可只导出 ``denoise_step``。
        """
        cfg = self.config
        s = cfg.siglip
        vlm = cfg.vlm
        kv, hd, depth = vlm.num_kv_heads, vlm.head_dim, vlm.depth
        kv_shape = [depth, 1, kv, cfg.prefix_len, hd]
        # prefix 加性 mask（0=有效 / -inf=padding），广播到 [B,heads,Tq,prefix_len]
        prefix_mask_shape = [1, 1, 1, cfg.prefix_len]
        # 宿主预算的 suffix RoPE 表（offset=有效 prefix 长度）
        rope_shape = [1, cfg.suffix_len, 1, hd // 2]
        mod_spec = {
            "embed_image": {
                "image": nn.spec.Tensor([1, s.image_size, s.image_size, s.num_channels], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "embed_language": {
                "input_ids": nn.spec.Tensor([1, cfg.max_token_len], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "denoise_step": {
                "prefix_embs": nn.spec.Tensor([1, cfg.prefix_len, self._vlm_width], self.dtype),
                "x_t": nn.spec.Tensor([1, cfg.action_horizon, cfg.action_dim], self.dtype),
                "time_emb": nn.spec.Tensor([1, self._ae_width], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "prefix_embs": nn.spec.Tensor([1, cfg.prefix_len, self._vlm_width], self.dtype),
                "prefix_mask": nn.spec.Tensor(prefix_mask_shape, "float32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "denoise_step_kv": {
                "keys": nn.spec.Tensor(kv_shape, "float32"),
                "values": nn.spec.Tensor(kv_shape, self.dtype),
                "x_t": nn.spec.Tensor([1, cfg.action_horizon, cfg.action_dim], self.dtype),
                "time_emb": nn.spec.Tensor([1, self._ae_width], self.dtype),
                "suffix_cos": nn.spec.Tensor(rope_shape, "float32"),
                "suffix_sin": nn.spec.Tensor(rope_shape, "float32"),
                "prefix_mask": nn.spec.Tensor(prefix_mask_shape, "float32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        }
        if functions is not None:
            mod_spec = {k: mod_spec[k] for k in functions}
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
