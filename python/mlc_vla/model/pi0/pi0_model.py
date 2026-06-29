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


class Pi0Model(nn.Module):
    def __init__(self, config: Pi0Config):
        self.config = config
        vlm = config.vlm
        ae = config.action_expert

        self.dtype = config.dtype
        self.vision = SiglipVisionTower(config.siglip)
        self.embed_tokens = nn.Embedding(PALIGEMMA_VOCAB_SIZE := 257_152, vlm.width)
        self.vocab_size = PALIGEMMA_VOCAB_SIZE
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
        suffix_out = outs[1].astype("float32")
        return self.action_out_proj(suffix_out)

    # ---------- 导出 spec ----------
    def get_default_spec(self):
        cfg = self.config
        s = cfg.siglip
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
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
