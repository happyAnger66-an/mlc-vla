"""双专家 Gemma backbone（Mixture-of-Transformers，联合注意力）。

忠实复刻 openpi：
- ``gemma.py`` 的 ``Attention`` / ``FeedForward`` / ``Block`` / ``RMSNorm`` / ``_apply_rope``
- ``gemma_pytorch.py`` 的 ``compute_layer_complete``（prefix/suffix 拼接后做一次 attention）

M0 设计取舍：
- batch=1、固定 shape、**不使用 PagedKVCache**，用显式拼接 + 加性 mask 的 eager 注意力，
  与 openpi eager 路径逐算子对齐（M1 再换 CUDA Graph / KV cache）。
- RoPE 的 cos/sin 表按编译期已知的序列长度用 numpy 常量预生成。
"""

from __future__ import annotations

from typing import List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .pi0_config import GemmaExpertConfig, Pi0Config


def _rope_tables(num_positions: int, head_dim: int, theta: float, dtype: str) -> Tuple[Tensor, Tensor]:
    """生成 RoPE 的 cos/sin 常量表，形状 [1, L, 1, head_dim//2]。

    对齐 openpi ``_apply_rope``：
        freq_exponents = (2/H) * arange(H/2)
        timescale      = theta ** freq_exponents
        radians        = positions / timescale
    """
    half = head_dim // 2
    positions = np.arange(num_positions, dtype=np.float64)
    freq_exponents = (2.0 / head_dim) * np.arange(half, dtype=np.float64)
    timescale = theta**freq_exponents
    radians = positions[:, None] / timescale[None, :]  # [L, H/2]
    cos = np.cos(radians).reshape(1, num_positions, 1, half).astype(dtype)
    sin = np.sin(radians).reshape(1, num_positions, 1, half).astype(dtype)
    return nn.Tensor.from_const(cos), nn.Tensor.from_const(sin)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, L, n_heads, head_dim]；cos/sin: [1, L, 1, head_dim//2]。"""
    x1, x2 = op.split(x, 2, axis=-1)
    first = x1 * cos - x2 * sin
    second = x2 * cos + x1 * sin
    return op.concat([first, second], dim=-1)


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm，支持普通模式与 adaRMS（π0.5 动作专家）。"""

    def __init__(self, dim: int, eps: float, use_adarms: bool, cond_dim: Optional[int] = None):
        self.dim = dim
        self.eps = eps
        self.use_adarms = use_adarms
        if use_adarms:
            assert cond_dim is not None
            # modulation: cond -> [scale, shift, gate]，对齐 openpi 的 zero-init Dense(3*dim)
            self.modulation = nn.Linear(cond_dim, dim * 3, bias=False)
        else:
            # 普通 RMSNorm：存原始 scale，forward 里 (1 + scale)
            self.weight = nn.Parameter((dim,))

    def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        xf = x.astype("float32")
        var = op.sum(op.square(xf), axis=-1, keepdims=True) / float(self.dim)
        normed = op.divide(xf, op.sqrt(var + self.eps))
        if not self.use_adarms:
            scale = self.weight.astype("float32")
            out = normed * (1.0 + scale)
            return out.astype(x.dtype), None
        # adaRMS
        modulation = self.modulation(cond)  # [B, 3*dim]
        modulation = op.reshape(modulation, (modulation.shape[0], 1, self.dim * 3))
        scale, shift, gate = op.split(modulation, 3, axis=-1)  # each [B,1,dim]
        out = normed * (1.0 + scale.astype("float32")) + shift.astype("float32")
        return out.astype(x.dtype), gate


def _gated_residual(x: Tensor, y: Tensor, gate: Optional[Tensor]) -> Tensor:
    if gate is None:
        return x + y
    return x + y * gate


class GemmaExpertMLP(nn.Module):
    """Gemma 门控 FFN：gelu(x@Wg) * (x@Wu) -> @Wd。"""

    def __init__(self, cfg: GemmaExpertConfig):
        self.gate_up_proj = nn.Linear(cfg.width, 2 * cfg.mlp_dim, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_dim, cfg.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = op.split(gate_up, 2, axis=-1)
        return self.down_proj(op.gelu(gate, approximate="tanh") * up)


class _ExpertAttnProj(nn.Module):
    """单专家的 Q/K/V/O 投影（无 bias）。"""

    def __init__(self, cfg: GemmaExpertConfig):
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.width, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.width, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.width, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.width, bias=False)


class DualExpertAttention(nn.Module):
    """双专家联合注意力：各专家独立 QKV 投影，拼接后做一次 attention。"""

    def __init__(self, configs: List[GemmaExpertConfig]):  # noqa: UP006
        self.configs = configs
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        self.head_dim = configs[0].head_dim
        self.experts = nn.ModuleList([_ExpertAttnProj(c) for c in configs])

    def forward(
        self,
        xs: List[Optional[Tensor]],  # noqa: UP006
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,  # 加性 mask [B,1,T,S]
    ) -> List[Optional[Tensor]]:  # noqa: UP006
        n, k, hd = self.num_heads, self.num_kv_heads, self.head_dim
        qs, ks, vs, lengths = [], [], [], []
        for i, x in enumerate(xs):
            if x is None:
                lengths.append(0)
                continue
            b, t, _ = x.shape
            proj = self.experts[i]
            q = op.reshape(proj.q_proj(x), (b, t, n, hd))
            kk = op.reshape(proj.k_proj(x), (b, t, k, hd))
            vv = op.reshape(proj.v_proj(x), (b, t, k, hd))
            qs.append(q)
            ks.append(kk)
            vs.append(vv)
            lengths.append(t)

        # 沿序列维拼接（对齐 openpi concatenate axis=1）
        q = op.concat(qs, dim=1) if len(qs) > 1 else qs[0]
        kc = op.concat(ks, dim=1) if len(ks) > 1 else ks[0]
        vc = op.concat(vs, dim=1) if len(vs) > 1 else vs[0]

        # RoPE
        q = _apply_rope(q, cos, sin)
        kc = _apply_rope(kc, cos, sin)
        q = q * (hd**-0.5)

        b = q.shape[0]
        tq = q.shape[1]
        ts = kc.shape[1]
        # [B,L,heads,H] -> [B,heads,L,H]
        q = op.permute_dims(q, [0, 2, 1, 3])  # [B,N,Tq,H]
        kc = op.permute_dims(kc, [0, 2, 1, 3])  # [B,K,Ts,H]
        vc = op.permute_dims(vc, [0, 2, 1, 3])  # [B,K,Ts,H]
        # GQA: K=1 时广播到 N 头
        if k != n:
            kc = op.broadcast_to(kc, (b, n, ts, hd))
            vc = op.broadcast_to(vc, (b, n, ts, hd))

        # logits = q @ k^T，float32 累加，对齐 openpi
        kt = op.permute_dims(kc, [0, 1, 3, 2])  # [B,N,H,Ts]
        logits = op.matmul(q.astype("float32"), kt.astype("float32"))  # [B,N,Tq,Ts]
        logits = logits + attn_mask.astype("float32")
        probs = op.softmax(logits, axis=-1).astype(q.dtype)

        encoded = op.matmul(probs, vc)  # [B,N,Tq,H]
        encoded = op.permute_dims(encoded, [0, 2, 1, 3])  # [B,Tq,N,H]
        encoded = op.reshape(encoded, (b, tq, n * hd))

        # 各专家在自己的序列切片上做 o_proj：按 present 专家长度沿序列维 split
        present_lengths = [lengths[i] for i, x in enumerate(xs) if x is not None]
        if len(present_lengths) > 1:
            split_points = [int(p) for p in np.cumsum(present_lengths)[:-1]]
            chunks = list(op.split(encoded, split_points, axis=1))
        else:
            chunks = [encoded]
        outs: List[Optional[Tensor]] = []  # noqa: UP006
        c = 0
        for i, x in enumerate(xs):
            if x is None:
                outs.append(None)
                continue
            outs.append(self.experts[i].o_proj(chunks[c]))
            c += 1
        return outs


class DualExpertBlock(nn.Module):
    """一层双专家 transformer block。"""

    def __init__(self, configs: List[GemmaExpertConfig], eps: float, use_adarms: List[bool]):  # noqa: UP006
        self.use_adarms = use_adarms
        self.self_attn = DualExpertAttention(configs)
        self.mlp = nn.ModuleList([GemmaExpertMLP(c) for c in configs])
        self.input_layernorm = nn.ModuleList(
            [GemmaRMSNorm(c.width, eps, use_adarms[i], c.width) for i, c in enumerate(configs)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [GemmaRMSNorm(c.width, eps, use_adarms[i], c.width) for i, c in enumerate(configs)]
        )

    def forward(
        self,
        xs: List[Optional[Tensor]],  # noqa: UP006
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
        adarms_cond: List[Optional[Tensor]],  # noqa: UP006
    ) -> List[Optional[Tensor]]:  # noqa: UP006
        # --- attention 子层 ---
        pre, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                pre.append(None)
                gates.append(None)
                continue
            normed, gate = self.input_layernorm[i](x, adarms_cond[i])
            pre.append(normed)
            gates.append(gate)
        post = self.self_attn(pre, cos, sin, attn_mask)
        xs = [
            _gated_residual(x, y, g) if x is not None else None
            for x, y, g in zip(xs, post, gates)
        ]

        # --- FFN 子层 ---
        out, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                gates.append(None)
                continue
            normed, gate = self.post_attention_layernorm[i](x, adarms_cond[i])
            out.append(self.mlp[i](normed))
            gates.append(gate)
        xs = [
            _gated_residual(x, y, g) if x is not None else None
            for x, y, g in zip(xs, out, gates)
        ]
        return xs


class DualExpertGemma(nn.Module):
    """双专家 Gemma 主干：joint forward over all layers + 末端 norm。"""

    def __init__(self, config: Pi0Config):
        self.config = config
        configs = config.experts()
        eps = config.rms_norm_eps
        # π0.5：expert-0 (PaliGemma) 普通 RMSNorm；expert-1 (action) 用 adaRMS
        self.use_adarms = [False, True] if config.pi05 else [False, False]
        self.depth = configs[0].depth
        self.layers = nn.ModuleList(
            [DualExpertBlock(configs, eps, self.use_adarms) for _ in range(self.depth)]
        )
        self.final_norm = nn.ModuleList(
            [GemmaRMSNorm(c.width, eps, self.use_adarms[i], c.width) for i, c in enumerate(configs)]
        )

    def forward(
        self,
        xs: List[Optional[Tensor]],  # noqa: UP006
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
        adarms_cond: List[Optional[Tensor]],  # noqa: UP006
    ) -> List[Optional[Tensor]]:  # noqa: UP006
        for layer in self.layers:
            xs = layer(xs, cos, sin, attn_mask, adarms_cond)
        outs: List[Optional[Tensor]] = []  # noqa: UP006
        for i, x in enumerate(xs):
            if x is None:
                outs.append(None)
                continue
            normed, _ = self.final_norm[i](x, adarms_cond[i])
            outs.append(normed)
        return outs

    def make_rope_tables(self, num_positions: int) -> Tuple[Tensor, Tensor]:
        return _rope_tables(
            num_positions, self.config.vlm.head_dim, self.config.rope_theta, self.config.dtype
        )
