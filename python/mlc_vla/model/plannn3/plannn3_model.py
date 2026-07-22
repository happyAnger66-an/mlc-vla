"""plannn3 顶层模型（M0：GPT 主干 + KV-cache prefill/decode）。

对齐 ``network.py`` 的 GPT 主干：
- ``CausalSelfAttention``：c_attn(3*n_embd) / c_proj，interleaved RoPE（偶/奇通道交错旋转）
- ``MLP``：c_fc(4*n_embd) -> GELU -> c_proj
- ``Block``：x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))（LayerNorm 无 bias）
- ``TrajHead``：LayerNorm + Linear -> logits(vocab)

M0 把模型拆成三个可导出函数（照搬 mlc-vla 三段图骨架，固定 shape）：
- ``embed_token``  : 单个 traj token id -> embedding（解码侧 ids_to_embed 的替身）
- ``prefill``      : prompt token embeds -> 首步 logits + 每层 KV（padding 到 max_seq）
- ``decode_step``  : 单 token embed + KV + 运行时量(cos/sin/mask/onehot) -> logits + 更新后的 KV

变长解码 → 固定 shape：KV 为预分配 ``[n_layer,1,max_seq,2*n_embd]`` 缓冲，新 token 用
``write_onehot`` 以「乘加」写入指定槽位（避免 where/动态 shape）；有效长度、RoPE 位置、
注意力 mask 全部由宿主按 ``valid_kv_len`` 算好后作为运行时张量传入，保证 trace 稳定。
"""

from __future__ import annotations

import math
from typing import List, Optional  # noqa: UP035

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .plannn3_config import Plannn3Config


# --------------------------------------------------------------------------- #
# RoPE（interleaved / 偶奇交错，对齐 network.apply_rotary_emb_interleaved）
# --------------------------------------------------------------------------- #
def _rope_tables_np(num_positions: int, head_dim: int, theta: float, offset: int = 0):
    """生成 cos/sin 常量，形状 ``[1,1,L,head_dim//2]``（恒 fp32）。

    对齐 network.py：``inv_freq = 1/theta**(arange(0,H,2)/H)``；``freqs = outer(pos, inv_freq)``。
    """
    half = head_dim // 2
    positions = np.arange(offset, offset + num_positions, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))  # [half]
    freqs = np.outer(positions, inv_freq)  # [L, half]
    cos = np.cos(freqs).reshape(1, 1, num_positions, half).astype("float32")
    sin = np.sin(freqs).reshape(1, 1, num_positions, half).astype("float32")
    return cos, sin


def _apply_rope_interleaved(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: ``[b, n_head, T, head_dim]``；cos/sin: ``[1,1,T,head_dim//2]``（fp32）。

    偶/奇通道交错旋转：y[2i]=x[2i]*cos-x[2i+1]*sin, y[2i+1]=x[2i]*sin+x[2i+1]*cos。
    返回 fp32（rope 后 q/k 直接进 fp32 logits matmul）。
    """
    b, nh, t, hd = x.shape
    half = hd // 2
    xr = op.reshape(x, (b, nh, t, half, 2))
    x1, x2 = op.split(xr, 2, axis=-1)  # each [b,nh,t,half,1]
    xf1 = x1.astype("float32")
    xf2 = x2.astype("float32")
    c = op.reshape(cos, (1, 1, t, half, 1))
    s = op.reshape(sin, (1, 1, t, half, 1))
    y1 = xf1 * c - xf2 * s
    y2 = xf1 * s + xf2 * c
    y = op.concat([y1, y2], dim=-1)  # [b,nh,t,half,2] -> 交错
    return op.reshape(y, (b, nh, t, hd))


# --------------------------------------------------------------------------- #
# 子模块
# --------------------------------------------------------------------------- #
class LayerNormNoBias(nn.Module):
    """LayerNorm（仅 weight，无 bias），fp32 内部计算，对齐 network.LayerNorm(bias=False)。"""

    def __init__(self, dim: int, eps: float):
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter((dim,))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.astype("float32")
        mean = op.sum(xf, axis=-1, keepdims=True) / float(self.dim)
        xc = xf - mean
        var = op.sum(op.square(xc), axis=-1, keepdims=True) / float(self.dim)
        normed = op.divide(xc, op.sqrt(var + self.eps))
        out = normed * self.weight.astype("float32")
        return out.astype(x.dtype)


class MLP(nn.Module):
    def __init__(self, cfg: Plannn3Config):
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(op.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Plannn3Config):
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def qkv(self, x: Tensor, cos: Tensor, sin: Tensor):
        """x: [b,T,n_embd] -> q,k(fp32,含 rope), v；均为 [b,n_head,T,head_dim]。"""
        b, t, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = op.split(qkv, 3, axis=-1)  # each [b,t,n_embd]
        q = op.permute_dims(op.reshape(q, (b, t, self.n_head, self.head_dim)), [0, 2, 1, 3])
        k = op.permute_dims(op.reshape(k, (b, t, self.n_head, self.head_dim)), [0, 2, 1, 3])
        v = op.permute_dims(op.reshape(v, (b, t, self.n_head, self.head_dim)), [0, 2, 1, 3])
        q = _apply_rope_interleaved(q, cos, sin)
        k = _apply_rope_interleaved(k, cos, sin)
        return q, k, v

    def sdpa(self, q: Tensor, k: Tensor, v: Tensor, add_mask: Tensor) -> Tensor:
        """q,k,v: [b,n_head,Tq/Tk,head_dim]（q/k fp32）；add_mask 可广播到 [b,n_head,Tq,Tk]。

        返回合并头后的 [b,Tq,n_embd]。QK^T 与 softmax 全程 fp32 累加。
        """
        b = q.shape[0]
        tq = q.shape[2]
        kf = k.astype("float32")
        vf = v.astype("float32")
        kt = op.permute_dims(kf, [0, 1, 3, 2])  # [b,n_head,head_dim,Tk]
        logits = op.matmul(q.astype("float32"), kt, out_dtype="float32") * self.scale
        logits = logits + add_mask.astype("float32")
        probs = op.softmax(logits, axis=-1)
        ctx = op.matmul(probs, vf)  # [b,n_head,Tq,head_dim]
        ctx = op.permute_dims(ctx, [0, 2, 1, 3])  # [b,Tq,n_head,head_dim]
        return op.reshape(ctx, (b, tq, self.n_embd))


class Block(nn.Module):
    def __init__(self, cfg: Plannn3Config):
        self.ln_1 = LayerNormNoBias(cfg.n_embd, cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNormNoBias(cfg.n_embd, cfg.layer_norm_eps)
        self.mlp = MLP(cfg)


class TrajHead(nn.Module):
    """轨迹 token 输出头：LayerNorm + Linear -> logits（对齐 trajectory_head.TrajHead）。"""

    def __init__(self, cfg: Plannn3Config):
        self.ln = LayerNormNoBias(cfg.n_embd, cfg.layer_norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.ln(x)).astype("float32")


# --------------------------------------------------------------------------- #
# 顶层模型
# --------------------------------------------------------------------------- #
class Plannn3Model(nn.Module):
    def __init__(self, cfg: Plannn3Config):
        self.config = cfg
        self.dtype = cfg.dtype
        self.n_layer = cfg.n_layer
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.vocab_size = cfg.vocab_size

        # traj token embedding（encode 侧 ids_to_embed 的 M0 替身）
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.traj_head = TrajHead(cfg)

        # 编译期 bake：prefill 的 RoPE 表 + causal mask + 末 token 选择器 + padding 零块
        cos, sin = _rope_tables_np(cfg.prompt_len, cfg.head_dim, cfg.rope_theta)
        self._prefill_cos = cos
        self._prefill_sin = sin
        causal = np.triu(np.ones((cfg.prompt_len, cfg.prompt_len), "float32"), k=1) * cfg.attn_neg_inf
        self._causal_mask = causal.reshape(1, 1, cfg.prompt_len, cfg.prompt_len)
        last_sel = np.zeros((1, 1, cfg.prompt_len), "float32")
        last_sel[0, 0, cfg.prompt_len - 1] = 1.0  # one-hot 取末 token（matmul 选择，避免 strided_slice）
        self._last_sel = last_sel
        pad_len = cfg.max_seq_len - cfg.prompt_len
        self._kv_pad = np.zeros((1, pad_len, 2 * cfg.n_embd), cfg.dtype)

    # ---------- 图内 argmax（贪心采样，对齐 infer.py 的 torch.argmax 以做 bit-exact 对拍）----------
    @staticmethod
    def _argmax_last(logits: Tensor) -> Tensor:
        """logits [1,1,vocab] fp32 -> 末维 argmax 的 id [1,1] int32（图内计算）。

        nn 前端未暴露 argmax，用 ``tensor_expr_op`` + ``topi.argmax`` 下沉为 TE，
        使整段解码环可留在图内（配合 CUDA Graph 整环捕获）。
        """
        def _te_argmax(x):
            from tvm import topi  # noqa: PLC0415

            return topi.argmax(x, axis=-1, keepdims=False)  # [1,1]

        idx = op.tensor_expr_op(_te_argmax, "argmax", [logits])
        return op.astype(idx, "int32")

    # ---------- 可导出函数 ----------
    def embed_token(self, input_ids: Tensor) -> Tensor:
        """单个 traj token id [1,1] -> embedding [1,1,n_embd]。"""
        return self.embed(input_ids)

    def prefill(self, token_embeds: Tensor):
        """token_embeds [1,prompt_len,n_embd] -> (logits [1,1,vocab], kv [n_layer,1,max_seq,2*n_embd])。"""
        cfg = self.config
        b = 1
        t = cfg.prompt_len
        cos = nn.Tensor.from_const(self._prefill_cos)
        sin = nn.Tensor.from_const(self._prefill_sin)
        add_mask = nn.Tensor.from_const(self._causal_mask)
        kv_pad = nn.Tensor.from_const(self._kv_pad)

        x = token_embeds.astype(self.dtype)
        kv_layers: List[Tensor] = []
        for blk in self.h:
            q, k, v = blk.attn.qkv(blk.ln_1(x), cos, sin)  # [b,n_head,t,head_dim]
            ctx = blk.attn.sdpa(q, k, v, add_mask)  # [b,t,n_embd]
            x = x + blk.attn.c_proj(ctx.astype(self.dtype))
            x = x + blk.mlp(blk.ln_2(x))
            # pack k(post-rope)/v 到 [b,t,2*n_embd]，再 padding 到 max_seq
            k_flat = op.reshape(op.permute_dims(k, [0, 2, 1, 3]), (b, t, self.n_embd))
            v_flat = op.reshape(op.permute_dims(v, [0, 2, 1, 3]), (b, t, self.n_embd))
            packed = op.concat([k_flat.astype(self.dtype), v_flat.astype(self.dtype)], dim=-1)
            full = op.concat([packed, kv_pad], dim=1)  # [b,max_seq,2*n_embd]
            kv_layers.append(op.reshape(full, (1, 1, cfg.max_seq_len, 2 * self.n_embd)))

        kv = op.concat(kv_layers, dim=0) if self.n_layer > 1 else kv_layers[0]
        # 取末 token 的 logits：sel[1,1,t] @ x[1,t,n_embd] -> [1,1,n_embd]
        sel = nn.Tensor.from_const(self._last_sel)
        last = op.matmul(sel.astype(self.dtype), x)  # [1,1,n_embd]
        logits = self.traj_head(last)  # [1,1,vocab]
        return logits, kv

    def decode_step(
        self,
        latest_embed: Tensor,
        step_cos: Tensor,
        step_sin: Tensor,
        add_mask: Tensor,
        write_onehot: Tensor,
        kv: Tensor,
    ):
        """单步自回归解码。

        - ``latest_embed`` : [1,1,n_embd]，上一步 token 的 embedding（宿主已做 argmax+embed）
        - ``step_cos/sin`` : [1,1,1,head_dim//2]，当前写入位置的 RoPE（宿主按 valid_kv_len 算）
        - ``add_mask``     : [1,1,1,max_seq]，<=pos 为 0，其余大负数
        - ``write_onehot`` : [1,max_seq,1]，pos 处为 1，其余 0（把新 k/v 写入缓冲槽位）
        - ``kv``           : [n_layer,1,max_seq,2*n_embd]

        返回 (logits [1,1,vocab], kv_new [同上])。
        """
        cfg = self.config
        b = 1
        max_seq = cfg.max_seq_len
        inv_onehot = write_onehot * (-1.0) + 1.0  # 1 - onehot（避免 float.__rsub__）

        kv_layers = list(op.split(kv, self.n_layer, axis=0)) if self.n_layer > 1 else [kv]
        x = latest_embed.astype(self.dtype)
        new_layers: List[Tensor] = []
        for i, blk in enumerate(self.h):
            packed = op.reshape(kv_layers[i], (b, max_seq, 2 * self.n_embd))
            past_k, past_v = op.split(packed, 2, axis=-1)  # each [b,max_seq,n_embd]

            q, k, v = blk.attn.qkv(blk.ln_1(x), step_cos, step_sin)  # [b,n_head,1,head_dim]
            k_out = op.reshape(op.permute_dims(k, [0, 2, 1, 3]), (b, 1, self.n_embd)).astype(self.dtype)
            v_out = op.reshape(op.permute_dims(v, [0, 2, 1, 3]), (b, 1, self.n_embd)).astype(self.dtype)
            # 乘加写入：pos 槽取新值，其余保留旧值（k_out[b,1,c]*onehot[1,max_seq,1] 广播）
            k_full = past_k * inv_onehot + k_out * write_onehot
            v_full = past_v * inv_onehot + v_out * write_onehot

            k_all = op.permute_dims(
                op.reshape(k_full, (b, max_seq, self.n_head, self.head_dim)), [0, 2, 1, 3]
            )
            v_all = op.permute_dims(
                op.reshape(v_full, (b, max_seq, self.n_head, self.head_dim)), [0, 2, 1, 3]
            )
            ctx = blk.attn.sdpa(q, k_all, v_all, add_mask)  # [b,1,n_embd]
            x = x + blk.attn.c_proj(ctx.astype(self.dtype))
            x = x + blk.mlp(blk.ln_2(x))

            new_packed = op.concat([k_full, v_full], dim=-1)  # [b,max_seq,2*n_embd]
            new_layers.append(op.reshape(new_packed, (1, 1, max_seq, 2 * self.n_embd)))

        kv_new = op.concat(new_layers, dim=0) if self.n_layer > 1 else new_layers[0]
        logits = self.traj_head(x)  # [1,1,vocab]
        return logits, kv_new

    def decode_loop_kv(self, token_embeds: Tensor) -> Tensor:
        """图内整段自回归解码：prefill + 固定 ``pred_times-1`` 步 AR 环，返回 traj id [1,pred_times]。

        把宿主 ``Plannn3Runner.generate`` 的整个环下沉进计算图（对齐 mlc-vla ``denoise_loop_kv``）：
        - 逐步的 RoPE cos/sin、注意力 add_mask、写入 onehot 均在编译期 bake 为常量
          （位置 ``pos=prompt_len+step`` 完全确定，无需宿主传入）；
        - argmax 与 embedding 查表在图内完成，消除每步 host↔device 往返，便于整段 CUDA Graph 捕获；
        - 直接复用 ``prefill`` / ``decode_step`` 方法体，数值上与宿主逐步环逐算子一致。

        返回离散轨迹 token id ``[1, pred_times]`` (int32)，宿主再走 PCA 反解得到 waypoints。
        """
        cfg = self.config
        max_seq = cfg.max_seq_len

        logits, kv = self.prefill(token_embeds)
        cur = self._argmax_last(logits)  # [1,1] int32
        ids: List[Tensor] = [cur]

        idx = np.arange(max_seq)
        for step in range(cfg.pred_times - 1):
            pos = cfg.prompt_len + step
            emb = self.embed_token(cur)  # [1,1,n_embd]
            cos, sin = _rope_tables_np(1, cfg.head_dim, cfg.rope_theta, offset=pos)
            add = np.where(idx <= pos, 0.0, cfg.attn_neg_inf).astype("float32").reshape(1, 1, 1, max_seq)
            onehot = (idx == pos).astype(cfg.dtype).reshape(1, max_seq, 1)
            logits, kv = self.decode_step(
                emb,
                nn.Tensor.from_const(cos),
                nn.Tensor.from_const(sin),
                nn.Tensor.from_const(add),
                nn.Tensor.from_const(onehot),
                kv,
            )
            cur = self._argmax_last(logits)
            ids.append(cur)

        return op.concat(ids, dim=1)  # [1, pred_times]

    # ---------- 导出 spec ----------
    def get_default_spec(self, functions=None):
        cfg = self.config
        half = cfg.head_dim // 2
        kv_shape = [cfg.n_layer, 1, cfg.max_seq_len, 2 * cfg.n_embd]
        mod_spec = {
            "embed_token": {
                "input_ids": nn.spec.Tensor([1, 1], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "token_embeds": nn.spec.Tensor([1, cfg.prompt_len, cfg.n_embd], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode_step": {
                "latest_embed": nn.spec.Tensor([1, 1, cfg.n_embd], self.dtype),
                "step_cos": nn.spec.Tensor([1, 1, 1, half], "float32"),
                "step_sin": nn.spec.Tensor([1, 1, 1, half], "float32"),
                "add_mask": nn.spec.Tensor([1, 1, 1, cfg.max_seq_len], "float32"),
                "write_onehot": nn.spec.Tensor([1, cfg.max_seq_len, 1], self.dtype),
                "kv": nn.spec.Tensor(kv_shape, self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode_loop_kv": {
                "token_embeds": nn.spec.Tensor([1, cfg.prompt_len, cfg.n_embd], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        }
        if functions is not None:
            mod_spec = {k: mod_spec[k] for k in functions}
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
