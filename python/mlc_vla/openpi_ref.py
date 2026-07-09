"""openpi π0.5 单步 denoise 的 PyTorch "联合前向" 参考实现（对拍金标准）。

本模块给出与 mlc-vla ``Pi0Model.denoise_step`` 同签名的 PyTorch 参考：

    v_t = ref.denoise(prefix_embs, x_t, time_emb)

用于对拍 mlc-vla TVM 产物的数值正确性（cosine≥0.99）。

两条实现路径（自动选择）：

1. **real**（首选）：若环境装好了 openpi（含 ``transformers_replace`` 的 adaRMS 版
   ``modeling_gemma``），直接构造 ``PaliGemmaWithExpertModel`` 并调用其训练态联合前向
   ``forward([prefix, suffix], joint mask, adarms)``（openpi ``pi0_pytorch.PI0Pytorch.forward``
   的核心路径）。这是最权威的金标准。

2. **selfcontained**（回退）：当 openpi/transformers_replace 不可用时，用纯 PyTorch
   逐算子复刻 openpi 的联合前向（``gemma_pytorch.compute_layer_complete`` +
   ``modeling_gemma`` 的 ``GemmaRMSNorm`` / ``apply_rotary_pos_emb`` /
   ``eager_attention_forward``）。仅依赖 torch，直接吃真实 checkpoint 权重。

两条路径均使用真实 π0.5 权重，数学上等价于 openpi 的 KV 版推理（M0 无 KV、eager）。

与 mlc-vla 的约定一致：
- ``time_emb`` 是**已算好的正弦时间嵌入** [1, ae_width]，reference 内只跑 time MLP，
  保证双方喂入完全相同的 ``time_emb``。
- mask 为 π0.5 联合注意力：prefix 全可见、suffix 首 token 起新 block、suffix 可见 prefix+全 suffix。
- 位置编码 positions = 0..(prefix_len+suffix_len-1)（pad 全 1）。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional  # noqa: UP035

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from .model.pi0 import Pi0Config

_ATTN_NEG_INF = -2.3819763e38  # openpi big_neg


# --------------------------------------------------------------------------- #
# 输入构造工具（与 mlc-vla 对齐）
# --------------------------------------------------------------------------- #
def sinusoidal_time_emb(timestep: float, dim: int) -> np.ndarray:
    """openpi ``create_sinusoidal_pos_embedding``（min=4e-3, max=4.0），返回 [1, dim]。"""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    time = np.array([timestep], dtype=np.float64)
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_in = scaling[None, :] * time[:, None]
    return np.concatenate([np.sin(sin_in), np.cos(sin_in)], axis=1).astype(np.float32)


def build_joint_additive_mask(prefix_len: int, suffix_len: int) -> np.ndarray:
    """π0.5 联合注意力加性 mask [1,1,T,T]（与 mlc-vla ``_build_additive_mask`` 语义一致）。"""
    total = prefix_len + suffix_len
    att = np.zeros(total, dtype=np.int32)
    att[prefix_len] = 1
    cumsum = np.cumsum(att)
    allow = cumsum[None, :] <= cumsum[:, None]
    mask = np.where(allow, 0.0, _ATTN_NEG_INF).astype(np.float32)
    return mask.reshape(1, 1, total, total)


# --------------------------------------------------------------------------- #
# 自包含 PyTorch 参考（无需 openpi / transformers_replace）
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(total: int, head_dim: int, theta: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-Gemma 风格 rope 表：cos/sin [1,1,total,head_dim]（fp32 计算）。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    pos = torch.arange(total, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)  # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    cos = emb.cos().to(device=device, dtype=torch.float32)[None, None]
    sin = emb.sin().to(device=device, dtype=torch.float32)[None, None]
    return cos, sin


class _SelfContainedRef:
    """逐算子复刻 openpi 双专家联合前向的纯 torch 参考。"""

    def __init__(self, config: Pi0Config, sd: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        self.cfg = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.depth = config.vlm.depth
        self.num_heads = config.vlm.num_heads
        self.num_kv = config.vlm.num_kv_heads
        self.head_dim = config.vlm.head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        # 常驻权重：norm/modulation 用 fp32，其余用目标 dtype（对齐 openpi selective fp32）
        self._w: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            keep_fp32 = any(s in k for s in ("layernorm", "model.norm", "patch_embedding", "position_embedding"))
            tgt = torch.float32 if keep_fp32 else dtype
            self._w[k] = v.to(device=self.device, dtype=tgt)

    def _g(self, key: str) -> torch.Tensor:
        return self._w[key]

    def _rmsnorm(self, x: torch.Tensor, prefix: str, cond: Optional[torch.Tensor]):
        """对齐 modeling_gemma.GemmaRMSNorm.forward。返回 (out, gate)。"""
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if cond is None:
            w = self._g(prefix + ".weight").float()
            return (normed * (1.0 + w)).to(dtype), None
        # adaRMS：modulation = dense(cond) -> chunk(scale, shift, gate)
        mod = F.linear(cond.float(), self._g(prefix + ".dense.weight").float(), self._g(prefix + ".dense.bias").float())
        mod = mod.unsqueeze(1)  # [B,1,3*dim]
        scale, shift, gate = torch.chunk(mod, 3, dim=-1)
        out = normed * (1.0 + scale) + shift
        return out.to(dtype), gate.to(dtype)

    @staticmethod
    def _gated_residual(x, y, gate):
        return x + y if gate is None else x + y * gate

    def denoise(self, prefix_embs: np.ndarray, x_t: np.ndarray, time_emb: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dev, dt = self.device, self.dtype
        prefix_len, suffix_len = cfg.prefix_len, cfg.suffix_len
        total = prefix_len + suffix_len
        n, k, hd = self.num_heads, self.num_kv, self.head_dim

        prefix = torch.as_tensor(prefix_embs, device=dev, dtype=dt)
        x_t_t = torch.as_tensor(x_t, device=dev, dtype=dt)
        time_t = torch.as_tensor(time_emb, device=dev, dtype=dt)

        # suffix 输入 emb（action_in_proj）与 adaRMS 条件（time MLP）
        suffix = F.linear(x_t_t, self._g("action_in_proj.weight").to(dt), self._g("action_in_proj.bias").to(dt))
        tc = F.linear(time_t, self._g("time_mlp_in.weight").to(dt), self._g("time_mlp_in.bias").to(dt))
        tc = F.silu(tc)
        tc = F.linear(tc, self._g("time_mlp_out.weight").to(dt), self._g("time_mlp_out.bias").to(dt))
        adarms_cond = F.silu(tc)  # [1, ae_width]

        cos, sin = _rope_cos_sin(total, hd, cfg.rope_theta, dev, dt)
        mask = torch.as_tensor(build_joint_additive_mask(prefix_len, suffix_len), device=dev, dtype=torch.float32)

        # 源键前缀
        LM = "paligemma_with_expert.paligemma.model.language_model.layers."
        EXP = "paligemma_with_expert.gemma_expert.model.layers."
        LN_FINAL = ["paligemma_with_expert.paligemma.model.language_model.norm",
                    "paligemma_with_expert.gemma_expert.model.norm"]
        conds = [None, adarms_cond]
        layer_pfx = [LM, EXP]
        hs = [prefix, suffix]

        for L in range(self.depth):
            qs, ks, vs, gates, normed_list = [], [], [], [], []
            for e in range(2):
                pfx = f"{layer_pfx[e]}{L}."
                normed, gate = self._rmsnorm(hs[e], pfx + "input_layernorm", conds[e])
                normed_list.append(normed)
                gates.append(gate)
                b, t = normed.shape[0], normed.shape[1]
                q = F.linear(normed, self._g(pfx + "self_attn.q_proj.weight").to(dt)).view(b, t, n, hd).transpose(1, 2)
                kk = F.linear(normed, self._g(pfx + "self_attn.k_proj.weight").to(dt)).view(b, t, k, hd).transpose(1, 2)
                vv = F.linear(normed, self._g(pfx + "self_attn.v_proj.weight").to(dt)).view(b, t, k, hd).transpose(1, 2)
                qs.append(q); ks.append(kk); vs.append(vv)  # noqa: E702

            q = torch.cat(qs, dim=2)   # [B,n,total,hd]
            kc = torch.cat(ks, dim=2)  # [B,k,total,hd]
            vc = torch.cat(vs, dim=2)
            # rope（fp32 稳定），再回 dt
            qf = (q.float() * cos) + (_rotate_half(q.float()) * sin)
            kf = (kc.float() * cos) + (_rotate_half(kc.float()) * sin)
            # GQA: k=1 -> n
            if k != n:
                kf = kf.repeat_interleave(n // k, dim=1)
                vc = vc.repeat_interleave(n // k, dim=1)
            logits = torch.matmul(qf, kf.transpose(2, 3)) * self.scaling + mask  # fp32
            probs = torch.softmax(logits, dim=-1).to(dt)
            ctx = torch.matmul(probs, vc.to(dt))  # [B,n,total,hd]
            ctx = ctx.transpose(1, 2).reshape(q.shape[0], total, n * hd)  # [B,total,n*hd]

            start = 0
            new_hs = []
            for e in range(2):
                pfx = f"{layer_pfx[e]}{L}."
                t = hs[e].shape[1]
                seg = ctx[:, start:start + t]
                start += t
                o = F.linear(seg, self._g(pfx + "self_attn.o_proj.weight").to(dt))
                x1 = self._gated_residual(hs[e], o, gates[e])
                normed2, gate2 = self._rmsnorm(x1, pfx + "post_attention_layernorm", conds[e])
                gate = self._g(pfx + "mlp.gate_proj.weight").to(dt)
                up = self._g(pfx + "mlp.up_proj.weight").to(dt)
                down = self._g(pfx + "mlp.down_proj.weight").to(dt)
                mlp = F.linear(F.gelu(F.linear(normed2, gate), approximate="tanh") * F.linear(normed2, up), down)
                x2 = self._gated_residual(x1, mlp, gate2)
                new_hs.append(x2)
            hs = new_hs

        # 末端 norm（仅需 suffff = expert-1）
        suffix_out, _ = self._rmsnorm(hs[1], LN_FINAL[1], adarms_cond)
        suffix_out = suffix_out[:, -cfg.action_horizon:].float()
        v_t = F.linear(suffix_out, self._g("action_out_proj.weight").float(), self._g("action_out_proj.bias").float())
        return v_t.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# 真实 openpi 路径（需 transformers_replace）
# --------------------------------------------------------------------------- #
def _try_import_openpi(openpi_src: Optional[str]):
    """尝试导入 openpi 的 gemma_pytorch（仅需 transformers_replace，不需 flax）。"""
    if openpi_src and openpi_src not in sys.path:
        sys.path.insert(0, openpi_src)
    try:
        from transformers.models.gemma import modeling_gemma
        import inspect

        if "cond" not in inspect.signature(modeling_gemma.GemmaRMSNorm.forward).parameters:
            return None  # transformers_replace 未安装（无 adaRMS）
        from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

        return PaliGemmaWithExpertModel
    except Exception:  # noqa: BLE001
        return None


class _GemmaCfgLite:
    """PaliGemmaWithExpertModel 只读取这几个字段，避免依赖 openpi.models.gemma(flax)。"""

    def __init__(self, ge):
        self.width = ge.width
        self.depth = ge.depth
        self.mlp_dim = ge.mlp_dim
        self.num_heads = ge.num_heads
        self.num_kv_heads = ge.num_kv_heads
        self.head_dim = ge.head_dim


class _RealOpenpiRef:
    """用真实 openpi PaliGemmaWithExpertModel 做联合前向的金标准。"""

    def __init__(self, config: Pi0Config, sd, device, dtype, PWE):
        self.cfg = config
        self.device = torch.device(device)
        self.dtype = dtype
        precision = "bfloat16" if dtype == torch.bfloat16 else "float32"
        pwe = PWE(
            _GemmaCfgLite(config.vlm),
            _GemmaCfgLite(config.action_expert),
            use_adarms=[False, True] if config.pi05 else [False, False],
            precision=precision,
        )
        # 载入 paligemma_with_expert.* 权重
        sub = {k[len("paligemma_with_expert."):]: v for k, v in sd.items() if k.startswith("paligemma_with_expert.")}
        missing, unexpected = pwe.load_state_dict(sub, strict=False)
        # 忽略 vision/embed 未用键；断言核心 backbone 无缺失
        core_missing = [m for m in missing if "gemma_expert" in m or "language_model.layers" in m]
        assert not core_missing, f"real ref 缺少 backbone 权重: {core_missing[:5]}"
        self.pwe = pwe.to(self.device).eval()
        # 顶层投影
        aw = config.action_expert.width
        self.aip = torch.nn.Linear(config.action_dim, aw).to(self.device)
        self.aop = torch.nn.Linear(aw, config.action_dim).to(self.device)
        self.tmi = torch.nn.Linear(aw, aw).to(self.device)
        self.tmo = torch.nn.Linear(aw, aw).to(self.device)
        for mod, name in ((self.aip, "action_in_proj"), (self.aop, "action_out_proj"),
                          (self.tmi, "time_mlp_in"), (self.tmo, "time_mlp_out")):
            mod.weight.data = sd[f"{name}.weight"].to(self.device, torch.float32)
            mod.bias.data = sd[f"{name}.bias"].to(self.device, torch.float32)

    @torch.no_grad()
    def denoise(self, prefix_embs, x_t, time_emb):
        cfg = self.cfg
        dev = self.device
        prefix = torch.as_tensor(prefix_embs, device=dev, dtype=self.dtype)
        x_t_t = torch.as_tensor(x_t, device=dev, dtype=torch.float32)
        time_t = torch.as_tensor(time_emb, device=dev, dtype=torch.float32)

        suffix = self.aip(x_t_t)
        tc = F.silu(self.tmi(time_t))
        adarms_cond = F.silu(self.tmo(tc))

        total = cfg.prefix_len + cfg.suffix_len
        mask4d = torch.as_tensor(
            build_joint_additive_mask(cfg.prefix_len, cfg.suffix_len), device=dev, dtype=torch.float32
        )
        position_ids = torch.arange(total, device=dev)[None]

        self.pwe.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        self.pwe.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (_, suffix_out), _ = self.pwe.forward(
            attention_mask=mask4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix.to(self.dtype), suffix.to(self.dtype)],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -cfg.action_horizon:].float()
        return self.aop(suffix_out).detach().cpu().numpy()


def build_reference(
    config: Pi0Config,
    sd: Dict[str, torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    prefer_real: bool = True,
    openpi_src: Optional[str] = None,
):
    """构造 denoise 参考。返回带 ``.denoise(prefix, x_t, time_emb)`` 的对象与实现名。"""
    if prefer_real:
        PWE = _try_import_openpi(openpi_src or os.environ.get("OPENPI_SRC"))
        if PWE is not None:
            try:
                return _RealOpenpiRef(config, sd, device, dtype, PWE), "real"
            except Exception as e:  # noqa: BLE001
                print(f"[ref] real openpi 构造失败，回退自包含实现: {e}")
    return _SelfContainedRef(config, sd, device, dtype), "selfcontained"


def load_state_dict_torch(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)
