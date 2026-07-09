"""prefix padding 正确性自洽对拍：验证 prefill/denoise 的 pad mask 真的屏蔽了 padded token。

思路（无需 openpi 参考）：
- 造一段有效长度 V 的 prefix，右侧补 ``pad`` 个 **随机垃圾** token（总长 P=V+pad）。
- 路径 A（padded）：prefix_len=P 的 engine，prefix_mask 标记前 V 有效，suffix RoPE offset=V。
- 路径 B（truncated）：prefix_len=V 的 engine（无 padding），offset=V。
两条路径权重完全相同（prefix_len 不参与任何权重）。若 pad mask 正确屏蔽了垃圾 token，
则 v_A ≈ v_B（cosine≈1）。若不屏蔽，垃圾 token 会污染 K/V，cosine 显著下降。

用法：
    python -m mlc_vla.compare_pad --target llvm       # CPU fp32（默认）
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.model.pi0 import Pi0Config
from mlc_vla.sample import make_prefix_mask_np, make_rope_np

_FUNCS = ["prefill", "denoise_step_kv"]


def _cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _unpack(ret):
    return [ret] if hasattr(ret, "numpy") else [ret[i] for i in range(len(ret))]


def _run(config, target, src, prefix_np, num_valid):
    import tvm
    from tvm import relax

    dev = _device_for(target)
    ex, npar = compile_model(config, target, functions=_FUNCS)
    vm = relax.VirtualMachine(ex, dev)
    dt = config.dtype
    tp = [tvm.runtime.tensor(src[name].astype(p.dtype), dev) for name, p in npar]

    pmask = tvm.runtime.tensor(make_prefix_mask_np(config.prefix_len, num_valid=num_valid), dev)
    cos_np, sin_np = make_rope_np(config.suffix_len, config.vlm.head_dim, config.rope_theta, offset=num_valid)
    scos, ssin = tvm.runtime.tensor(cos_np, dev), tvm.runtime.tensor(sin_np, dev)

    x_t = tvm.runtime.tensor(np.zeros((1, config.action_horizon, config.action_dim), dt), dev)
    from mlc_vla.openpi_ref import sinusoidal_time_emb
    te = tvm.runtime.tensor(sinusoidal_time_emb(0.7, config.action_expert.width).astype(dt), dev)

    kv = _unpack(vm["prefill"](tvm.runtime.tensor(prefix_np.astype(dt), dev), pmask, tp))
    keys, values = kv[0], kv[1]
    v = vm["denoise_step_kv"](keys, values, x_t, te, scos, ssin, pmask, tp)
    return (v.numpy() if hasattr(v, "numpy") else v[0].numpy()).astype("float32")


def run(target: str, pad: int, seed: int):
    # dummy 小模型；两 config 只在 max_token_len 上差 pad（=> prefix_len 差 pad）
    base = dict(paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, dtype="float32")
    cfg_full = dataclasses.replace(Pi0Config(**base, max_token_len=16))
    cfg_trunc = dataclasses.replace(Pi0Config(**base, max_token_len=16 - pad))
    P, V = cfg_full.prefix_len, cfg_trunc.prefix_len
    assert P - V == pad
    W = cfg_full.vlm.width

    rng = np.random.default_rng(seed)
    # 共享权重（两 config 权重 shape 完全一致）
    from mlc_vla.compile_quant import fp_named_params
    npar = fp_named_params(cfg_full, _FUNCS)
    src = {name: (np.zeros([int(s) for s in p.shape], p.dtype) if p.dtype.startswith(("int", "uint"))
                  else (0.02 * rng.standard_normal([int(s) for s in p.shape])).astype("float32"))
           for name, p in npar}

    prefix_full = (0.02 * rng.standard_normal((1, P, W))).astype("float32")
    # padded 区（后 pad 个）填 **大随机垃圾**，若未被屏蔽会显著污染结果
    prefix_full[:, V:, :] = 10.0 * rng.standard_normal((1, pad, W)).astype("float32")

    v_full = _run(cfg_full, target, src, prefix_full, num_valid=V)
    v_trunc = _run(cfg_trunc, target, src, prefix_full[:, :V, :].copy(), num_valid=V)

    cos = _cosine(v_full, v_trunc)
    max_abs = float(np.max(np.abs(v_full - v_trunc)))
    print(f"[compare_pad] P={P} V={V} pad={pad}  cosine={cos:.6f} max_abs_diff={max_abs:.3e}")
    ok = cos >= 0.9999
    print(f"[compare_pad] {'PASS' if ok else 'FAIL'} (padded 垃圾 token 被 pad mask 正确屏蔽)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="prefix padding 屏蔽正确性自洽对拍")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--pad", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(0 if run(args.target, args.pad, args.seed) else 1)


if __name__ == "__main__":
    main()
