"""M1 KV 路径对拍：prefill + denoise_step_kv  vs  M0 joint denoise_step。

两条路径数学等价（M1 只是把 prefix 的 K/V 预先固化），故用 **相同随机权重 + 相同输入**
即可验证 M1 重构的正确性，无需真实 checkpoint / torch 参考。

用法：
    python -m mlc_vla.compare_kv --target llvm            # CPU fp32（快）
    python -m mlc_vla.compare_kv --target cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.model.pi0 import Pi0Config
from mlc_vla.sample import euler_loop, make_time_embs

_FUNCS = ["denoise_step", "prefill", "denoise_step_kv"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(a @ b / denom)


def _unpack(ret):
    """relax VM 多输出返回可能是 tuple/Array，统一成 list。"""
    if hasattr(ret, "numpy"):
        return [ret]
    return [ret[i] for i in range(len(ret))]


def run(config: Pi0Config, target: str, loop_steps: int = 0):
    import tvm
    from tvm import relax

    ex, named_params = compile_model(config, target, functions=_FUNCS)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)

    rng = np.random.default_rng(0)
    params = []
    for _name, p in named_params:
        shape = [int(s) for s in p.shape]
        if p.dtype.startswith("int"):
            arr = np.zeros(shape, dtype=p.dtype)
        else:
            arr = (0.02 * rng.standard_normal(shape)).astype(p.dtype)
        params.append(tvm.runtime.tensor(arr, dev))

    dt = config.dtype
    prefix = tvm.runtime.tensor(
        rng.standard_normal((1, config.prefix_len, config.vlm.width)).astype(dt), dev
    )
    x_t = tvm.runtime.tensor(
        rng.standard_normal((1, config.action_horizon, config.action_dim)).astype(dt), dev
    )
    time_emb = tvm.runtime.tensor(
        rng.standard_normal((1, config.action_expert.width)).astype(dt), dev
    )

    # M0：联合 denoise
    v0 = vm["denoise_step"](prefix, x_t, time_emb, params)
    v0 = (v0.numpy() if hasattr(v0, "numpy") else v0[0].numpy())

    # M1：prefill 固化 -> suffix-only denoise
    kv = _unpack(vm["prefill"](prefix, params))
    keys, values = kv[0], kv[1]
    v1 = vm["denoise_step_kv"](keys, values, x_t, time_emb, params)
    v1 = (v1.numpy() if hasattr(v1, "numpy") else v1[0].numpy())

    cos = _cosine(v0, v1)
    max_abs = float(np.max(np.abs(v0.astype(np.float64) - v1.astype(np.float64))))
    print(f"[compare_kv] single-step shape={v0.shape} keys={tuple(int(s) for s in keys.shape)} "
          f"cosine={cos:.6f} max_abs_diff={max_abs:.3e}")
    ok = cos >= 0.99
    print(f"[compare_kv] single-step {'PASS' if ok else 'FAIL'} (threshold=0.99)")

    if loop_steps > 0:
        # 全环对拍：M1 环（prefill+denoise_kv） vs M0 环（每步重算 denoise），共享 noise/权重
        noise = rng.standard_normal((1, config.action_horizon, config.action_dim)).astype(np.float32)
        time_embs = make_time_embs(loop_steps, config.action_expert.width)
        te_dev = [tvm.runtime.tensor(te.astype(dt), dev) for te in time_embs]

        def m0_step(x_np, i):
            xd = tvm.runtime.tensor(x_np.astype(dt), dev)
            r = vm["denoise_step"](prefix, xd, te_dev[i], params)
            return (r.numpy() if hasattr(r, "numpy") else r[0].numpy())

        def m1_step(x_np, i):
            xd = tvm.runtime.tensor(x_np.astype(dt), dev)
            r = vm["denoise_step_kv"](keys, values, xd, te_dev[i], params)
            return (r.numpy() if hasattr(r, "numpy") else r[0].numpy())

        a0 = euler_loop(m0_step, noise, loop_steps)
        a1 = euler_loop(m1_step, noise, loop_steps)
        lcos = _cosine(a0, a1)
        lmax = float(np.max(np.abs(a0.astype(np.float64) - a1.astype(np.float64))))
        print(f"[compare_kv] {loop_steps}-step loop cosine={lcos:.6f} max_abs_diff={lmax:.3e}")
        lok = lcos >= 0.99
        print(f"[compare_kv] {loop_steps}-step loop {'PASS' if lok else 'FAIL'} (threshold=0.99)")
        ok = ok and lok
    return ok


def main():
    ap = argparse.ArgumentParser(description="M1 KV 路径 vs M0 联合 denoise 对拍")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--dtype", default=None, help="覆盖 config.dtype（如 bfloat16/float16/float32）")
    ap.add_argument("--dummy", action="store_true", help="小尺寸专家，快速验证 M1==M0")
    ap.add_argument("--loop", type=int, default=0, help="额外跑 N 步 Euler 全环对拍（M1 vs M0）")
    args = ap.parse_args()

    if args.dummy:
        config = Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            max_token_len=8,
            action_horizon=4,
        )
    else:
        config = Pi0Config()
    dtype = args.dtype or ("float32" if "llvm" in args.target or args.target == "c" else "bfloat16")
    config = dataclasses.replace(config, dtype=dtype)
    ok = run(config, args.target, loop_steps=args.loop)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
