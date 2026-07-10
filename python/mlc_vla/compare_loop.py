"""图内去噪环 (denoise_loop_kv) vs 宿主去噪环 (逐步 denoise_step_kv) 数值对拍。

二者应逐算子等价（同 dtype、同 Euler 迭代），故结果应近乎 bit-identical。用于验证把
Euler 环下沉进计算图（消除每步 host↔device 往返 / IPC）未改变数值。

用法：
    python -m mlc_vla.compare_loop --target llvm            # CPU fp32（默认）
    python -m mlc_vla.compare_loop --target cuda --dtype float16
"""

from __future__ import annotations

import argparse

import numpy as np

from mlc_vla.model.pi0 import Pi0Config
from mlc_vla.sample import PiZeroRunner


def _cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run(target: str, dtype: str, steps: int, seed: int) -> bool:
    config = Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=16,
        action_horizon=4,
        dtype=dtype,
        num_denoise_steps=steps,
    )
    runner = PiZeroRunner(config, target)

    rng = np.random.default_rng(seed)
    arrays = []
    for _name, p in runner.named_params:
        shape = [int(s) for s in p.shape]
        if p.dtype.startswith(("int", "uint")):
            arrays.append(np.zeros(shape, p.dtype))
        else:
            arrays.append((0.02 * rng.standard_normal(shape)).astype(p.dtype))
    params = runner.to_params(arrays)

    prefix = (0.02 * rng.standard_normal((1, config.prefix_len, config.vlm.width))).astype("float32")
    noise = rng.standard_normal((1, config.action_horizon, config.action_dim)).astype("float32")

    x_host = runner.sample(params, prefix, noise=noise, num_steps=steps)
    x_graph = runner.sample_graph(params, prefix, noise=noise)

    cos = _cosine(x_host, x_graph)
    max_abs = float(np.max(np.abs(x_host - x_graph)))
    print(f"[compare_loop] steps={steps} dtype={dtype}  cosine={cos:.6f} max_abs_diff={max_abs:.3e}")
    # fp32 应几乎 bit-identical；fp16/bf16 允许极小舍入差
    thr = 0.99999 if dtype == "float32" else 0.9999
    ok = cos >= thr
    print(f"[compare_loop] {'PASS' if ok else 'FAIL'} (图内环 == 宿主环)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="in-graph Euler loop vs host loop 对拍")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(0 if run(args.target, args.dtype, args.steps, args.seed) else 1)


if __name__ == "__main__":
    main()
