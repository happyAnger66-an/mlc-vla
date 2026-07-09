"""M1 测速：prefill / denoise_step_kv(M1) / denoise_step(M0) 单步延迟 + 全环对比。

CUDA 目标下 TVM 默认 pipeline 已含 ``RewriteCUDAGraph``，denoise_step_kv 的静态区
会被捕获为 CUDA Graph（一次 launch 重放多个 kernel），显著降低 host 端 launch 开销。

用法：
    python -m mlc_vla.bench_kv --target cuda --dtype bfloat16 --steps 10 --iters 30
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.model.pi0 import Pi0Config

_FUNCS = ["denoise_step", "prefill", "denoise_step_kv"]


def _unpack(ret):
    if hasattr(ret, "numpy"):
        return [ret]
    return [ret[i] for i in range(len(ret))]


def _bench(fn, iters: int, sync) -> float:
    for _ in range(3):  # warmup
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def run(config: Pi0Config, target: str, steps: int, iters: int, cuda_graph: bool = False):
    import tvm
    from tvm import relax

    ex, named_params = compile_model(config, target, functions=_FUNCS, cuda_graph=cuda_graph)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    sync = (lambda: dev.sync()) if hasattr(dev, "sync") else (lambda: None)
    print(f"[bench] cuda_graph requested={cuda_graph}")

    rng = np.random.default_rng(0)
    params = [
        tvm.runtime.tensor(
            np.zeros([int(s) for s in p.shape], p.dtype) if p.dtype.startswith("int")
            else (0.02 * rng.standard_normal([int(s) for s in p.shape])).astype(p.dtype), dev)
        for _n, p in named_params
    ]
    dt = config.dtype
    prefix = tvm.runtime.tensor(
        rng.standard_normal((1, config.prefix_len, config.vlm.width)).astype(dt), dev)
    x_t = tvm.runtime.tensor(
        rng.standard_normal((1, config.action_horizon, config.action_dim)).astype(dt), dev)
    time_emb = tvm.runtime.tensor(
        rng.standard_normal((1, config.action_expert.width)).astype(dt), dev)

    keys, values = _unpack(vm["prefill"](prefix, params))

    t_prefill = _bench(lambda: vm["prefill"](prefix, params), iters, sync)
    t_m1 = _bench(lambda: vm["denoise_step_kv"](keys, values, x_t, time_emb, params), iters, sync)
    t_m0 = _bench(lambda: vm["denoise_step"](prefix, x_t, time_emb, params), iters, sync)

    total_m1 = t_prefill + steps * t_m1
    total_m0 = steps * t_m0
    print(f"[bench] prefill            : {t_prefill:8.3f} ms")
    print(f"[bench] denoise_step_kv(M1): {t_m1:8.3f} ms/step")
    print(f"[bench] denoise_step   (M0): {t_m0:8.3f} ms/step")
    print(f"[bench] {steps}-step total  M1 (prefill+{steps}x): {total_m1:8.3f} ms")
    print(f"[bench] {steps}-step total  M0 ({steps}x joint)   : {total_m0:8.3f} ms")
    print(f"[bench] per-step speedup M0/M1 = {t_m0 / max(t_m1, 1e-6):.2f}x ; "
          f"end-to-end M0/M1 = {total_m0 / max(total_m1, 1e-6):.2f}x")


def run_quant(config: Pi0Config, target: str, quant_name: str, steps: int, iters: int,
              cuda_graph: bool = False):
    """量化 M1 路径测速（随机权重；延迟与权重值无关）。"""
    import tvm
    from tvm import relax

    from mlc_vla.compile_quant import compile_model_quant

    kv_funcs = ["prefill", "denoise_step_kv"]
    ex, q_named_params, _qmap, quant = compile_model_quant(
        config, target, kv_funcs, quant_name, cuda_graph=cuda_graph)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    sync = (lambda: dev.sync()) if hasattr(dev, "sync") else (lambda: None)

    rng = np.random.default_rng(0)
    params, q_bytes = [], 0
    for _n, p in q_named_params:
        shape = [int(s) for s in p.shape]
        if p.dtype.startswith(("int", "uint")):
            arr = rng.integers(0, 7, size=shape, dtype=np.uint32).astype(p.dtype)
        else:
            arr = (0.02 * rng.standard_normal(shape)).astype(p.dtype)
        q_bytes += arr.nbytes
        params.append(tvm.runtime.tensor(arr, dev))

    dt = config.dtype
    prefix = tvm.runtime.tensor(rng.standard_normal((1, config.prefix_len, config.vlm.width)).astype(dt), dev)
    x_t = tvm.runtime.tensor(rng.standard_normal((1, config.action_horizon, config.action_dim)).astype(dt), dev)
    time_emb = tvm.runtime.tensor(rng.standard_normal((1, config.action_expert.width)).astype(dt), dev)
    keys, values = _unpack(vm["prefill"](prefix, params))

    t_prefill = _bench(lambda: vm["prefill"](prefix, params), iters, sync)
    t_step = _bench(lambda: vm["denoise_step_kv"](keys, values, x_t, time_emb, params), iters, sync)
    print(f"[bench-quant] quant={quant_name} param bytes={q_bytes/1e6:.1f}MB")
    print(f"[bench-quant] prefill            : {t_prefill:8.3f} ms")
    print(f"[bench-quant] denoise_step_kv    : {t_step:8.3f} ms/step")
    print(f"[bench-quant] {steps}-step total  : {t_prefill + steps * t_step:8.3f} ms")


def main():
    ap = argparse.ArgumentParser(description="M1 KV 路径测速")
    ap.add_argument("--target", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cuda-graph", action="store_true", help="开启 RewriteCUDAGraph 捕获去噪步")
    ap.add_argument("--quant", default=None, help="量化预设（如 q4bf16_1）；给定则测量化 M1 路径")
    args = ap.parse_args()

    if args.quant:
        from mlc_vla.quant import get_quant

        dtype = args.dtype or get_quant(args.quant).model_dtype
        config = dataclasses.replace(Pi0Config(), dtype=dtype)
        run_quant(config, args.target, args.quant, args.steps, args.iters, cuda_graph=args.cuda_graph)
        return

    dtype = args.dtype or ("float32" if "llvm" in args.target or args.target == "c" else "bfloat16")
    config = dataclasses.replace(Pi0Config(), dtype=dtype)
    run(config, args.target, args.steps, args.iters, cuda_graph=args.cuda_graph)


if __name__ == "__main__":
    main()
