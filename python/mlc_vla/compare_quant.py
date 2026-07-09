"""量化对拍：group-quant 的 prefill+denoise_step_kv vs fp（同一份源权重）。

同一份 fp 源权重分别 (a) 直接编译运行、(b) 量化后编译运行，比较 v_t 的 cosine，
量化误差应使 cosine 仍 ≥ 阈值（int4 group 一般 ≥0.98）。

用法：
    python -m mlc_vla.compare_quant --target cuda --quant q4bf16_1 --dummy   # 随机权重结构验证
    python -m mlc_vla.compare_quant --target cuda --quant q4bf16_1 \
        --ckpt /path/to/model.safetensors                                    # 真实权重
"""

from __future__ import annotations

import argparse
import dataclasses
import gc

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.compile_quant import compile_model_quant, fp_named_params, quantize_params
from mlc_vla.model.pi0 import Pi0Config
from mlc_vla.model.pi0 import pi0_loader

_FUNCS = ["prefill", "denoise_step_kv"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _unpack(ret):
    return [ret] if hasattr(ret, "numpy") else [ret[i] for i in range(len(ret))]


def _random_src(fp_np, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    src = {}
    for name, p in fp_np:
        shape = [int(s) for s in p.shape]
        if p.dtype.startswith("int"):
            src[name] = np.zeros(shape, dtype=p.dtype)
        else:
            src[name] = (0.02 * rng.standard_normal(shape)).astype("float32")
    return src


def _run_kv(ex, named_params, params_dict, prefix, x_t, time_emb, dt, dev):
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(ex, dev)
    tp = [tvm.runtime.tensor(params_dict[name], dev) for name, _ in named_params]
    kv = _unpack(vm["prefill"](tvm.runtime.tensor(prefix.astype(dt), dev), tp))
    keys, values = kv[0], kv[1]
    v = vm["denoise_step_kv"](
        keys, values,
        tvm.runtime.tensor(x_t.astype(dt), dev),
        tvm.runtime.tensor(time_emb.astype(dt), dev),
        tp,
    )
    out = v.numpy() if hasattr(v, "numpy") else v[0].numpy()
    return out.astype("float32")


def run(config: Pi0Config, target: str, quant_name: str, seed: int, ckpt: str | None):
    from mlc_vla.openpi_ref import sinusoidal_time_emb

    dev = _device_for(target)
    dt = config.dtype

    fp_np = fp_named_params(config, _FUNCS)
    if ckpt:
        raw = pi0_loader.load_safetensors(ckpt, dtype="float32")
        src_fp = pi0_loader.load_params(config, raw, named_params=fp_np, dtype="float32")
        del raw
    else:
        src_fp = _random_src(fp_np, seed)

    rng = np.random.default_rng(seed + 7)
    prefix = (0.02 * rng.standard_normal((1, config.prefix_len, config.vlm.width))).astype("float32")
    x_t = rng.standard_normal((1, config.action_horizon, config.action_dim)).astype("float32")
    time_emb = sinusoidal_time_emb(0.7, config.action_expert.width).astype("float32")

    # ---- fp 路径 ----
    ex_fp, np_fp = compile_model(config, target, functions=_FUNCS)
    fp_params = {name: src_fp[name].astype(p.dtype) for name, p in np_fp}
    v_fp = _run_kv(ex_fp, np_fp, fp_params, prefix, x_t, time_emb, dt, dev)
    del ex_fp
    gc.collect()

    # ---- 量化路径 ----
    ex_q, np_q, qmap, quant = compile_model_quant(config, target, _FUNCS, quant_name)
    q_params = quantize_params(quant, src_fp, np_q, qmap, dev)
    v_q = _run_kv(ex_q, np_q, q_params, prefix, x_t, time_emb, dt, dev)

    # ---- 字节数对比 ----
    fp_bytes = sum(a.nbytes for a in fp_params.values())
    q_bytes = sum(a.nbytes for a in q_params.values())
    cos = _cosine(v_fp, v_q)
    max_abs = float(np.max(np.abs(v_fp.astype(np.float64) - v_q.astype(np.float64))))
    print(f"[compare_quant] quant={quant_name} shape={v_fp.shape} cosine={cos:.6f} max_abs_diff={max_abs:.3e}")
    print(f"[compare_quant] param bytes: fp={fp_bytes/1e6:.1f}MB  quant={q_bytes/1e6:.1f}MB  "
          f"ratio={fp_bytes/max(q_bytes,1):.2f}x")
    ok = cos >= 0.97
    print(f"[compare_quant] {'PASS' if ok else 'FAIL'} (threshold=0.97)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="pi0 group 量化 vs fp 对拍")
    ap.add_argument("--target", default="cuda")
    ap.add_argument("--quant", default="q4bf16_1")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    from mlc_vla.quant import get_quant

    # config.dtype 必须与量化预设的 model_dtype 一致（KV/激活 dtype 与反量化计算 dtype 对齐）
    dtype = args.dtype or get_quant(args.quant).model_dtype
    if args.dummy:
        config = Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy",
                           max_token_len=8, action_horizon=4)
        config = dataclasses.replace(config, dtype=dtype)
    else:
        config = dataclasses.replace(Pi0Config(), dtype=dtype)
    ok = run(config, args.target, args.quant, args.seed, args.ckpt)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
