"""与 openpi 单步 v_t 数值对拍（M0 gate）。

同一组共享随机输入（prefix_embs / x_t / time_emb）分别喂给：
- openpi PyTorch 参考（``openpi_ref.build_reference``，真实/自包含二选一）
- mlc-vla 编译产物的 ``denoise_step``
比较 cosine 相似度，≥0.99 为通过。

两档：
- 档 A（结构对拍）：双方用**同一份随机权重**（openpi-key 源 sd 同步给 loader 与 ref），
  验证计算图逻辑一致（cosine≈1）。不需真实 checkpoint。
- 档 B（权重对拍）：加载真实 π0.5 checkpoint，端到端 cosine gate。

用法：
    # 档 A（CPU dummy 小尺寸，秒级验证图逻辑）
    python -m mlc_vla.compare --mode A --dummy --target llvm
    # 档 B（真实权重，CUDA，fp16）
    python -m mlc_vla.compare --mode B --target cuda --dtype float16 \
        --ckpt /path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import gc

import numpy as np

from mlc_vla.model.pi0 import Pi0Config, Pi0Model
from mlc_vla.model.pi0 import pi0_loader

_DEFAULT_CKPT = "/home/zhangxa/codes/edgeLLM/Chamleon/models/openpi/pytorch/model.safetensors"
_DEFAULT_CKPT_DIR = "/home/zhangxa/codes/edgeLLM/Chamleon/models/openpi/pytorch"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _report(v_ref: np.ndarray, v_tvm: np.ndarray, threshold: float) -> bool:
    ref_finite = bool(np.isfinite(v_ref).all())
    tvm_finite = bool(np.isfinite(v_tvm).all())
    if not (ref_finite and tvm_finite):
        print(f"[compare] NON-FINITE detected: ref_finite={ref_finite} tvm_finite={tvm_finite}")
    cos = cosine(v_ref, v_tvm)
    max_abs = float(np.abs(v_ref - v_tvm).max())
    rel = max_abs / (float(np.abs(v_ref).max()) + 1e-12)
    print(f"[compare] shape={v_ref.shape} cosine={cos:.6f} max_abs_diff={max_abs:.3e} rel={rel:.3e}")
    ok = cos >= threshold
    print(f"[compare] {'PASS' if ok else 'FAIL'} (threshold={threshold})")
    return ok, cos


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #
def make_inputs(config: Pi0Config, seed: int = 0):
    from mlc_vla.openpi_ref import sinusoidal_time_emb

    rng = np.random.default_rng(seed)
    prefix = (0.02 * rng.standard_normal((1, config.prefix_len, config.vlm.width))).astype("float32")
    x_t = rng.standard_normal((1, config.action_horizon, config.action_dim)).astype("float32")
    time_emb = sinusoidal_time_emb(0.7, config.action_expert.width).astype("float32")
    return prefix, x_t, time_emb


# --------------------------------------------------------------------------- #
# 档 A：随机源权重
# --------------------------------------------------------------------------- #
def _random_source_sd(config: Pi0Config, named_params, seed: int = 1):
    """按 build_name_map 反推源 openpi-key 的随机权重（正确 shape），供双方共享。"""
    rng = np.random.default_rng(seed)
    name_map = pi0_loader.build_name_map(config)
    tgt_shape = {n: tuple(int(s) for s in p.shape) for n, p in named_params}
    tgt_dtype = {n: p.dtype for n, p in named_params}
    src: dict[str, np.ndarray] = {}

    def rand(shape, dtype):
        if str(dtype).startswith("int"):
            return np.zeros(shape, dtype=dtype)
        return (0.02 * rng.standard_normal(shape)).astype("float32")

    for tname, spec in name_map.items():
        shp = tgt_shape[tname]
        dt = tgt_dtype[tname]
        keys = spec.keys
        if len(keys) == 2:  # gate_up concat -> 两个 [mlp, width]
            half = (shp[0] // 2, shp[1])
            for kk in keys:
                if kk not in src:
                    src[kk] = rand(half, dt)
        elif spec.fn is pi0_loader._conv_to_linear:  # linear[hidden,p*p*c] -> conv[hidden,c,p,p]
            hidden = shp[0]
            p = config.siglip.patch_size
            c = config.siglip.num_channels
            src[keys[0]] = rand((hidden, c, p, p), dt)
        else:
            if keys[0] not in src:
                src[keys[0]] = rand(shp, dt)
    return src


def run_mode_a(config: Pi0Config, target: str, threshold: float, seed: int) -> bool:
    import torch

    model = Pi0Model(config)
    _, named_params, _ = _build_irmodule(model)
    src_np = _random_source_sd(config, named_params, seed=seed)
    params = pi0_loader.load_params(config, src_np, named_params=named_params, dtype=config.dtype)

    prefix, x_t, time_emb = make_inputs(config, seed=seed + 100)

    # 参考（自包含，CPU fp32，用同一份随机权重）
    from mlc_vla.openpi_ref import build_reference

    sd_torch = {k: torch.as_tensor(v) for k, v in src_np.items()}
    ref, impl = build_reference(config, sd_torch, device="cpu", dtype=torch.float32, prefer_real=False)
    print(f"[modeA] reference impl = {impl}")
    v_ref = ref.denoise(prefix, x_t, time_emb)

    v_tvm = _run_tvm(config, params, prefix, x_t, time_emb, target)
    return _report(v_ref, v_tvm, threshold)


# --------------------------------------------------------------------------- #
# 档 B：真实权重
# --------------------------------------------------------------------------- #
def run_mode_b(
    config: Pi0Config,
    target: str,
    threshold: float,
    seed: int,
    ckpt: str,
    dtype: str,
    ref_dtype: str,
    prefer_real: bool,
    use_kv: bool = False,
) -> bool:
    import torch

    from mlc_vla.openpi_ref import build_reference, load_state_dict_torch

    prefix, x_t, time_emb = make_inputs(config, seed=seed)

    # ---- 阶段 1：参考（先跑，随后释放显存，避免与 TVM 共驻）----
    print(f"[modeB] loading checkpoint (torch) from {ckpt}")
    sd = load_state_dict_torch(ckpt)
    device = "cuda" if target != "llvm" and target != "c" else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[ref_dtype]
    ref, impl = build_reference(config, sd, device=device, dtype=torch_dtype, prefer_real=prefer_real)
    print(f"[modeB] reference impl = {impl}, ref_dtype={ref_dtype}, device={device}")
    v_ref = ref.denoise(prefix, x_t, time_emb)
    print(f"[modeB] v_ref ready {v_ref.shape}; releasing torch GPU memory")
    del ref, sd
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 阶段 2：TVM ----
    model = Pi0Model(config)
    _, named_params, _ = _build_irmodule(model)
    src_np = pi0_loader.load_safetensors(ckpt, dtype="float32")
    params = pi0_loader.load_params(config, src_np, named_params=named_params, dtype=dtype)
    del src_np
    gc.collect()
    v_tvm = (_run_tvm_kv if use_kv else _run_tvm)(config, params, prefix, x_t, time_emb, target)
    print(f"[modeB] tvm path = {'M1 (prefill+denoise_step_kv)' if use_kv else 'M0 (denoise_step)'}")
    return _report(v_ref, v_tvm, threshold)


# --------------------------------------------------------------------------- #
# TVM 编译 + 运行
# --------------------------------------------------------------------------- #
# 对拍只需 denoise_step；且 bf16 下必须排除含 layer_norm 的 embed_image。
_DENOISE_ONLY = ["denoise_step"]


def _build_irmodule(model: Pi0Model):
    model.to(model.config.dtype)
    return model.export_tvm(spec=model.get_default_spec(functions=_DENOISE_ONLY), allow_extern=True)


def _run_tvm(config, params, prefix, x_t, time_emb, target):
    import tvm
    from tvm import relax

    from mlc_vla.compile import compile_model

    ex, named_params = compile_model(config, target, functions=_DENOISE_ONLY)
    if target in ("llvm", "c"):
        dev = tvm.cpu(0)
    else:
        dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    dt = config.dtype
    tvm_params = [tvm.runtime.tensor(params[name], dev) for name, _ in named_params]
    v = vm["denoise_step"](
        tvm.runtime.tensor(prefix.astype(dt), dev),
        tvm.runtime.tensor(x_t.astype(dt), dev),
        tvm.runtime.tensor(time_emb.astype(dt), dev),
        tvm_params,
    )
    out = v.numpy() if hasattr(v, "numpy") else v[0].numpy()
    return out.astype("float32")


_KV_FUNCS = ["prefill", "denoise_step_kv"]


def _run_tvm_kv(config, params, prefix, x_t, time_emb, target):
    """M1 路径：prefill 固化 prefix K/V，再 suffix-only denoise_step_kv。"""
    import tvm
    from tvm import relax

    from mlc_vla.compile import compile_model

    ex, named_params = compile_model(config, target, functions=_KV_FUNCS)
    dev = tvm.cpu(0) if target in ("llvm", "c") else tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    dt = config.dtype
    tvm_params = [tvm.runtime.tensor(params[name], dev) for name, _ in named_params]

    kv = vm["prefill"](tvm.runtime.tensor(prefix.astype(dt), dev), tvm_params)
    keys, values = (kv[0], kv[1]) if not hasattr(kv, "numpy") else (kv, kv)
    v = vm["denoise_step_kv"](
        keys, values,
        tvm.runtime.tensor(x_t.astype(dt), dev),
        tvm.runtime.tensor(time_emb.astype(dt), dev),
        tvm_params,
    )
    out = v.numpy() if hasattr(v, "numpy") else v[0].numpy()
    return out.astype("float32")


# --------------------------------------------------------------------------- #
def _make_config(args) -> Pi0Config:
    if args.dummy:
        cfg = Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            max_token_len=8,
            action_horizon=4,
            dtype=args.dtype,
        )
        cfg.siglip.num_hidden_layers = 2
        return cfg
    return Pi0Config.from_openpi_config(_DEFAULT_CKPT_DIR, dtype=args.dtype)


def main():
    ap = argparse.ArgumentParser(description="MLC-VLA vs openpi 单步对拍")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--threshold", type=float, default=0.99, help="cosine 阈值")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", help="mlc-vla 编译/参数 dtype")
    ap.add_argument("--ref-dtype", default="bfloat16", help="参考实现 dtype（档 B）")
    ap.add_argument("--ckpt", default=_DEFAULT_CKPT)
    ap.add_argument("--dummy", action="store_true", help="dummy 小尺寸（仅档 A）")
    ap.add_argument("--no-real", action="store_true", help="档 B 强制用自包含参考")
    ap.add_argument("--kv", action="store_true", help="档 B 用 M1 KV 路径(prefill+denoise_step_kv)")
    args = ap.parse_args()

    config = _make_config(args)
    print(f"[compare] mode={args.mode} target={args.target} dtype={args.dtype} "
          f"prefix_len={config.prefix_len} horizon={config.action_horizon}")

    if args.mode == "A":
        ok, _ = run_mode_a(config, args.target, args.threshold, args.seed)
    else:
        if args.dummy:
            raise SystemExit("档 B 需真实 checkpoint，不能与 --dummy 合用")
        ok, _ = run_mode_b(
            config, args.target, args.threshold, args.seed, args.ckpt,
            args.dtype, args.ref_dtype, prefer_real=not args.no_real, use_kv=args.kv,
        )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
