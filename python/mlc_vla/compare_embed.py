"""B: embed_image / embed_language 的 bf16 vs fp32 数值对拍（验证 SigLIP LayerNorm 的
fp32-internal workaround 保号）。

TVM ``layer_norm`` 不支持 bf16，SigLIP 用 ``LayerNormF32`` 内部提升到 fp32 计算。这里以
**fp32 编译**为基线，验证 **bf16 编译**（含 55 个 LayerNorm）输出与之一致（仅 bf16 舍入误差）。

用法：
    python -m mlc_vla.compare_embed --target cuda --dummy      # 结构验证
    python -m mlc_vla.compare_embed --target cuda \
        --ckpt /path/to/model.safetensors                      # 真实权重
"""

from __future__ import annotations

import argparse
import dataclasses
import gc

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.model.pi0 import Pi0Config, Pi0Model
from mlc_vla.model.pi0 import pi0_loader

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _named_params(config: Pi0Config, func: str):
    from mlc_vla.model.pi0.pi0_model import include_for

    m = Pi0Model(config, include=include_for([func]))
    m.to(config.dtype)
    _, npar, _ = m.export_tvm(spec=m.get_default_spec(functions=[func]), allow_extern=True)
    return npar


def _run_one(config: Pi0Config, target: str, func: str, params_src, inp):
    """编译并运行单个 embed 函数，返回 fp32 numpy 输出。引擎/参数在返回后释放。"""
    import tvm
    from tvm import relax

    dev = _device_for(target)
    ex, npar = compile_model(config, target, functions=[func])
    vm = relax.VirtualMachine(ex, dev)
    tp = [tvm.runtime.tensor(params_src[name].astype(p.dtype), dev) for name, p in npar]
    out = vm[func](tvm.runtime.tensor(inp, dev), tp)
    res = out.numpy().astype("float32")
    del vm, ex, tp
    gc.collect()
    return res


def _src_for(config: Pi0Config, func: str, ckpt: str | None, rng):
    npar = _named_params(config, func)
    if ckpt:
        raw = pi0_loader.load_safetensors(ckpt, dtype="float32")
        src = pi0_loader.load_params(config, raw, named_params=npar, dtype="float32")
        del raw
        gc.collect()
        return src
    return {name: (0.02 * rng.standard_normal([int(x) for x in p.shape])).astype("float32")
            if not p.dtype.startswith(("int", "uint"))
            else np.zeros([int(x) for x in p.shape], p.dtype)
            for name, p in npar}


def _parity_one(config_base, target, func, inp_dtype_fn, ckpt, rng):
    """对单个函数做 fp32 vs bf16 对拍（fp32/bf16 顺序编译，之间释放显存）。"""
    cfg_fp32 = dataclasses.replace(config_base, dtype="float32")
    src = _src_for(cfg_fp32, func, ckpt, rng)
    inp_fp32 = inp_dtype_fn("float32")
    out_fp32 = _run_one(cfg_fp32, target, func, src, inp_fp32)

    cfg_bf16 = dataclasses.replace(config_base, dtype="bfloat16")
    inp_bf16 = inp_dtype_fn("bfloat16") if inp_fp32.dtype != np.int32 else inp_fp32
    out_bf16 = _run_one(cfg_bf16, target, func, src, inp_bf16)
    del src
    gc.collect()
    return out_fp32, out_bf16


def run(config_base: Pi0Config, target: str, seed: int, ckpt: str | None):
    rng = np.random.default_rng(seed)
    s = config_base.siglip
    image = rng.standard_normal((1, s.image_size, s.image_size, s.num_channels)).astype("float32")
    ids = rng.integers(0, 1000, size=(1, config_base.max_token_len)).astype("int32")

    # embed_image：SigLIP 55x LayerNorm，B 的核心验证
    img_fp32, img_bf16 = _parity_one(
        config_base, target, "embed_image",
        lambda dt: image.astype(dt), ckpt, np.random.default_rng(seed + 1))
    ci = _cosine(img_fp32, img_bf16)
    print(f"[compare_embed] embed_image    fp32 vs bf16: shape={img_fp32.shape} cosine={ci:.6f} "
          f"(SigLIP 55x LayerNorm)")

    # embed_language：无 LayerNorm，仅 Embedding gather × sqrt(width)
    lang_fp32, lang_bf16 = _parity_one(
        config_base, target, "embed_language",
        lambda dt: ids, ckpt, np.random.default_rng(seed + 2))
    cl = _cosine(lang_fp32, lang_bf16)
    print(f"[compare_embed] embed_language fp32 vs bf16: shape={lang_fp32.shape} cosine={cl:.6f}")

    ok = ci >= 0.99 and cl >= 0.99
    print(f"[compare_embed] {'PASS' if ok else 'FAIL'} (threshold=0.99) — SigLIP LayerNorm bf16 编译验证")
    return ok


def main():
    ap = argparse.ArgumentParser(description="embed_image/embed_language bf16 vs fp32 对拍")
    ap.add_argument("--target", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    if args.dummy:
        config = Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy",
                           max_token_len=8, action_horizon=4)
    else:
        config = Pi0Config()
    ok = run(config, args.target, args.seed, args.ckpt)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
