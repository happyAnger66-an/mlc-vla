"""plannn3 编译 / 冒烟测试入口（M0）。

用法：
    # 冒烟：随机权重跑通 prefill + decode_step（dummy 小尺寸，秒级）
    python -m mlc_vla.plannn3_compile --dummy --smoke --target c

    # 仅导出 IRModule（打印 relax 文本）
    python -m mlc_vla.plannn3_compile --dummy --dump-ir

M0 目标：图能 legalize / build / 运行，shape/dtype 自洽。数值对齐留待后续 milestone。
"""

from __future__ import annotations

import argparse

import numpy as np

from mlc_vla.model.plannn3 import Plannn3Config, Plannn3Model
from mlc_vla.model.plannn3.plannn3_model import _rope_tables_np


def build_irmodule(config: Plannn3Config, functions=None):
    """实例化模型并导出 relax IRModule + 参数规格。"""
    model = Plannn3Model(config)
    model.to(config.dtype)
    mod, named_params, ext_mods = model.export_tvm(
        spec=model.get_default_spec(functions=functions),
        allow_extern=True,
    )
    return model, mod, named_params, ext_mods


def compile_model(config: Plannn3Config, target: str = "c", functions=None):
    """编译模型，返回 (VM 可加载对象, named_params)。

    - ``c`` 目标（默认 CPU 路径）：zero pipeline + 本机 gcc 编译为 .so 后加载。
    - 其它目标（llvm/cuda/...）：使用 target 感知的默认 pipeline。
    """
    import os
    import tempfile

    import tvm
    from tvm import relax

    _, mod, named_params, _ = build_irmodule(config, functions=functions)
    tgt = tvm.target.Target(target)
    if tgt.kind.name == "cuda":
        os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    if tgt.kind.name == "c":
        mod = relax.get_pipeline("zero")(mod)
        ex = relax.build(mod, target=tgt)
        so_path = os.path.join(tempfile.mkdtemp(prefix="mlc_vla_p3_"), "plannn3.so")
        ex.export_library(so_path)
        return tvm.runtime.load_module(so_path), named_params
    relax_pipeline = relax.get_default_pipeline(tgt)
    with tgt:
        ex = relax.build(mod, target=tgt, relax_pipeline=relax_pipeline)
    return ex, named_params


def _device_for(target: str):
    import tvm

    tgt = tvm.target.Target(target)
    if tgt.kind.name in ("c", "llvm"):
        return tvm.cpu(0)
    return tvm.device(tgt.kind.name, 0)


def _random_params(named_params, dev):
    import tvm

    params = []
    for _name, param in named_params:
        shape = [int(s) for s in param.shape]
        if param.dtype.startswith("int"):
            arr = np.zeros(shape, dtype=param.dtype)
        else:
            arr = (0.02 * np.random.randn(*shape)).astype(param.dtype)
        params.append(tvm.runtime.tensor(arr, dev))
    return params


def _unpack(ret):
    """VM 返回单张量或 tuple，统一成 numpy list。"""
    if hasattr(ret, "numpy"):
        return [ret.numpy()]
    return [ret[i].numpy() for i in range(len(ret))]


def smoke_test(config: Plannn3Config, target: str = "c"):
    """随机权重跑一次 prefill + decode_step，验证图自洽。"""
    import tvm
    from tvm import relax

    ex, named_params = compile_model(config, target)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    params = _random_params(named_params, dev)

    dt = config.dtype
    half = config.head_dim // 2
    max_seq = config.max_seq_len
    pos = config.prompt_len  # 首个解码步写入位置

    # ---- prefill ----
    token_embeds = tvm.runtime.tensor(
        np.random.randn(1, config.prompt_len, config.n_embd).astype(dt), dev
    )
    ret = vm["prefill"](token_embeds, params)
    logits0, kv = ret[0], ret[1]
    l0 = logits0.numpy()
    print(f"[smoke] prefill OK, logits={l0.shape}, kv={[int(s) for s in kv.shape]}, finite={np.isfinite(l0).all()}")

    # ---- decode_step ----
    latest = tvm.runtime.tensor(np.random.randn(1, 1, config.n_embd).astype(dt), dev)
    cos, sin = _rope_tables_np(1, config.head_dim, config.rope_theta, offset=pos)
    step_cos = tvm.runtime.tensor(cos, dev)  # [1,1,1,half]
    step_sin = tvm.runtime.tensor(sin, dev)
    idx = np.arange(max_seq)
    add = np.where(idx <= pos, 0.0, config.attn_neg_inf).astype("float32").reshape(1, 1, 1, max_seq)
    onehot = (idx == pos).astype(dt).reshape(1, max_seq, 1)
    add_mask = tvm.runtime.tensor(add, dev)
    write_onehot = tvm.runtime.tensor(onehot, dev)

    ret = vm["decode_step"](latest, step_cos, step_sin, add_mask, write_onehot, kv, params)
    logits1, _kv2 = ret[0], ret[1]
    l1 = logits1.numpy()
    print(
        f"[smoke] decode_step OK, logits={l1.shape}, half={half}, finite={np.isfinite(l1).all()}"
    )
    return l0, l1


def main():
    ap = argparse.ArgumentParser(description="MLC-VLA plannn3 compile / smoke (M0)")
    ap.add_argument("--target", default="c")
    ap.add_argument("--smoke", action="store_true", help="随机权重跑通 prefill + decode_step")
    ap.add_argument("--dump-ir", action="store_true", help="打印导出的 relax IRModule")
    ap.add_argument("--dummy", action="store_true", help="用 dummy 小尺寸，加速验证")
    args = ap.parse_args()

    config = Plannn3Config.dummy() if args.dummy else Plannn3Config()

    if args.dump_ir:
        _, mod, _, _ = build_irmodule(config)
        print(mod)
        return
    if args.smoke:
        smoke_test(config, args.target)
        return
    compile_model(config, args.target)
    print("[compile] build OK")


if __name__ == "__main__":
    main()
