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

from mlc_vla.model.plannn3 import (
    Dinov3Config,
    Dinov3VisualEncoder,
    Plannn3Config,
    Plannn3Model,
)
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


def build_visual_irmodule(vcfg: Dinov3Config, functions=None):
    """实例化 DINOv3 视觉塔并导出 relax IRModule + 参数规格。"""
    model = Dinov3VisualEncoder(vcfg)
    model.to(vcfg.dtype)
    mod, named_params, ext_mods = model.export_tvm(
        spec=model.get_default_spec(functions=functions),
        allow_extern=True,
    )
    return model, mod, named_params, ext_mods


def cublas_available() -> bool:
    """本机 TVM 是否编入 cuBLAS BYOC 扩展（``relax.ext.cublas``）。

    无扩展时若强开 cuBLAS，``RunCodegen`` 生成的 extern 调用会在 build 期失败；据此自动回退。
    对齐 ``mlc_vla.compile.cublas_available``。
    """
    try:
        import tvm

        return bool(tvm.get_global_func("relax.ext.cublas", True))
    except Exception:  # noqa: BLE001 - 探测失败一律视为不可用并回退 dlight
        return False


def resolve_cublas(cublas, target_kind: str) -> bool:
    """把三态 ``cublas``（None=自动 / True / False）解析成实际是否启用（仅 CUDA 目标）。"""
    if target_kind != "cuda":
        return False
    avail = cublas_available()
    if cublas is None:
        return avail
    if cublas and not avail:
        import warnings

        warnings.warn(
            "cublas=True 但本机 TVM 未启用 relax.ext.cublas 扩展，回退 dlight GEMM。",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return bool(cublas)


def apply_gemm_prepasses(mod, tgt):
    """CUDA 目标下把 matmul 卸载 cuBLAS 并融合 transpose+matmul（对齐 mlc_vla.compile）。"""
    from tvm import relax
    from tvm.relax.backend.cuda.cublas import partition_for_cublas

    with tgt:
        mod = partition_for_cublas(mod)
        mod = relax.transform.RunCodegen()(mod)
        mod = relax.transform.FuseTransposeMatmul()(mod)
    return mod


def _build_and_compile(mod, named_params, target: str, cuda_graph: bool = False, cublas=None):
    """公共编译尾：c 目标走 zero pipeline+gcc；CUDA 可选 cuBLAS BYOC + CUDA Graph；其它走默认 pipeline。

    - ``cuda_graph``：CUDA 目标下开启 ``relax.backend.use_cuda_graph``，把静态 kernel 序列
      捕获成 CUDA Graph（decode 单步或 ``decode_loop_kv`` 整环一次 launch 重放）。
    - ``cublas``：三态。``None`` 在 CUDA 且扩展可用时自动开启（把 matmul 卸载 cuBLAS + 吃掉转置税）。
      不可用时默认 pipeline 内的 dlight 会给出 GEMM/GEMV/Reduction 调度作为回退。
    """
    import os
    import tempfile

    import tvm
    from tvm import relax

    tgt = tvm.target.Target(target)
    if tgt.kind.name == "cuda":
        os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    if tgt.kind.name == "c":
        mod = relax.get_pipeline("zero")(mod)
        ex = relax.build(mod, target=tgt)
        so_path = os.path.join(tempfile.mkdtemp(prefix="mlc_vla_p3_"), "plannn3.so")
        ex.export_library(so_path)
        return tvm.runtime.load_module(so_path), named_params
    if resolve_cublas(cublas, tgt.kind.name):
        mod = apply_gemm_prepasses(mod, tgt)
    pass_cfg = (
        {"relax.backend.use_cuda_graph": True} if (cuda_graph and tgt.kind.name == "cuda") else {}
    )
    with tgt, tvm.transform.PassContext(config=pass_cfg):
        ex = relax.build(mod, target=tgt, relax_pipeline=relax.get_default_pipeline(tgt))
    return ex, named_params


def compile_visual(vcfg: Dinov3Config, target: str = "c", functions=None,
                   cuda_graph: bool = False, cublas=None):
    """编译 DINOv3 视觉塔，返回 (VM 可加载对象, named_params)。"""
    _, mod, named_params, _ = build_visual_irmodule(vcfg, functions=functions)
    return _build_and_compile(mod, named_params, target, cuda_graph=cuda_graph, cublas=cublas)


def compile_model(config: Plannn3Config, target: str = "c", functions=None,
                  cuda_graph: bool = False, cublas=None):
    """编译模型，返回 (VM 可加载对象, named_params)。

    - ``c`` 目标（默认 CPU 路径）：zero pipeline + 本机 gcc 编译为 .so 后加载。
    - CUDA 目标：默认 pipeline（含 dlight），可选 cuBLAS BYOC（``cublas``）与 CUDA Graph（``cuda_graph``）。
    - 其它目标（llvm/...）：使用 target 感知的默认 pipeline。
    """
    _, mod, named_params, _ = build_irmodule(config, functions=functions)
    return _build_and_compile(mod, named_params, target, cuda_graph=cuda_graph, cublas=cublas)


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


def smoke_loop(config: Plannn3Config, target: str = "c", cuda_graph: bool = False, cublas=None):
    """随机权重跑通图内整段解码环 ``decode_loop_kv``，验证图自洽（含图内 argmax）。"""
    import tvm
    from tvm import relax

    ex, named_params = compile_model(
        config, target, functions=["decode_loop_kv"], cuda_graph=cuda_graph, cublas=cublas
    )
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    params = _random_params(named_params, dev)

    token_embeds = tvm.runtime.tensor(
        np.random.randn(1, config.prompt_len, config.n_embd).astype(config.dtype), dev
    )
    ret = vm["decode_loop_kv"](token_embeds, params)
    ids = ret.numpy() if hasattr(ret, "numpy") else ret[0].numpy()
    print(
        f"[smoke] decode_loop_kv OK, traj_ids={ids.shape} (expect [1,{config.pred_times}]): "
        f"{ids.reshape(-1).tolist()}"
    )
    return ids


def smoke_visual(vcfg: Dinov3Config, target: str = "c"):
    """随机权重跑一次 embed_visual，验证视觉塔图自洽。"""
    import tvm
    from tvm import relax

    ex, named_params = compile_visual(vcfg, target)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    params = _random_params(named_params, dev)

    image = tvm.runtime.tensor(
        np.random.randn(1, 3, vcfg.image_h, vcfg.image_w).astype(vcfg.dtype), dev
    )
    ret = vm["embed_visual"](image, params)
    out = ret.numpy() if hasattr(ret, "numpy") else ret[0].numpy()
    hp, wp = vcfg.grid()
    print(
        f"[smoke] embed_visual OK, tokens={out.shape} (expect [1,{hp * wp},{vcfg.out_channel}]), "
        f"finite={np.isfinite(out).all()}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="MLC-VLA plannn3 compile / smoke (M0/M1)")
    ap.add_argument("--target", default="c")
    ap.add_argument("--smoke", action="store_true", help="随机权重跑通 prefill + decode_step")
    ap.add_argument("--visual", action="store_true", help="随机权重跑通视觉塔 embed_visual")
    ap.add_argument("--generate", action="store_true", help="随机权重跑通宿主 18 步 AR 环")
    ap.add_argument("--loop", action="store_true", help="随机权重跑通图内整段解码环 decode_loop_kv")
    ap.add_argument("--dump-ir", action="store_true", help="打印导出的 relax IRModule")
    ap.add_argument("--dummy", action="store_true", help="用 dummy 小尺寸，加速验证")
    ap.add_argument("--cuda-graph", action="store_true", help="CUDA 目标下开启 CUDA Graph 捕获")
    ap.add_argument("--cublas", dest="cublas", action="store_true", default=None,
                    help="强制启用 cuBLAS BYOC（默认自动探测；不可用回退 dlight）")
    ap.add_argument("--no-cublas", dest="cublas", action="store_false",
                    help="强制禁用 cuBLAS BYOC，走 dlight")
    args = ap.parse_args()

    config = Plannn3Config.dummy() if args.dummy else Plannn3Config()

    if args.visual:
        vcfg = Dinov3Config.dummy() if args.dummy else Dinov3Config()
        if args.dump_ir:
            _, mod, _, _ = build_visual_irmodule(vcfg)
            print(mod)
        else:
            smoke_visual(vcfg, args.target)
        return
    if args.generate:
        from mlc_vla.plannn3_runner import smoke_generate

        smoke_generate(config, args.target)
        return
    if args.loop:
        smoke_loop(config, args.target, cuda_graph=args.cuda_graph, cublas=args.cublas)
        return
    if args.dump_ir:
        _, mod, _, _ = build_irmodule(config)
        print(mod)
        return
    if args.smoke:
        smoke_test(config, args.target)
        return
    compile_model(config, args.target, cuda_graph=args.cuda_graph, cublas=args.cublas)
    print("[compile] build OK")


if __name__ == "__main__":
    main()
