"""编译 / 冒烟测试入口（M0）。

用法：
    # 冒烟测试：随机权重，验证 relax 计算图能编译并跑通（CPU）
    python -m mlc_vla.compile --smoke --target llvm

    # 仅导出 IRModule（打印 relax 文本）
    python -m mlc_vla.compile --dump-ir

M0 的目标是「打通」：图能 legalize / build / 运行，shape/dtype 自洽。
数值对齐见 ``mlc_vla.compare``。
"""

from __future__ import annotations

import argparse

import numpy as np

from mlc_vla.model.pi0 import Pi0Config, Pi0Model
from mlc_vla.model.pi0.pi0_model import include_for


def build_irmodule(config: Pi0Config, functions=None):
    """实例化模型并导出 relax IRModule + 参数规格。

    ``functions``：可选子函数列表（如 ``["denoise_step"]``）。分段编译时只构建该 stage 需要
    的子模块（vision/embed/backbone），使每个 engine 只携带自身权重。SigLIP 的 bf16 layer_norm
    已由 ``LayerNormF32`` 内部 fp32 计算解决。
    """
    model = Pi0Model(config, include=include_for(functions))
    # 把参数 dtype 统一到 config.dtype（fp16/bf16 省显存；默认 fp32 时为 no-op）
    model.to(config.dtype)
    mod, named_params, ext_mods = model.export_tvm(
        spec=model.get_default_spec(functions=functions),
        allow_extern=True,
    )
    return model, mod, named_params, ext_mods


def apply_gemm_prepasses(mod, tgt):
    """CUDA 目标下把 matmul 卸载到 cuBLAS，并把 transpose+matmul 融合掉「转置税」。

    对齐 MLC LLM 的 ``BLASDispatch`` + ``FuseTransposeMatmul``（须在 LegalizeOps 之前）：
      - ``partition_for_cublas`` + ``RunCodegen``：把 ``matmul`` / ``matmul_transposed`` 等
        pattern 分区并生成 cuBLAS 外部调用（``matmul_transposed`` 直接吃掉显式 transpose）。
      - ``FuseTransposeMatmul``：处理 cuBLAS 未覆盖的残留 ``transpose(w) @ x``。
    nsys 基线里 ``transpose*`` 约占 GPU 时间 1/3，本 pass 是 Phase B 主收益来源。
    """
    import tvm
    from tvm import relax
    from tvm.relax.backend.cuda.cublas import partition_for_cublas

    with tgt:
        mod = partition_for_cublas(mod)
        mod = relax.transform.RunCodegen()(mod)
        mod = relax.transform.FuseTransposeMatmul()(mod)
    return mod


def compile_model(config: Pi0Config, target: str = "c", functions=None,
                  cuda_graph: bool = False, cublas: bool = False):
    """编译模型，返回 (VM 可加载对象, named_params)。

    - ``c`` 目标（默认 CPU 路径）：zero pipeline + 本机 gcc 编译为 .so 后加载。
    - GPU 目标（cuda/metal/...）：使用 target 感知的默认 pipeline。
    - ``cuda_graph``：CUDA 目标下开启 ``RewriteCUDAGraph``，把去噪步的静态 kernel 序列
      捕获成 CUDA Graph（一次 launch 重放），降低 host 端多 kernel launch 开销。
    - ``cublas``：CUDA 目标下把 matmul 卸载到 cuBLAS 并融合 transpose（Phase B）。
    """
    import os
    import tempfile

    import tvm
    from tvm import relax

    _, mod, named_params, _ = build_irmodule(config, functions=functions)
    tgt = tvm.target.Target(target)
    # CUDA 13.x 的 NVRTC JIT 无法编译 cuda_fp8.hpp（double4_16a 未定义），改用 nvcc 子进程。
    if tgt.kind.name == "cuda":
        os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    if tgt.kind.name == "c":
        mod = relax.get_pipeline("zero")(mod)
        ex = relax.build(mod, target=tgt)
        so_path = os.path.join(tempfile.mkdtemp(prefix="mlc_vla_"), "pi0.so")
        ex.export_library(so_path)
        return tvm.runtime.load_module(so_path), named_params
    if cublas and tgt.kind.name == "cuda":
        mod = apply_gemm_prepasses(mod, tgt)
    relax_pipeline = relax.get_default_pipeline(tgt)
    pass_cfg = {"relax.backend.use_cuda_graph": True} if (cuda_graph and tgt.kind.name == "cuda") else {}
    with tgt, tvm.transform.PassContext(config=pass_cfg):
        ex = relax.build(mod, target=tgt, relax_pipeline=relax_pipeline)
    return ex, named_params


def _device_for(target: str):
    import tvm

    tgt = tvm.target.Target(target)
    if tgt.kind.name == "c":
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


def smoke_test(config: Pi0Config, target: str = "c"):
    """随机权重跑一次 denoise_step，验证图自洽。"""
    import tvm
    from tvm import relax

    ex, named_params = compile_model(config, target)
    dev = _device_for(target)
    vm = relax.VirtualMachine(ex, dev)
    params = _random_params(named_params, dev)

    x_t = tvm.runtime.tensor(
        np.random.randn(1, config.action_horizon, config.action_dim).astype(config.dtype), dev
    )
    prefix = tvm.runtime.tensor(
        np.random.randn(1, config.prefix_len, config.vlm.width).astype(config.dtype), dev
    )
    time_emb = tvm.runtime.tensor(
        np.random.randn(1, config.action_expert.width).astype(config.dtype), dev
    )
    v_t = vm["denoise_step"](prefix, x_t, time_emb, params)
    out = v_t.numpy() if hasattr(v_t, "numpy") else v_t[0].numpy()
    print(f"[smoke] denoise_step OK, v_t shape = {out.shape}, finite = {np.isfinite(out).all()}")
    return out


def main():
    ap = argparse.ArgumentParser(description="MLC-VLA π0.5 compile / smoke (M0)")
    ap.add_argument("--target", default="c")
    ap.add_argument("--smoke", action="store_true", help="随机权重跑通 denoise_step")
    ap.add_argument("--dump-ir", action="store_true", help="打印导出的 relax IRModule")
    ap.add_argument("--dummy", action="store_true", help="用 dummy 小尺寸专家，加速验证")
    args = ap.parse_args()

    if args.dummy:
        config = Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            max_token_len=8,
            action_horizon=4,
        )
        # dummy 时缩小视觉以加速
        config.siglip.num_hidden_layers = 2
    else:
        config = Pi0Config()

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
