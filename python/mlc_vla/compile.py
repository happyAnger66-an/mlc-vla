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


def build_irmodule(config: Pi0Config):
    """实例化模型并导出 relax IRModule + 参数规格。"""
    model = Pi0Model(config)
    mod, named_params, ext_mods = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    return model, mod, named_params, ext_mods


def _make_pipeline():
    import tvm
    from tvm import relax

    @tvm.transform.module_pass(opt_level=0, name="mlc_vla_m0")
    def _pipeline(mod, _ctx):
        seq = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        return seq(mod)

    return _pipeline


def compile_model(config: Pi0Config, target: str = "llvm"):
    import tvm
    from tvm import dlight as dl
    from tvm import relax

    _, mod, named_params, _ = build_irmodule(config)
    tgt = tvm.target.Target(target)
    with tgt:
        mod = _make_pipeline()(mod)
        if tgt.kind.name != "llvm":
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
    ex = relax.build(mod, target=tgt)
    return ex, named_params


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


def smoke_test(config: Pi0Config, target: str = "llvm"):
    """随机权重跑一次 denoise_step，验证图自洽。"""
    import tvm
    from tvm import relax

    ex, named_params = compile_model(config, target)
    dev = tvm.device(target, 0)
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
    ap.add_argument("--target", default="llvm")
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
