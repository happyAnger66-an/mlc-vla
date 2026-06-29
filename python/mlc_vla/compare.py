"""与 openpi 单步 v_t 数值对拍（M0 gate）。

思路：同一组随机输入（prefix_embs / x_t / time_emb）分别喂给
- openpi PyTorch 参考实现的 ``denoise_step`` 等价路径
- MLC-VLA 编译产物的 ``denoise_step``
比较 cosine 相似度。

依赖（需在装有 openpi 的环境运行）：
- openpi（``model_optimizer/third_party/openpi``）及其 transformers_replace
- π0.5 checkpoint（用于真实权重对拍；M0 也可先用同步随机权重做结构对拍）

M0 两档对拍：
- 档 A（结构对拍）：双方用同一份随机权重（通过 loader 反向同步），验证计算图逻辑一致。
- 档 B（权重对拍）：加载真实 π0.5 权重，端到端 cosine gate。

本文件提供档 A 的脚手架；档 B 需补全 loader（见 pi0_loader.py）。
"""

from __future__ import annotations

import argparse

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_mlc(config, params, prefix, x_t, time_emb, target="llvm"):
    import tvm
    from tvm import relax

    from mlc_vla.compile import compile_model

    ex, named_params = compile_model(config, target)
    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    tvm_params = [tvm.runtime.tensor(params[name]) for name, _ in named_params]
    v = vm["denoise_step"](
        tvm.runtime.tensor(prefix, dev),
        tvm.runtime.tensor(x_t, dev),
        tvm.runtime.tensor(time_emb, dev),
        tvm_params,
    )
    return (v.numpy() if hasattr(v, "numpy") else v[0].numpy())


def main():
    ap = argparse.ArgumentParser(description="MLC-VLA vs openpi 单步对拍 (M0)")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--rtol", type=float, default=0.99, help="cosine 阈值")
    args = ap.parse_args()

    print(
        "M0 对拍脚手架：\n"
        "  档 A 需要把同一份随机权重同步给 openpi 与 MLC-VLA（待 loader 反向映射）。\n"
        "  档 B 需要真实 π0.5 权重 + 完整 loader（见 pi0_loader.py）。\n"
        "当前为占位实现：请在装有 openpi 的环境补全权重同步后启用。"
    )
    raise SystemExit(
        "TODO(M0): 补全 openpi 权重同步以启用对拍。先用 `python -m mlc_vla.compile --smoke` 验证图自洽。"
    )


if __name__ == "__main__":
    main()
