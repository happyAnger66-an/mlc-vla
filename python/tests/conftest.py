"""共享 fixtures：面向对外接口的 e2e 测试。

设计约束：
- 全部走 TVM ``target="c"``（CPU + gcc codegen），**不需要 GPU**。文档 M0.md 指出本仓 TVM
  的 ``target.build.llvm`` 未注册，CPU 验证一律用 ``c`` 目标。
- 未装 TVM / 无 C 编译器的环境自动 skip（``pytest.importorskip`` + 编译异常兜底），
  以便在纯 CI 上也能安全收集。
- 用 ``dummy`` 双专家 + 小步数缩小编译/执行规模，但**保留真实的图结构与对外调用契约**
  （prefill / denoise_step / denoise_step_kv / denoise_loop_kv 与 openpi 语义一致）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# dummy 专家 + 小 token/步数：编译快，图结构不变。num_denoise_steps 同时决定 denoise_loop_kv
# 展开步数与 sample_graph 的步数，需与 sample(num_steps=...) 对齐做「图内环==宿主环」对拍。
DUMMY_KWARGS = dict(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    max_token_len=8,
    action_horizon=4,
    num_denoise_steps=3,
    dtype="float32",
)

# 覆盖模型所有对外调用入口，一次编译复用（session 级）。
_ALL_FUNCS = ["denoise_step", "prefill", "denoise_step_kv", "denoise_loop_kv"]


@pytest.fixture(scope="session")
def tvm():
    """未安装 TVM 时整体 skip（而非收集报错）。"""
    return pytest.importorskip("tvm")


@pytest.fixture
def dummy_config(tvm):
    from mlc_vla.model.pi0 import Pi0Config

    return Pi0Config(**DUMMY_KWARGS)


@pytest.fixture
def make_params(tvm):
    """按 named_params 规格生成确定性随机权重并搬到 device 的工厂。"""

    def _make(named_params, dev, seed: int = 0):
        rng = np.random.default_rng(seed)
        out = []
        for _name, p in named_params:
            shape = [int(s) for s in p.shape]
            if p.dtype.startswith(("int", "uint")):
                arr = np.zeros(shape, dtype=p.dtype)
            else:
                arr = (0.02 * rng.standard_normal(shape)).astype(p.dtype)
            out.append(tvm.runtime.tensor(arr, dev))
        return out

    return _make


@pytest.fixture(scope="session")
def kv_runtime(tvm):
    """一次性把 dummy 模型编到 ``c`` 目标并建 VM，供多个 e2e 复用。

    返回 SimpleNamespace(config, ex, vm, named_params, dev)。编译失败（无 gcc / TVM
    codegen 不可用）则 skip 整组测试。
    """
    from tvm import relax

    from mlc_vla.compile import _device_for, compile_model
    from mlc_vla.model.pi0 import Pi0Config

    config = Pi0Config(**DUMMY_KWARGS)
    try:
        ex, named_params = compile_model(config, "c", functions=_ALL_FUNCS)
        dev = _device_for("c")
        vm = relax.VirtualMachine(ex, dev)
    except Exception as exc:  # noqa: BLE001 - 环境不具备 c 目标编译能力则跳过
        pytest.skip(f"TVM 'c' 目标编译不可用（缺 gcc 或 codegen）：{exc}")
    return SimpleNamespace(config=config, ex=ex, vm=vm, named_params=named_params, dev=dev)


@pytest.fixture
def kv_inputs(tvm, kv_runtime, make_params):
    """构造一组与 config 对齐的去噪输入张量（含 prefill 固化的 keys/values）。

    返回 SimpleNamespace，字段覆盖 4 个对外函数的全部入参。
    """
    from mlc_vla.sample import make_prefix_mask_np, make_rope_np, make_time_embs

    cfg = kv_runtime.config
    dev = kv_runtime.dev
    vm = kv_runtime.vm
    dt = cfg.dtype
    rng = np.random.default_rng(1)

    params = make_params(kv_runtime.named_params, dev)

    def _t(arr):
        return tvm.runtime.tensor(np.asarray(arr, dtype=dt), dev)

    prefix = _t(rng.standard_normal((1, cfg.prefix_len, cfg.vlm.width)))
    x_t = _t(rng.standard_normal((1, cfg.action_horizon, cfg.action_dim)))
    time_emb = _t(rng.standard_normal((1, cfg.action_expert.width)))

    pmask = tvm.runtime.tensor(make_prefix_mask_np(cfg.prefix_len), dev)
    cos_np, sin_np = make_rope_np(
        cfg.suffix_len, cfg.vlm.head_dim, cfg.rope_theta, offset=cfg.prefix_len
    )
    scos, ssin = tvm.runtime.tensor(cos_np, dev), tvm.runtime.tensor(sin_np, dev)

    n = cfg.num_denoise_steps
    x0 = tvm.runtime.tensor(
        rng.standard_normal((1, cfg.action_horizon, cfg.action_dim)).astype("float32"), dev
    )
    time_embs = np.concatenate(
        make_time_embs(n, cfg.action_expert.width), axis=0
    ).astype(dt)
    te_dev = tvm.runtime.tensor(time_embs, dev)

    # prefill 固化 K/V
    kv = vm["prefill"](prefix, pmask, params)
    keys, values = (kv[0], kv[1]) if not hasattr(kv, "numpy") else (kv, kv)

    return SimpleNamespace(
        params=params, prefix=prefix, x_t=x_t, time_emb=time_emb,
        pmask=pmask, scos=scos, ssin=ssin, x0=x0, te_dev=te_dev,
        keys=keys, values=values,
    )


def as_numpy(ret):
    """relax VM 返回值（单张量或 tuple）统一取第 0 个的 numpy。"""
    if hasattr(ret, "numpy"):
        return ret.numpy()
    return ret[0].numpy()
