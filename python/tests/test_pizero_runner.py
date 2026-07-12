"""PiZeroRunner 对外契约（Chamleon worker 实际驱动的推理编排类）。

覆盖：构造 → to_params → sample（宿主环）/ sample_graph（图内环），形状/有限性，
两条环的数值一致性，以及 prefix_pad（右侧 padding）路径。全部 CPU ``c`` 目标。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DUMMY_KWARGS


@pytest.fixture(scope="module")
def runner(tvm):
    from mlc_vla.model.pi0 import Pi0Config
    from mlc_vla.sample import PiZeroRunner

    config = Pi0Config(**DUMMY_KWARGS)
    try:
        return PiZeroRunner(config, target="c")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PiZeroRunner 'c' 目标不可用：{exc}")


@pytest.fixture
def loaded(runner):
    """随机权重 + 一个 prefix_embs，返回 (params, prefix_embs)。"""
    rng = np.random.default_rng(0)
    arrays = []
    for _name, p in runner.named_params:
        shape = [int(s) for s in p.shape]
        if p.dtype.startswith(("int", "uint")):
            arrays.append(np.zeros(shape, p.dtype))
        else:
            arrays.append((0.02 * rng.standard_normal(shape)).astype(p.dtype))
    params = runner.to_params(arrays)
    cfg = runner.config
    prefix = (0.02 * rng.standard_normal((1, cfg.prefix_len, cfg.vlm.width))).astype("float32")
    return params, prefix


def _cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_cublas_guard_off_on_cpu(runner):
    """非 CUDA 目标：cuBLAS 守卫恒关（Chamleon 在 CPU/无扩展环境不应误开）。"""
    assert runner.cublas is False


def test_sample_output_shape_and_finite(runner, loaded):
    params, prefix = loaded
    cfg = runner.config
    out = runner.sample(params, prefix, num_steps=cfg.num_denoise_steps, seed=7)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_sample_graph_matches_host_loop(runner, loaded):
    """图内环（sample_graph）与宿主环（sample）在 fp32 下应近乎一致。"""
    params, prefix = loaded
    cfg = runner.config
    noise = np.random.default_rng(3).standard_normal(
        (1, cfg.action_horizon, cfg.action_dim)
    ).astype("float32")

    host = runner.sample(params, prefix, noise=noise, num_steps=cfg.num_denoise_steps)
    graph = runner.sample_graph(params, prefix, noise=noise)

    assert graph.shape == host.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert _cosine(host, graph) >= 0.99999


def test_sample_accepts_prefix_pad(runner, loaded):
    """prefix_pad（部分有效 + 右侧 padding）路径可运行且输出有限。"""
    params, prefix = loaded
    cfg = runner.config
    prefix_pad = np.zeros((cfg.prefix_len,), dtype=np.float32)
    prefix_pad[: cfg.prefix_len // 2] = 1.0  # 前一半有效

    out = runner.sample(params, prefix, num_steps=cfg.num_denoise_steps,
                        seed=1, prefix_pad=prefix_pad)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert np.all(np.isfinite(out))
