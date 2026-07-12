"""端到端：``compile_model`` 编译 + VM 执行模型全部对外函数（CPU ``c`` 目标）。

验证「图能 legalize / build / 运行，且 shape/dtype 自洽、输出有限」——即 M0/M1 的打通契约。
数值等价另见 ``test_numerical_gates.py``。
"""

from __future__ import annotations

import numpy as np

from conftest import as_numpy


def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(np.asarray(arr, dtype=np.float64))))


def test_prefill_returns_kv_with_expected_shapes(kv_runtime, kv_inputs):
    cfg = kv_runtime.config
    kv = kv_runtime.vm["prefill"](kv_inputs.prefix, kv_inputs.pmask, kv_inputs.params)
    keys, values = kv[0], kv[1]
    # [1, depth, num_kv_heads, prefix_len, head_dim] 语义：至少 prefix_len 与 head_dim 出现在形状里
    kshape = tuple(int(s) for s in keys.shape)
    assert cfg.prefix_len in kshape
    assert cfg.vlm.head_dim in kshape
    assert _finite(keys.numpy()) and _finite(values.numpy())


def test_denoise_step_kv_output_shape_and_finite(kv_runtime, kv_inputs):
    cfg = kv_runtime.config
    v = kv_runtime.vm["denoise_step_kv"](
        kv_inputs.keys, kv_inputs.values, kv_inputs.x_t, kv_inputs.time_emb,
        kv_inputs.scos, kv_inputs.ssin, kv_inputs.pmask, kv_inputs.params,
    )
    out = as_numpy(v)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert str(out.dtype) == "float32"  # 速度场恒以 fp32 输出
    assert _finite(out)


def test_denoise_step_joint_output_shape_and_finite(kv_runtime, kv_inputs):
    cfg = kv_runtime.config
    v = kv_runtime.vm["denoise_step"](
        kv_inputs.prefix, kv_inputs.x_t, kv_inputs.time_emb, kv_inputs.params
    )
    out = as_numpy(v)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert _finite(out)


def test_denoise_loop_kv_runs_full_euler(kv_runtime, kv_inputs):
    """图内整段 Euler 环一次调用跑完 num_denoise_steps 步，输出即 t=0 处动作。"""
    cfg = kv_runtime.config
    out = as_numpy(kv_runtime.vm["denoise_loop_kv"](
        kv_inputs.keys, kv_inputs.values, kv_inputs.x0, kv_inputs.te_dev,
        kv_inputs.scos, kv_inputs.ssin, kv_inputs.pmask, kv_inputs.params,
    ))
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert _finite(out)
