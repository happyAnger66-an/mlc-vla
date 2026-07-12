"""数值验收门禁（把项目既有 compare_* 对拍脚本固化为回归测试）。

这两条是 MLC-VLA 重构正确性的核心 e2e gate：
- ``compare_kv``：M1（prefill 固化 K/V + suffix-only denoise）≡ M0（联合 denoise）。
- ``compare_loop``：图内整段 Euler 环（denoise_loop_kv）≡ 宿主逐步环（denoise_step_kv）。

均用相同随机权重 + 相同输入即可验证（无需真实 checkpoint / torch），CPU fp32 下应近乎等价。
"""

from __future__ import annotations

import dataclasses


def test_m1_equivalent_to_m0_single_and_loop(dummy_config):
    """compare_kv：单步与 3 步全环，M1 与 M0 cosine ≥ 0.99。"""
    from mlc_vla import compare_kv

    cfg = dataclasses.replace(dummy_config, dtype="float32")
    assert compare_kv.run(cfg, "c", loop_steps=3) is True


def test_ingraph_loop_equivalent_to_host_loop():
    """compare_loop：denoise_loop_kv 与宿主环 fp32 下近乎 bit-identical（阈值 0.99999）。"""
    from mlc_vla import compare_loop

    assert compare_loop.run("c", "float32", steps=3, seed=0) is True
