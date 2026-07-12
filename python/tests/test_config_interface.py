"""Pi0Config 对外契约（Chamleon worker 构造模型的入口）。

覆盖 ``worker.py`` 实际路径：``from_openpi_config(checkpoint_dir, dtype=..., attn_logits_dtype=...)``，
以及下游依赖的派生量（prefix_len/suffix_len/num_images）与双专家一致性校验。

这些断言本身只算派生量、不执行任何 kernel（无需 GPU），但 ``mlc_vla.model.pi0`` 包 __init__
会导入 Pi0Model（依赖 TVM），故仍在无 TVM 环境自动 skip。
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tvm")

from mlc_vla.model.pi0.pi0_config import Pi0Config  # noqa: E402


def test_default_config_derived_shapes():
    cfg = Pi0Config()
    # 3 路图像 + 200 文本 token；每图 (224/14)^2 = 256 patch
    assert cfg.num_images == 3
    assert cfg.prefix_len == 3 * cfg.siglip.num_patches + cfg.max_token_len == 3 * 256 + 200
    assert cfg.suffix_len == cfg.action_horizon == 10
    # 默认严格对齐 openpi：QK^T 走 fp32
    assert cfg.attn_logits_dtype == "float32"
    # 双专家解析后 head_dim/num_heads 一致（__post_init__ 已断言）
    assert cfg.vlm.head_dim == cfg.action_expert.head_dim


def test_from_openpi_config_reads_json_and_applies_overrides(tmp_path):
    """复刻 worker：从 checkpoint 目录 config.json 读取，并用 kwargs 覆盖 dtype/attn_logits_dtype。"""
    ckpt = tmp_path
    (ckpt / "config.json").write_text(json.dumps({
        "action_dim": 7,
        "action_horizon": 5,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "precision": "bfloat16",  # 非 Pi0Config 字段，应被安全忽略
    }))

    cfg = Pi0Config.from_openpi_config(
        str(ckpt), dtype="float16", attn_logits_dtype="float16"
    )

    # 来自 json 的字段
    assert cfg.action_dim == 7
    assert cfg.action_horizon == 5
    assert cfg.paligemma_variant == "dummy"
    # 来自 overrides（worker 的 --dtype / --attn-logits-dtype）
    assert cfg.dtype == "float16"
    assert cfg.attn_logits_dtype == "float16"
    # 派生量随 json 尺寸更新
    assert cfg.suffix_len == 5


def test_from_openpi_config_accepts_directory_or_file(tmp_path):
    payload = {"action_dim": 3, "action_horizon": 2,
               "paligemma_variant": "dummy", "action_expert_variant": "dummy"}
    (tmp_path / "config.json").write_text(json.dumps(payload))

    from_dir = Pi0Config.from_openpi_config(str(tmp_path))
    from_file = Pi0Config.from_openpi_config(str(tmp_path / "config.json"))
    assert from_dir.action_dim == from_file.action_dim == 3


def test_dual_expert_mismatch_rejected():
    """双专家 head_dim/num_heads/... 不一致时必须在构造期报错（联合注意力前提）。"""
    with pytest.raises(AssertionError):
        # gemma_2b(head_dim=256) 与 dummy(head_dim=16) 不匹配
        Pi0Config(paligemma_variant="gemma_2b", action_expert_variant="dummy")
