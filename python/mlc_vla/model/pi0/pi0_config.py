"""Configuration for the π0 / π0.5 model.

数值与结构对齐 openpi：
- ``openpi/src/openpi/models/gemma.py``      （Gemma 各 variant 配置）
- ``openpi/src/openpi/models/pi0_config.py`` （Pi0Config 默认值）
- ``openpi/src/openpi/models_pytorch/gemma_pytorch.py`` （SigLIP / PaliGemma 维度）
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple  # noqa: UP035


@dataclasses.dataclass
class GemmaExpertConfig:
    """单个 Gemma 专家的配置（对应 openpi ``gemma.Config``）。

    注意：双专家必须共享 ``head_dim`` / ``num_heads`` / ``num_kv_heads`` / ``depth``，
    才能做联合注意力（见 openpi ``gemma.Attention``）。
    """

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @staticmethod
    def from_variant(variant: str) -> "GemmaExpertConfig":
        if variant == "gemma_2b":  # PaliGemma backbone (expert-0)
            return GemmaExpertConfig(
                width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256
            )
        if variant == "gemma_300m":  # action expert (expert-1)
            return GemmaExpertConfig(
                width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
            )
        if variant == "dummy":  # 小尺寸，单测/对拍用
            return GemmaExpertConfig(
                width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16
            )
        raise ValueError(f"Unknown gemma variant: {variant}")


@dataclasses.dataclass
class SiglipConfig:
    """SigLIP So400m/14 视觉塔配置（PaliGemma 默认）。"""

    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-6
    # PaliGemma 的 multi-modal projector：vision hidden -> gemma width
    projection_dim: int = 2048

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


@dataclasses.dataclass
class Pi0Config:
    """π0.5 顶层配置。"""

    # 专家
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # 动作空间
    # 默认对齐随附的 LIBERO π0.5 checkpoint（models/openpi/pytorch/config.json:
    # action_dim=32, action_horizon=10）。其它 checkpoint 可用 from_openpi_config 覆盖。
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200  # pi05 默认 200（pi0 为 48）

    # π0.5 特性：动作专家用 adaRMSNorm 注入 flow-matching timestep
    pi05: bool = True

    # 视觉
    siglip: SiglipConfig = dataclasses.field(default_factory=SiglipConfig)

    # 输入图像路数（base + 双腕）
    image_keys: Tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # 数值
    dtype: str = "float32"  # M0 用 float32 便于 CPU 对拍；后续 phase 切 bf16
    # 注意力 QK^T logits 的输入 dtype：
    #   "float32"（默认，安全）——q/k 以 fp32 相乘（openpi 严格对齐，但走非 tensor-core sgemm，慢）
    #   模型 dtype（如 "float16"）——q/k 降到 fp16 走 tensor-core，softmax 前仍以 fp32 累加输出，
    #     与 TRT `_gemm_mha_v2` 同策略；prefill QK 显著提速，精度需过 compare gate。
    attn_logits_dtype: str = "float32"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    attn_neg_inf: float = -2.3819763e38  # 对齐 openpi big_neg
    # 时间正弦编码
    time_min_period: float = 4e-3
    time_max_period: float = 4.0

    # flow-matching 去噪步数（M0 固定常量，便于后续整图化）
    num_denoise_steps: int = 10

    kwargs: Dict[str, object] = dataclasses.field(default_factory=dict)  # noqa: UP006

    # ---- 派生配置 ----
    @property
    def vlm(self) -> GemmaExpertConfig:
        return GemmaExpertConfig.from_variant(self.paligemma_variant)

    @property
    def action_expert(self) -> GemmaExpertConfig:
        return GemmaExpertConfig.from_variant(self.action_expert_variant)

    @property
    def num_images(self) -> int:
        return len(self.image_keys)

    @property
    def prefix_len(self) -> int:
        """M0 固定 shape：prefix = 图像 token + 语言 token。"""
        return self.num_images * self.siglip.num_patches + self.max_token_len

    @property
    def suffix_len(self) -> int:
        """π0.5 的 suffix 仅含动作 token（state 进了离散语言 token）。"""
        return self.action_horizon

    def experts(self) -> List[GemmaExpertConfig]:  # noqa: UP006
        return [self.vlm, self.action_expert]

    @classmethod
    def from_openpi_config(cls, path: str, **overrides) -> "Pi0Config":
        """从 openpi checkpoint 目录的 ``config.json`` 读取关键字段构造配置。

        openpi PyTorch checkpoint 附带的 ``config.json`` 例：
            {"action_dim": 32, "action_horizon": 10,
             "paligemma_variant": "gemma_2b",
             "action_expert_variant": "gemma_300m",
             "precision": "bfloat16"}
        """
        import json
        import os

        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        kwargs: Dict[str, object] = {}  # noqa: UP006
        for key in ("action_dim", "action_horizon", "paligemma_variant", "action_expert_variant"):
            if key in cfg:
                kwargs[key] = cfg[key]
        kwargs.update(overrides)
        return cls(**kwargs)

    def __post_init__(self):
        a, b = self.vlm, self.action_expert
        assert a.head_dim == b.head_dim, "双专家 head_dim 必须一致"
        assert a.num_heads == b.num_heads, "双专家 num_heads 必须一致"
        assert a.num_kv_heads == b.num_kv_heads, "双专家 num_kv_heads 必须一致"
        assert a.depth == b.depth, "双专家 depth 必须一致"
