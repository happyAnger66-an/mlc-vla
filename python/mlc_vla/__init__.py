"""MLC-VLA: TVM 的 VLA（Vision-Language-Action）垂直领域实现。

第一个落地目标是 π0.5（PaliGemma SigLIP ViT + 双专家 Gemma + flow-matching
action expert）。本包复用 TVM 的 ``relax.frontend.nn`` 建模框架，结构上对标 MLC LLM。

M0 阶段目标：π0.5 单步前向在 TVM 跑通，并与 openpi 数值对齐。
"""

from .libinfo import __version__
