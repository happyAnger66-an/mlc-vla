"""openpi -> MLC-VLA 权重映射（M0 骨架）。

目标：把 openpi 的 π0.5 checkpoint 参数映射到 ``Pi0Model`` 的 nn 参数命名。

nn.frontend 生成的参数名遵循属性路径，例如：
- ``vision.layers.{i}.self_attn.q_proj.weight``
- ``backbone.layers.{i}.self_attn.experts.{e}.q_proj.weight``
- ``backbone.layers.{i}.input_layernorm.{e}.weight``（普通 RMSNorm）
- ``backbone.layers.{i}.input_layernorm.{e}.modulation.weight``（adaRMS，expert-1）
- ``backbone.layers.{i}.mlp.{e}.gate_up_proj.weight`` / ``down_proj.weight``
- ``action_in_proj.weight`` / ``time_mlp_in.weight`` ...

关键转换规则（与 openpi 结构对照）：
1. **双专家 QKV**：openpi 用 einsum 权重；HF PyTorch 路径已是 q_proj/k_proj/v_proj，
   直接搬。expert-0 来自 PaliGemma language_model，expert-1 来自 gemma_expert。
2. **gate_up_proj**：openpi/HF 的 gate_proj 与 up_proj 需 concat 成一个 [2*mlp, width]。
   注意 openpi FeedForward 的 ``gating_einsum[0]`` 是 gate、``[1]`` 是 up。
3. **RMSNorm scale**：本实现 forward 用 ``(1 + scale)``，故直接搬 openpi 的原始 scale，
   **不要**像某些 HF 导出那样预先 +1。
4. **adaRMS modulation**：expert-1 的 ``input/post_attention_layernorm`` 用 Dense(3*dim)；
   映射 openpi adaRMS 的 modulation 权重（注意 scale/shift/gate 的拼接顺序为 split-3）。
5. **patch_embedding**：openpi/HF 是 conv2d 权重 [hidden,3,14,14]；本实现是 Linear
   [hidden, 14*14*3]，需 permute 到 [hidden, 14,14,3] 再 flatten（与 _patchify 的
   [B, side, side, p, p, c] 展平顺序一致）。
6. **embed_tokens**：来自 PaliGemma language_model.embed_tokens（注意本实现在 forward
   里乘 sqrt(width)，权重本身直接搬）。

TODO(M0)：需对照实际 checkpoint（orbax/safetensors）键名补全 ``NAME_MAP``。
建议从 ``openpi`` 的 PyTorch ``state_dict()`` 导出后逐键映射，最稳。
"""

from __future__ import annotations

from typing import Callable, Dict  # noqa: UP035

import numpy as np

from .pi0_config import Pi0Config

# 形如 {mlc_param_name: (src_key, transform_fn)} 的映射表；M0 待补全。
NameMap = Dict[str, "SourceSpec"]


class SourceSpec:
    """描述一个目标参数如何从源 state_dict 构造。"""

    def __init__(self, keys, fn: Callable[..., np.ndarray] | None = None):
        # keys: 单个源键或多个源键（如 gate/up 需要拼接）
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.fn = fn or (lambda *xs: xs[0])

    def build(self, src: Dict[str, np.ndarray]) -> np.ndarray:
        return self.fn(*[src[k] for k in self.keys])


def _concat_gate_up(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    # 目标 Linear 权重布局 [out=2*mlp, in=width]；gate 在前、up 在后
    return np.concatenate([gate, up], axis=0)


def _conv_to_linear(conv_w: np.ndarray) -> np.ndarray:
    # conv [hidden, c, p, p] -> linear [hidden, p*p*c]，匹配 _patchify 的 (p,p,c) 展平
    hidden, c, p, _ = conv_w.shape
    w = np.transpose(conv_w, (0, 2, 3, 1))  # [hidden, p, p, c]
    return w.reshape(hidden, p * p * c)


def build_name_map(config: Pi0Config) -> NameMap:
    """返回 MLC 参数名 -> SourceSpec 的映射（M0 骨架，键名待对照 checkpoint 补全）。"""
    raise NotImplementedError(
        "M0 loader 骨架：请对照 openpi 实际 checkpoint 键名补全。"
        "推荐流程：torch state_dict -> numpy dict -> 按本文件注释的 6 条规则映射。"
    )


def load_params(config: Pi0Config, src: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """按映射表构造 MLC-VLA 参数字典。"""
    name_map = build_name_map(config)
    return {name: spec.build(src) for name, spec in name_map.items()}
