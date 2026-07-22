"""plannn3 端到端规划器配置（M0）。

数值与结构对齐 NIO 车端 plannn3：
- ``laser_model_export/resource/plannn3/model_config.json`` （n_embd/n_head/n_layer 等）
- ``laser_model_export/resource/plannn3/model/network.py``   （GPT 主干 + interleaved RoPE）
- ``.../model/head/trajectory_head.py``                      （TrajHead: LayerNorm + Linear）

M0 只落地 **GPT 主干 + KV-cache prefill/decode**（多相机视觉 tokenizer 等重前端延后）。
参考 mlc-vla π0.5 的姿势：纯开源 TVM Relax，固定 shape，宿主侧编排。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Plannn3Config:
    """plannn3 顶层配置。

    默认值对齐真实 checkpoint（``model_config.json``）；``prompt_len`` 为编译期固定的
    prompt token 数（视觉 1570 + navi/history/traj 若干），M0 冒烟用 ``dummy`` 缩小。
    """

    # ---- GPT 主干 ----
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 12
    bias: bool = False  # network.py 全程 bias=False

    # ---- 词表 / 序列 ----
    # traj token 词表（TrajHead.output_dim / traj_encoder.vocab_size）
    vocab_size: int = 1169
    # 编译期固定的 prompt 长度（拼接后的 token 数）；真实模型 max_sequence_size=1656
    prompt_len: int = 1600
    # 自回归解码步数（io_spec: 18 步）
    pred_times: int = 18
    max_sequence_size: int = 1656

    # ---- 数值 ----
    dtype: str = "float32"  # M0 用 fp32 便于 CPU 对拍；后续 phase 切混精
    layer_norm_eps: float = 1e-5  # network.py LayerNorm 默认 eps
    rope_theta: float = 10_000.0
    attn_neg_inf: float = -60000.0  # 对齐 relay.py decode add_mask 的大负数

    # ---- 派生 ----
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def max_seq_len(self) -> int:
        """KV-cache 预分配长度：prompt + 全部解码步。"""
        return self.prompt_len + self.pred_times

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd 必须能被 n_head 整除"
        assert self.head_dim % 2 == 0, "interleaved RoPE 需要 head_dim 为偶数"
        assert self.max_seq_len <= self.max_sequence_size, (
            f"max_seq_len={self.max_seq_len} 超过 max_sequence_size={self.max_sequence_size}"
        )

    @classmethod
    def dummy(cls) -> "Plannn3Config":
        """单测/冒烟用小尺寸配置（秒级编译）。"""
        return cls(
            n_embd=64,
            n_head=4,
            n_layer=2,
            vocab_size=32,
            prompt_len=16,
            pred_times=4,
            max_sequence_size=64,
        )
