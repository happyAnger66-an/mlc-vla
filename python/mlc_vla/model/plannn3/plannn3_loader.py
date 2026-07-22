"""plannn3 真实 checkpoint -> MLC-VLA 权重映射（M2）。

把 NIO plannn3 训练 checkpoint（``network.Net`` 的 ``state_dict``）中的 **GPT 主干 + 轨迹头**
参数映射到 ``Plannn3Model`` 的 relax nn 参数命名，用于 M2 数值对拍。

对照 ``resource/plannn3/model/network.py`` / ``head/trajectory_head.py`` 的键名：

| 源（torch state_dict）                       | 目标（relax nn）             |
|----------------------------------------------|------------------------------|
| ``transfomer.h.{L}.ln_1.weight``             | ``h.{L}.ln_1.weight``        |
| ``transfomer.h.{L}.attn.c_attn.weight``      | ``h.{L}.attn.c_attn.weight`` |
| ``transfomer.h.{L}.attn.c_proj.weight``      | ``h.{L}.attn.c_proj.weight`` |
| ``transfomer.h.{L}.mlp.c_fc.weight``         | ``h.{L}.mlp.c_fc.weight``    |
| ``transfomer.h.{L}.mlp.c_proj.weight``       | ``h.{L}.mlp.c_proj.weight``  |
| ``transfomer.h.{L}.ln_2.weight``             | ``h.{L}.ln_2.weight``        |
| ``traj_head.ln.weight``                      | ``traj_head.ln.weight``      |
| ``traj_head.head.weight``                    | ``traj_head.head.weight``    |

`bias=False` 全程无 bias；``use_rmsnorm=false`` 时 ``ln_*`` 为 LayerNorm(仅 weight)。

**traj 嵌入（M3 补齐）**：``ids_to_embed`` 其实就是 ``embed_tokens(ids)`` 查表，故 M0 的
``embed`` 占位结构本就正确——真实权重来自 ``traj_encoder.embed_tokens.weight``
（``nn.Embedding(vocab_size, hidden_size)``），本 loader 已映射：``embed.weight`` ←
``traj_encoder.embed_tokens.weight``。若 checkpoint 缺该键（如只交付主干），退回 ``allow_missing``。

**仍在宿主侧（按 arch.md 设计，非 TVM engine）**：多相机 DINOv3 外层编排、navi/history tokenizer、
数据相关 crop/resize、PCA 轨迹反解——这些是宿主/PVA 预处理，不属于 TVM 固定 shape 图。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence  # noqa: UP035

import numpy as np

from .plannn3_config import Plannn3Config

NameMap = Dict[str, "SourceSpec"]

# 主干各层需要直接搬运（源前缀 transfomer.h.{L}. -> 目标 h.{L}.）的子键
_BLOCK_KEYS = (
    "ln_1.weight",
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "ln_2.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
_HEAD_KEYS = ("ln.weight", "head.weight")

# traj token embedding：ids_to_embed == embed_tokens(ids)，源自 traj_encoder.embed_tokens
_TRAJ_EMBED_SRC = "traj_encoder.embed_tokens.weight"

# checkpoint 若缺 traj 嵌入（如只交付主干）可退回；默认能映射到真实权重时不会用到
DEFAULT_ALLOW_MISSING = ("embed.weight",)


class SourceSpec:
    """描述一个目标参数如何从源 state_dict 构造（对齐 ``pi0_loader.SourceSpec``）。"""

    def __init__(self, keys, fn: Callable[..., np.ndarray] | None = None):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.fn = fn or (lambda *xs: xs[0])

    def build(self, src: Dict[str, np.ndarray]) -> np.ndarray:
        return self.fn(*[src[k] for k in self.keys])


def build_name_map(
    config: Plannn3Config,
    src_prefix: str = "transfomer.h.",
    traj_embed_src: str = _TRAJ_EMBED_SRC,
    src_keys: Iterable[str] | None = None,
) -> NameMap:
    """返回 MLC 参数名 -> SourceSpec 的映射（GPT 主干 + 轨迹头 + traj 嵌入）。

    - ``src_prefix``：源 checkpoint 里 transformer block 列表的前缀（默认 ``transfomer.h.``，
      注意是训练代码里的拼写 ``transfomer``）。
    - ``traj_embed_src``：traj token 嵌入源键（``ids_to_embed`` == ``embed_tokens`` 查表）。
    - ``src_keys``：可选，传入 checkpoint 已有键集合；仅当 ``traj_embed_src`` 存在时才映射
      ``embed.weight``（否则退回 ``allow_missing``）。
    """
    name_map: NameMap = {}
    for layer in range(config.n_layer):
        for sub in _BLOCK_KEYS:
            name_map[f"h.{layer}.{sub}"] = SourceSpec(f"{src_prefix}{layer}.{sub}")
    for sub in _HEAD_KEYS:
        name_map[f"traj_head.{sub}"] = SourceSpec(f"traj_head.{sub}")
    if src_keys is None or traj_embed_src in set(src_keys):
        name_map["embed.weight"] = SourceSpec(traj_embed_src)
    return name_map


def _normalize_state_dict(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """去掉 DDP ``module.`` 前缀、``gamma`` -> ``weight`` 重命名（对齐参考 loader）。"""
    out: Dict[str, np.ndarray] = {}
    for k, v in sd.items():
        key = k[len("module.") :] if k.startswith("module.") else k
        key = key.replace("gamma", "weight") if "gamma" in key else key
        out[key] = v
    return out


def load_state_dict(path: str, dtype: str = "float32") -> Dict[str, np.ndarray]:
    """读取 checkpoint 为 numpy dict。支持 ``.safetensors`` 与 torch ``.bin``/``.pt``。

    bf16/fp16 统一上转到 ``dtype``（默认 fp32，便于 CPU 对拍）。
    """
    np_dtype = np.dtype(dtype)
    out: Dict[str, np.ndarray] = {}
    if path.endswith(".safetensors"):
        import torch
        from safetensors.torch import load_file

        sd = load_file(path)
    else:
        import torch

        sd = torch.load(path, map_location="cpu")
        # 兼容包了一层 {"model": state_dict} / {"state_dict": ...} 的 checkpoint
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

    for k, v in sd.items():
        if v.dtype in (torch.bfloat16, torch.float16):
            v = v.to(torch.float32)
        arr = v.numpy()
        if not str(arr.dtype).startswith("int"):
            arr = arr.astype(np_dtype, copy=False)
        out[k] = arr
    return _normalize_state_dict(out)


def load_params(
    config: Plannn3Config,
    src: Dict[str, np.ndarray],
    named_params=None,
    dtype: str | None = None,
    src_prefix: str = "transfomer.h.",
    allow_missing: Iterable[str] = DEFAULT_ALLOW_MISSING,
) -> Dict[str, np.ndarray]:
    """按映射表构造 MLC-VLA 参数字典（GPT 主干 + 轨迹头）。

    - ``named_params``：可选，传入 ``model.export_tvm`` 的 ``[(name, param), ...]``，
      用于逐参数 shape/缺键断言，第一时间暴露键名/形状错误。
    - ``allow_missing``：checkpoint 无源、由调用方另行提供的目标参数（默认 ``embed.weight``）。
    """
    name_map = build_name_map(config, src_prefix=src_prefix, src_keys=src.keys())
    target_dtype = dtype or config.dtype
    params: Dict[str, np.ndarray] = {}
    for name, spec in name_map.items():
        arr = spec.build(src)
        if not str(arr.dtype).startswith("int"):
            arr = arr.astype(target_dtype, copy=False)
        params[name] = arr

    if named_params is not None:
        expected = {n: tuple(int(s) for s in p.shape) for n, p in named_params}
        allow = set(allow_missing)
        missing = set(expected) - set(params) - allow
        assert not missing, f"loader 缺少 {len(missing)} 个参数，例：{sorted(missing)[:5]}"
        for n, shp in expected.items():
            if n in params:
                got = tuple(int(s) for s in params[n].shape)
                assert got == shp, f"参数 {n} shape 不匹配：loader={got} vs model={shp}"
    return params


def to_tvm_params(named_params, params: Dict[str, np.ndarray], dev, *, allow_missing: Sequence[str] = DEFAULT_ALLOW_MISSING):
    """按 ``named_params`` 顺序把 numpy 参数打成 tvm ndarray 列表（缺项用零占位）。

    ``allow_missing`` 中的目标参数（如 ``embed.weight``）checkpoint 无源，这里用 0 占位，
    便于只跑 ``prefill``/``decode_step`` 的主干对拍；真正解码需外部提供真实 traj-embedding。
    """
    import tvm

    allow = set(allow_missing)
    out = []
    for name, p in named_params:
        shape = [int(s) for s in p.shape]
        if name in params:
            arr = params[name].astype(p.dtype, copy=False)
        elif name in allow:
            arr = np.zeros(shape, dtype=p.dtype)
        else:
            raise KeyError(f"参数 {name} 未在 loader 中提供，且不在 allow_missing。")
        out.append(tvm.runtime.tensor(arr, dev))
    return out
