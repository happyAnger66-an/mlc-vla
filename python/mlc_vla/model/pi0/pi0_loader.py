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


# ---- openpi checkpoint（PI0Pytorch state_dict）源键前缀 ----
_PWE = "paligemma_with_expert."
_PALI = _PWE + "paligemma.model."
_LM = _PALI + "language_model."  # expert-0 (PaliGemma backbone)
_EXP = _PWE + "gemma_expert.model."  # expert-1 (action expert)
_VIS = _PALI + "vision_tower.vision_model."
_PROJ = _PALI + "multi_modal_projector.linear."
# embed_tokens 与 lm_head 权重绑定（checkpoint 只存 lm_head）
_EMBED_SRC = _PWE + "paligemma.lm_head.weight"

# 两个专家在源 checkpoint 中的 layer 前缀（按 mlc-vla expert index 排列）
_EXPERT_LAYER_PREFIX = [_LM + "layers.", _EXP + "layers."]
_EXPERT_FINAL_NORM = [_LM + "norm", _EXP + "norm"]


def build_name_map(config: Pi0Config) -> NameMap:
    """返回 MLC 参数名 -> SourceSpec 的映射。

    覆盖全部 ``Pi0Model.export_tvm`` 导出的 named_params（vision / embed_tokens /
    backbone 双专家 / 动作头 / time MLP），键名对照 openpi PI0Pytorch state_dict。
    """
    depth = config.vlm.depth
    vis_layers = config.siglip.num_hidden_layers
    name_map: NameMap = {}

    # ---------- 1. SigLIP 视觉塔 + multi-modal projector ----------
    name_map["vision.patch_embedding.weight"] = SourceSpec(
        _VIS + "embeddings.patch_embedding.weight", _conv_to_linear
    )
    name_map["vision.patch_embedding.bias"] = SourceSpec(_VIS + "embeddings.patch_embedding.bias")
    name_map["vision.position_embedding"] = SourceSpec(_VIS + "embeddings.position_embedding.weight")
    for i in range(vis_layers):
        p = f"{_VIS}encoder.layers.{i}."
        t = f"vision.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            name_map[t + f"self_attn.{proj}.weight"] = SourceSpec(p + f"self_attn.{proj}.weight")
            name_map[t + f"self_attn.{proj}.bias"] = SourceSpec(p + f"self_attn.{proj}.bias")
        for fc in ("fc1", "fc2"):
            name_map[t + f"mlp.{fc}.weight"] = SourceSpec(p + f"mlp.{fc}.weight")
            name_map[t + f"mlp.{fc}.bias"] = SourceSpec(p + f"mlp.{fc}.bias")
        for ln in ("layer_norm1", "layer_norm2"):
            name_map[t + f"{ln}.weight"] = SourceSpec(p + f"{ln}.weight")
            name_map[t + f"{ln}.bias"] = SourceSpec(p + f"{ln}.bias")
    name_map["vision.post_layernorm.weight"] = SourceSpec(_VIS + "post_layernorm.weight")
    name_map["vision.post_layernorm.bias"] = SourceSpec(_VIS + "post_layernorm.bias")
    name_map["vision.multi_modal_projector.weight"] = SourceSpec(_PROJ + "weight")
    name_map["vision.multi_modal_projector.bias"] = SourceSpec(_PROJ + "bias")

    # ---------- 2. embed_tokens（与 lm_head 绑定）----------
    name_map["embed_tokens.weight"] = SourceSpec(_EMBED_SRC)

    # ---------- 3. 双专家 backbone ----------
    use_adarms = [False, True] if config.pi05 else [False, False]
    for L in range(depth):
        for e in range(2):
            src_layer = f"{_EXPERT_LAYER_PREFIX[e]}{L}."
            tgt = f"backbone.layers.{L}."
            # QKVO 投影（HF 路径已是 q/k/v/o_proj，直接搬）
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                name_map[f"{tgt}self_attn.experts.{e}.{proj}.weight"] = SourceSpec(
                    src_layer + f"self_attn.{proj}.weight"
                )
            # MLP：gate/up concat -> gate_up_proj；down 直接搬
            name_map[f"{tgt}mlp.{e}.gate_up_proj.weight"] = SourceSpec(
                (src_layer + "mlp.gate_proj.weight", src_layer + "mlp.up_proj.weight"),
                _concat_gate_up,
            )
            name_map[f"{tgt}mlp.{e}.down_proj.weight"] = SourceSpec(src_layer + "mlp.down_proj.weight")
            # 归一化：expert-0 普通 RMSNorm(scale)；expert-1 adaRMS(modulation dense)
            for ln in ("input_layernorm", "post_attention_layernorm"):
                if use_adarms[e]:
                    name_map[f"{tgt}{ln}.{e}.modulation.weight"] = SourceSpec(src_layer + f"{ln}.dense.weight")
                    name_map[f"{tgt}{ln}.{e}.modulation.bias"] = SourceSpec(src_layer + f"{ln}.dense.bias")
                else:
                    name_map[f"{tgt}{ln}.{e}.weight"] = SourceSpec(src_layer + f"{ln}.weight")
    # 末端 norm
    for e in range(2):
        if use_adarms[e]:
            name_map[f"backbone.final_norm.{e}.modulation.weight"] = SourceSpec(_EXPERT_FINAL_NORM[e] + ".dense.weight")
            name_map[f"backbone.final_norm.{e}.modulation.bias"] = SourceSpec(_EXPERT_FINAL_NORM[e] + ".dense.bias")
        else:
            name_map[f"backbone.final_norm.{e}.weight"] = SourceSpec(_EXPERT_FINAL_NORM[e] + ".weight")

    # ---------- 4. 动作头 + time MLP（顶层，键名直接一致）----------
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        name_map[f"{name}.weight"] = SourceSpec(f"{name}.weight")
        name_map[f"{name}.bias"] = SourceSpec(f"{name}.bias")

    return name_map


def load_safetensors(path: str, dtype: str = "float32") -> Dict[str, np.ndarray]:
    """把 openpi PI0Pytorch safetensors 读成 numpy dict（bf16 上转到 ``dtype``）。"""
    import torch
    from safetensors.torch import load_file

    sd = load_file(path)
    np_dtype = np.dtype(dtype)
    out: Dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if v.dtype in (torch.bfloat16, torch.float16):
            v = v.to(torch.float32)
        out[k] = v.numpy().astype(np_dtype, copy=False)
    return out


def load_params(
    config: Pi0Config,
    src: Dict[str, np.ndarray],
    named_params=None,
    dtype: str | None = None,
) -> Dict[str, np.ndarray]:
    """按映射表构造 MLC-VLA 参数字典。

    - ``named_params``：可选，传入 ``model.export_tvm`` 的 ``[(name, param), ...]``，
      用于逐参数 shape/缺键断言，第一时间暴露 concat/permute 顺序错误。
    - ``dtype``：可选，最终参数统一 cast（默认沿用 ``config.dtype``）。
    """
    name_map = build_name_map(config)
    target_dtype = dtype or config.dtype
    params: Dict[str, np.ndarray] = {}
    for name, spec in name_map.items():
        arr = spec.build(src)
        if not str(arr.dtype).startswith("int"):
            arr = arr.astype(target_dtype, copy=False)
        params[name] = arr

    if named_params is not None:
        expected = {n: tuple(int(s) for s in p.shape) for n, p in named_params}
        missing = set(expected) - set(params)
        assert not missing, f"loader 缺少 {len(missing)} 个参数，例：{sorted(missing)[:5]}"
        # 允许 loader 为超集（如只导出 denoise_step 时，vision/embed 参数用不到）
        for n, shp in expected.items():
            got = tuple(int(s) for s in params[n].shape)
            assert got == shp, f"参数 {n} shape 不匹配：loader={got} vs model={shp}"
    return params
