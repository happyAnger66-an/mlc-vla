"""Group 权重量化（vendored from mlc-llm ``group_quantization.py``）。

为什么 vendor：mlc-llm 顶层 ``import mlc_llm`` 需要已编译的 ``libmlc_llm_module.so``，
本环境未构建；而我们只需要其 group-quant 的纯 Relax nn 前端逻辑。这里把
``GroupQuantize`` / ``GroupQuantizeLinear`` / ``GroupQuantizeEmbedding`` 及所需 utils
按原实现搬入，去掉 tensor-parallel / MoE 分支，保持数值一致。

对外接口：
    PRESETS[name] -> GroupQuantize        # 如 "q4bf16_1"（int4, group=32, NK, bf16）
    quant.quantize_model(model, qmap, "") # mutate nn.Linear/nn.Embedding -> 量化层
    quant.quantize_weight(fp_weight_tensor, output_transpose=...) -> [q_weight, q_scale]

上层封装见 ``mlc_vla.compile_quant``。
"""

from __future__ import annotations

import dataclasses
import logging
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union  # noqa: UP035

from tvm import DataType, DataTypeCode, IRModule, relax, te, tirx, topi
from tvm.relax.frontend import nn
from tvm.runtime import Tensor
from tvm.s_tir import dlight as dl
from tvm.target import Target

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# QuantizeMapping（vendored from mlc_llm.loader.mapping）
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class QuantizeMapping:
    """fp 权重名 -> 量化后目标名列表 + 量化函数。未出现的参数视为不量化。"""

    param_map: dict[str, List[str]]  # noqa: UP006
    map_func: dict[str, Callable[[Tensor], List[Tensor]]]  # noqa: UP006


# --------------------------------------------------------------------------- #
# utils（vendored from mlc_llm.quantization.utils，去掉 tensor-parallel）
# --------------------------------------------------------------------------- #
def convert_uint_to_float(weight, bits, num_elem_per_storage, storage_dtype, model_dtype,
                          axis=-1, out_shape=None, ft_reorder=False):
    tir_bin_mask = tirx.const((1 << bits) - 1, storage_dtype)
    if out_shape is None:
        out_shape = weight.shape
        out_shape[axis] *= num_elem_per_storage
    axis = axis if axis >= 0 else len(out_shape) + axis
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tirx.bitwise_and(
            tirx.shift_right(
                weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1:]),
                ((idx[axis] % num_elem_per_storage) * bits).astype(storage_dtype),
            ),
            tir_bin_mask,
        ).astype(model_dtype),
    )


def pack_weight(weight, axis, num_elem_per_storage, weight_dtype, storage_dtype, out_shape=None):
    assert weight.dtype == storage_dtype
    shape = weight.shape
    if axis < 0:
        axis += len(shape)
    k = shape[axis]
    if out_shape is None:
        out_shape = (*shape[:axis], tirx.ceildiv(k, num_elem_per_storage), *shape[axis + 1:])
    r = te.reduce_axis((0, num_elem_per_storage), name="r")
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tirx.sum(
            tirx.if_then_else(
                idx[axis] * num_elem_per_storage + r < k,
                weight(*idx[:axis], idx[axis] * num_elem_per_storage + r, *idx[axis + 1:])
                << (r * DataType(weight_dtype).bits),
                tirx.const(0, storage_dtype),
            ),
            axis=r,
        ),
        name="packed_weight",
    ).astype(storage_dtype)


def is_final_fc(name: str) -> bool:
    return name in ["head", "lm_head", "lm_head.linear", "embed_out"]


def compile_quantize_func(mod: IRModule, device) -> Callable:
    device_type = device._DEVICE_TYPE_TO_NAME[device.dlpack_device_type()]  # noqa: SLF001
    if device_type in ["cuda", "rocm", "metal", "vulkan", "opencl"]:
        target = Target.current()
        if target is None:
            target = Target.from_device(device)
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
            )(mod)
    elif device_type == "cpu":
        target = "llvm"
        mod = relax.transform.LegalizeOps()(mod)
    else:
        raise NotImplementedError(f"Device type {device_type} is not supported")
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)
    return vm["main"]


# --------------------------------------------------------------------------- #
# GroupQuantize
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GroupQuantize:
    name: str
    kind: str
    group_size: int
    quantize_dtype: Literal["int3", "int4", "int8"]
    storage_dtype: Literal["uint32"]
    model_dtype: Literal["float16", "float32", "bfloat16"]
    linear_weight_layout: Literal["KN", "NK"]
    quantize_embedding: bool = True
    quantize_final_fc: bool = True

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0
    tensor_parallel_shards: int = 0

    def __post_init__(self):
        assert self.kind == "group-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        model_dtype = DataType(self.model_dtype)
        assert quantize_dtype.type_code == DataTypeCode.INT
        assert storage_dtype.type_code == DataTypeCode.UINT
        assert model_dtype.type_code in (DataTypeCode.FLOAT, DataTypeCode.BFLOAT)
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")
        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits
        if self.group_size % self.num_elem_per_storage != 0:
            raise ValueError("Group size should be divisible by numbers of elements per storage")
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1
        self.linear_quant_axis = 0 if self.linear_weight_layout == "KN" else 1
        self._quantize_func_cache = {}

    def quantize_model(self, model: nn.Module, quant_map: QuantizeMapping, name_prefix: str) -> nn.Module:
        config = self

        class _Mutator(nn.Mutator):
            def visit_module(self, name: str, node: nn.Module) -> Any:
                if getattr(node, "no_quantization", False):
                    return node
                if isinstance(node, nn.Linear) and (not is_final_fc(name) or config.quantize_final_fc):
                    weight_name = f"{name}.weight"
                    quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    quant_map.map_func[weight_name] = partial(
                        config.quantize_weight,
                        output_transpose=config.linear_weight_layout == "KN",
                    )
                    return GroupQuantizeLinear.from_linear(node, config)
                if isinstance(node, nn.Embedding) and config.quantize_embedding:
                    weight_name = f"{name}.weight"
                    quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    quant_map.map_func[weight_name] = config.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, config)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        return _Mutator().visit(name_prefix, model)

    def _dequantize(self, weight, scale, axis, out_shape=None):
        tir_max_int = tirx.const(self.max_int_value, self.model_dtype)
        float_weight = convert_uint_to_float(
            weight, DataType(self.quantize_dtype).bits, self.num_elem_per_storage,
            self.storage_dtype, self.model_dtype, axis=axis, out_shape=out_shape,
        )
        if out_shape is None:
            out_shape = weight.shape
            out_shape[axis] *= self.num_elem_per_storage
        axis = axis if axis >= 0 else len(out_shape) + axis
        return te.compute(
            shape=out_shape,
            fcompute=lambda *idx: tirx.Mul(
                tirx.Sub(float_weight(*idx), tir_max_int),
                scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1:]),
            ),
            name="dequantize",
        )

    def quantize_weight(self, weight: Tensor, axis: int = -1, output_transpose: bool = False) -> List[Tensor]:  # noqa: UP006
        device = weight.device
        device_type = device._DEVICE_TYPE_TO_NAME[device.dlpack_device_type()]  # noqa: SLF001
        axis = axis if axis >= 0 else len(weight.shape) + axis

        def _create_quantize_func() -> IRModule:
            bb = relax.BlockBuilder()
            weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, weight.dtype))
            with bb.function(name="main", params=[weight_var]):
                with bb.dataflow():
                    lv = bb.emit_te(self._quantize, weight_var, axis, output_transpose)
                    gv = bb.emit_output(lv)
                bb.emit_func_output(gv)
            return bb.finalize()

        key = f"({weight.shape}, {weight.dtype}, {device_type}, axis={axis}, output_transpose={output_transpose})"
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            quantize_func = compile_quantize_func(_create_quantize_func(), device=device)
            self._quantize_func_cache[key] = quantize_func
        return quantize_func(weight)

    def _quantize(self, weight: te.Tensor, axis: int = -1, output_transpose: bool = False) -> Tuple[te.Tensor, te.Tensor]:  # noqa: UP006
        max_int = tirx.const(self.max_int_value, self.model_dtype)
        shape = weight.shape
        axis = axis if axis >= 0 else len(shape) + axis
        k = shape[axis]
        r = te.reduce_axis((0, self.group_size), name="r")
        num_group = tirx.ceildiv(k, self.group_size)
        scale_shape = (*shape[:axis], num_group, *shape[axis + 1:])
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda *idx: te.max(
                tirx.if_then_else(
                    idx[axis] * self.group_size + r < k,
                    te.abs(weight(*idx[:axis], idx[axis] * self.group_size + r, *idx[axis + 1:])),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        scale = te.compute(
            scale_shape,
            lambda *idx: max_abs(*idx).astype(self.model_dtype) / max_int,
            name="scale",
        )
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda *idx: tirx.min(
                tirx.max(
                    tirx.round(
                        weight(*idx) / scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1:]) + max_int
                    ),
                    tirx.const(0, self.model_dtype),
                ),
                max_int * 2,
            ).astype(self.storage_dtype),
        )
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (*shape[:axis], num_storage, *shape[axis + 1:])
        quantized_weight = pack_weight(
            scaled_weight, axis=axis, num_elem_per_storage=self.num_elem_per_storage,
            weight_dtype=self.quantize_dtype, storage_dtype=self.storage_dtype,
            out_shape=quantized_weight_shape,
        )
        if output_transpose:
            if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                raise ValueError("Does not support transpose output quantized weight with ndim != 2")
            quantized_weight = topi.transpose(quantized_weight)
            scale = topi.transpose(scale)
        return quantized_weight, scale


class GroupQuantizeLinear(nn.Module):
    def __init__(self, in_features, out_features, config: GroupQuantize, bias=True, out_dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        num_group = tirx.ceildiv(in_features, config.group_size)
        if config.linear_weight_layout == "KN":
            self.q_weight = nn.Parameter(
                (config.num_storage_per_group * num_group, out_features), config.storage_dtype)
            self.q_scale = nn.Parameter((num_group, out_features), config.model_dtype)
        else:
            self.q_weight = nn.Parameter(
                (out_features, config.num_storage_per_group * num_group), config.storage_dtype)
            self.q_scale = nn.Parameter((out_features, num_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), config.model_dtype if out_dtype is None else out_dtype)
        else:
            self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: GroupQuantize) -> "GroupQuantizeLinear":
        out_features, in_features = src.weight.shape
        q = GroupQuantizeLinear(
            in_features=in_features, out_features=out_features, config=config,
            bias=getattr(src, "bias", None) is not None, out_dtype=src.out_dtype,
        )
        if q.bias is not None:
            q.bias.attrs = src.bias.attrs
        return q

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(  # noqa: SLF001
                weight, scale, axis=self.config.linear_quant_axis,
                out_shape=(
                    [
                        tirx.IntImm("int64", self.out_features) if isinstance(self.out_features, int) else weight.shape[0],
                        tirx.IntImm("int64", self.in_features),
                    ]
                    if self.config.linear_weight_layout == "NK"
                    else [
                        tirx.IntImm("int64", self.in_features),
                        tirx.IntImm("int64", self.out_features) if isinstance(self.out_features, int) else weight.shape[1],
                    ]
                ),
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if self.config.linear_weight_layout == "NK":
            w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype


class GroupQuantizeEmbedding(nn.Module):
    def __init__(self, num: Union[int, tirx.Var], dim: int, config: GroupQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        num_group = tirx.ceildiv(dim, config.group_size)
        self.q_weight = nn.Parameter((num, config.num_storage_per_group * num_group), config.storage_dtype)
        self.q_scale = nn.Parameter((num, num_group), config.model_dtype)

    @staticmethod
    def from_embedding(embedding: nn.Embedding, config: GroupQuantize) -> "GroupQuantizeEmbedding":
        num, dim = embedding.weight.shape
        return GroupQuantizeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):
        w = nn.op.tensor_expr_op(
            lambda weight, scale: self.config._dequantize(  # noqa: SLF001
                weight, scale, axis=-1,
                out_shape=[
                    tirx.IntImm("int64", self.num) if isinstance(self.num, int) else weight.shape[0],
                    tirx.IntImm("int64", self.dim),
                ],
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0), shape=[*x.shape, self.dim])


# --------------------------------------------------------------------------- #
# 预设（对齐 mlc-llm QUANTIZATION 注册表）
# --------------------------------------------------------------------------- #
PRESETS: dict[str, GroupQuantize] = {
    "q4bf16_1": GroupQuantize(
        name="q4bf16_1", kind="group-quant", group_size=32, quantize_dtype="int4",
        storage_dtype="uint32", model_dtype="bfloat16", linear_weight_layout="NK",
        quantize_embedding=False, quantize_final_fc=True,
    ),
    "q4f16_1": GroupQuantize(
        name="q4f16_1", kind="group-quant", group_size=32, quantize_dtype="int4",
        storage_dtype="uint32", model_dtype="float16", linear_weight_layout="NK",
        quantize_embedding=False, quantize_final_fc=True,
    ),
    "q3f16_1": GroupQuantize(
        name="q3f16_1", kind="group-quant", group_size=40, quantize_dtype="int3",
        storage_dtype="uint32", model_dtype="float16", linear_weight_layout="NK",
        quantize_embedding=False, quantize_final_fc=True,
    ),
}


def get_quant(name: str) -> GroupQuantize:
    if name not in PRESETS:
        raise KeyError(f"未知量化预设 {name!r}，可选：{list(PRESETS)}")
    return PRESETS[name]
