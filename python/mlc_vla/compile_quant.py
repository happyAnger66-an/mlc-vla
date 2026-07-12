"""pi0 group 量化编译 + 权重转换（M1+）。

流程：
    fp_model.export -> fp_named_params            # 供 pi0_loader 映射 safetensors
    quant.quantize_model(qmodel) -> 量化 IRModule + q_named_params + quant_map
    quantize_params(fp权重, quant_map) -> {q_weight/q_scale/bias/norm.weight}

量化只作用于 ``nn.Linear``（backbone q/k/v/o、gate_up/down、action_in/out_proj、
time_mlp、adaRMS dense）；RMSNorm / KV cache / 激活保持原 dtype。embed_tokens 不在
prefill/denoise 导出图中，故预设关闭 embedding 量化。
"""

from __future__ import annotations

import os

import numpy as np

from mlc_vla.model.pi0 import Pi0Config, Pi0Model
from mlc_vla.model.pi0.pi0_model import include_for
from mlc_vla.quant import QuantizeMapping, get_quant


def fp_named_params(config: Pi0Config, functions):
    """未量化模型的 named_params（name, param），供 pi0_loader 映射真实权重。"""
    m = Pi0Model(config, include=include_for(functions))
    m.to(config.dtype)
    _, named_params, _ = m.export_tvm(
        spec=m.get_default_spec(functions=functions), allow_extern=True
    )
    return named_params


def build_quant_irmodule(config: Pi0Config, functions, quant_name: str):
    """量化 pi0 并导出 (IRModule, q_named_params, quant_map)。"""
    quant = get_quant(quant_name)
    model = Pi0Model(config, include=include_for(functions))
    qmap = QuantizeMapping({}, {})
    model = quant.quantize_model(model, qmap, "")
    mod, q_named_params, _ = model.export_tvm(
        spec=model.get_default_spec(functions=functions), allow_extern=True
    )
    return mod, q_named_params, qmap, quant


def compile_model_quant(config: Pi0Config, target: str, functions, quant_name: str,
                        cuda_graph: bool = False, cublas: bool = False):
    """编译量化模型，返回 (ex, q_named_params, quant_map, quant)。"""
    import tvm
    from tvm import relax

    mod, q_named_params, qmap, quant = build_quant_irmodule(config, functions, quant_name)
    tgt = tvm.target.Target(target)
    if tgt.kind.name == "cuda":
        os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    if cublas and tgt.kind.name == "cuda":
        from mlc_vla.compile import apply_gemm_prepasses

        mod = apply_gemm_prepasses(mod, tgt)
    relax_pipeline = relax.get_default_pipeline(tgt)
    pass_cfg = {"relax.backend.use_cuda_graph": True} if (cuda_graph and tgt.kind.name == "cuda") else {}
    with tgt, tvm.transform.PassContext(config=pass_cfg):
        ex = relax.build(mod, target=tgt, relax_pipeline=relax_pipeline)
    return ex, q_named_params, qmap, quant


def quantize_params(quant, src_fp: dict, q_named_params, quant_map: QuantizeMapping, device=None):
    """把 fp 权重（keyed by fp 参数名）转成量化后 params dict（keyed by q 参数名）。

    - ``.q_weight`` / ``.q_scale``：对 fp ``.weight`` 跑 quant_map.map_func。
    - bias / RMSNorm.weight：直接从 src_fp 拷贝并 cast 到目标 dtype。

    量化 TE 固定在 **CPU（llvm）** 上跑：CUDA 的 dtype-legalizer 处理 bf16 归约临时量会失败；
    llvm 会把 bf16 合法化为 fp32 计算。产物为 numpy，运行时再载入目标 device。
    """
    import tvm

    cpu = tvm.cpu(0)  # 量化 TE 一律在 CPU 上编译/执行（规避 CUDA bf16 legalize 限制）

    rev = {}  # q 参数名 -> (fp 权重名, 在 param_map 列表中的下标)
    for base, qnames in quant_map.param_map.items():
        for i, qn in enumerate(qnames):
            rev[qn] = (base, i)

    cache: dict[str, list] = {}
    out: dict[str, np.ndarray] = {}
    for name, p in q_named_params:
        if name in rev:
            base, idx = rev[name]
            if base not in cache:
                wt = tvm.runtime.tensor(src_fp[base].astype(quant.model_dtype), cpu)
                res = quant_map.map_func[base](wt)
                cache[base] = [r.numpy() for r in res]
            out[name] = cache[base][idx]
        else:
            if name not in src_fp:
                raise KeyError(f"量化后非量化参数 {name!r} 在 fp 源权重中缺失")
            out[name] = src_fp[name].astype(p.dtype)
    return out
