# MLC-VLA

> MLC-VLA 是 **TVM 的第二个垂直领域实现**（与 [MLC LLM](https://github.com/mlc-ai/mlc-llm) 平级）——把 TVM 通用 ML 编译器底座针对 **VLA（Vision-Language-Action）/ 流匹配动作生成** 场景做满配。第一个落地目标是 **π0.5**。

```
TVM（通用编译器底座：Relax / TIR / VM / Codegen / Disco）
  ├── MLC LLM   （垂直实现①：自回归文本生成）
  └── MLC-VLA   （垂直实现②：多模态 → 流匹配动作生成）   ← 本项目
```

设计原则：**不 fork 编译器内核，只在 TVM 扩展点上注册 VLA 专用内容；最大化复用、最小化新建。**

---

## 为什么需要它

π0.5 不是普通 LLM，而是 **VLM 主干（SigLIP 视觉 + Gemma 语言）+ 流匹配动作专家** 的复合体。
相比文本 LLM，有三处现有 LLM 推理栈无法直接覆盖：

1. **双专家联合注意力（Mixture-of-Transformers）**：prefix（视觉/语言）与 suffix（动作）用不同专家权重，却共享一次注意力。
2. **flow-matching 去噪循环**：固定步数（N≈10）的连续动作向量迭代，而非自回归采样。
3. **连续动作 I/O 与实时控制**：输入机器人状态、输出动作 chunk，固定 shape、强调确定性低延迟。

视觉编码与语言 prefill 与 LLM 高度同构，可大量复用 TVM / MLC LLM 的建模与算子基建。

---

## 推理路径（当前默认：M1）

```
图像 / 语言 ──► [可选] embed_* ──► prefix_embs
                                      │
                                 prefill ──► 固化 prefix K/V
                                      │
              ┌───────────────────────┘
              ▼
   denoise_step_kv × N   或   denoise_loop_kv（图内 Euler 整环）
              │
              ▼
           actions
```

| 导出函数 | 作用 |
|----------|------|
| `embed_image` / `embed_language` | SigLIP + 语言 embedding（可分段编译） |
| `denoise_step` | M0：每步重算 prefix+suffix 联合注意力 |
| `prefill` | M1：prefix 一次前向，固化各层 K/V |
| `denoise_step_kv` | M1：suffix-only 单步，读固化 K/V |
| `denoise_loop_kv` | M1+：图内 N 步 Euler，可整段 CUDA Graph |

宿主编排见 `PiZeroRunner`（`sample.py`）：`sample()` 逐步环，`sample_graph()` 走 `denoise_loop_kv`。

---

## 项目结构

```
mlc-vla/
├── README.md
├── docs/
│   ├── arch.md                 # 架构设计 / 复用矩阵 / 路线图
│   ├── M0.md                   # M0：单步前向打通
│   └── M1.md                   # M1/M1+：KV 固化、loop、Chameleon 评测
└── python/
    ├── pyproject.toml
    └── mlc_vla/
        ├── compile.py          # 导出 IRModule + 编译 + 冒烟
        ├── compile_quant.py    # group 量化编译
        ├── quant.py            # GroupQuantize + 预设（q4f16_1 等）
        ├── sample.py           # PiZeroRunner：prefill + 宿主/图内去噪环
        ├── openpi_ref.py       # 自包含 / openpi 参考前向
        ├── compare.py          # vs openpi 单步对拍（mode A/B，可 --kv）
        ├── compare_kv.py       # M1 vs M0 数值对拍
        ├── compare_loop.py     # denoise_loop_kv vs 宿主逐步环
        ├── compare_pad.py      # prefix padding + mask 自洽
        ├── compare_embed.py    # embed bf16 vs fp32
        ├── compare_quant.py    # 量化 vs 全精度对拍
        ├── bench_kv.py         # prefill / denoise_step_kv 测速
        └── model/pi0/
            ├── pi0_config.py   # 双专家 Gemma + SigLIP + 动作维
            ├── gemma_dual.py   # 联合注意力 / RoPE / 门控 FFN / adaRMS
            ├── siglip.py       # SigLIP + LayerNormF32（bf16 友好）
            ├── pi0_model.py    # 导出函数 + 分段 include
            └── pi0_loader.py   # openpi → MLC 权重映射
```

端到端 LIBERO 评测在 **Chameleon**（双进程：openpi 3.11 ↔ TVM worker 3.12），不在本仓：
`Chamleon/chameleon/runtime/pi05_tvm/`、`configs/pi05/pi05_libero_tvm_*.yaml`。

---

## 环境

- **TVM**：Unity/Relax（含 `tvm.relax.frontend.nn`），建议本仓旁路 `edgeLLM/tvm`。
- **Python**：跑 TVM / mlc-vla 推荐 **3.12**（`tvm_ffi` 常按 3.12 构建）。
- **tvm_ffi**：新版 TVM 依赖独立包；`pyproject.toml` 在 `$TVM_HOME/3rdparty/tvm-ffi/`（**不是**其下的 `python/`）：

```bash
export TVM_HOME=/path/to/tvm
export MLC_VLA_HOME=/path/to/mlc-vla
# 若尚未安装：
#   python3.12 -m pip install -e "$TVM_HOME/3rdparty/tvm-ffi"

export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:$MLC_VLA_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib

python3.12 -c "import tvm_ffi, tvm, mlc_vla; print('ok', tvm.__file__)"
```

Chameleon 侧也可 `source Chamleon/scripts/tvm_env.sh`。

---

## 快速开始

```bash
cd $MLC_VLA_HOME
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:$(pwd)/python:$PYTHONPATH

# M0：打印 IR / 冒烟（dummy 秒级；全尺寸较慢）
python -m mlc_vla.compile --dummy --dump-ir
python -m mlc_vla.compile --dummy --smoke --target c
# GPU：python -m mlc_vla.compile --smoke --target cuda

# M1：KV 路径自洽（M1 == M0）
python -m mlc_vla.compare_kv --dummy --target llvm
python -m mlc_vla.compare_kv --target cuda --dtype float16

# M1：测速（形状默认 LIBERO：prefix_len=968，10 步）
python -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 30
python -m mlc_vla.bench_kv --target cuda --dtype float16 --cuda-graph

# 图内 loop vs 宿主逐步
python -m mlc_vla.compare_loop --target cuda --dtype float16

# 与 openpi 单步对拍（需权重；档 B + M1）
python -m mlc_vla.compare --mode B --kv --target cuda --dtype float16
```

> **后端**：无 LLVM/CUDA 时用 `--target c`（zero pipeline → gcc `.so`）。CUDA 构建若遇 NVRTC 问题，`compile.py` 会默认 `TVM_CUDA_COMPILE_MODE=nvcc`。详见 `docs/M0.md`。

### 最小推理示例

```python
from pathlib import Path
from mlc_vla.model.pi0 import Pi0Config, pi0_loader
from mlc_vla.sample import PiZeroRunner

ckpt = Path("/path/to/openpi/pytorch")
config = Pi0Config.from_openpi_config(str(ckpt), dtype="float16")
runner = PiZeroRunner(config, target="cuda", cuda_graph=True)
raw = pi0_loader.load_safetensors(str(ckpt / "model.safetensors"), dtype="float32")
src = pi0_loader.load_params(config, raw, named_params=runner.named_params, dtype="float16")
params = runner.to_params([src[n] for n, _ in runner.named_params])
# prefix_embs: [1, prefix_len, 2048]（可由 TRT Vit 或 embed_* 得到）
actions = runner.sample_graph(params, prefix_embs, noise=None, seed=0)
# 或宿主逐步：actions = runner.sample(params, prefix_embs, num_steps=10, seed=0)
```

---

## 工具一览

| 模块 | 用途 |
|------|------|
| `compile` | 导出 / 编译 / `denoise_step` 冒烟 |
| `compile_quant` + `quant` | group 量化（`q4f16_1` / `q4bf16_1` / `q3f16_1`） |
| `sample.PiZeroRunner` | 编译 M1 函数并采样 |
| `compare` | vs openpi；`--kv` 走 M1 |
| `compare_kv` / `compare_loop` / `compare_pad` | M1 自洽与 loop / padding |
| `compare_embed` / `compare_quant` | embed dtype、量化误差 |
| `bench_kv` | `prefill` / `denoise_step_kv`（及可选 M0）延迟 |

更深 profiling（nsys、与 TRT layer 对照）见 Chamleon：
`docs/optimizer/pi05/trt_tvm_profile.md`、`scripts/profile_pi05_trt_tvm.sh`。

延迟差距分析与对齐/超越计划：[`docs/optimize/tvm_vs_trt.md`](docs/optimize/tvm_vs_trt.md)
（Jetson Thor / Blackwell：[`docs/optimize/tvm_in_thor.md`](docs/optimize/tvm_in_thor.md)）。

---

## 测试

`python/tests/` 是**面向对外接口的 e2e 测试**（非细粒度单测），全部走 TVM `c` 目标（CPU + gcc，
**无需 GPU**）。缺 TVM / 无 C 编译器的环境会自动 skip。

```bash
source /path/to/Chamleon/scripts/tvm_env.sh   # 配好 TVM_HOME / PYTHONPATH / TVM_LIBRARY_PATH
cd python && "$MLC_VLA_PY" -m pytest tests/ -ra
```

| 文件 | 覆盖的对外接口 |
|------|----------------|
| `test_config_interface.py` | `Pi0Config.from_openpi_config`（worker 构造路径）+ 派生量 / 双专家校验 |
| `test_compile_and_run.py` | `compile_model` + VM 执行 `prefill` / `denoise_step` / `denoise_step_kv` / `denoise_loop_kv`（shape/dtype/finite） |
| `test_numerical_gates.py` | 验收门禁：`compare_kv`（M1≡M0）、`compare_loop`（图内环≡宿主环） |
| `test_pizero_runner.py` | `PiZeroRunner.sample` / `sample_graph` / `prefix_pad`（Chamleon worker 实际驱动的类） |
| `test_cublas_guard.py` | `resolve_cublas` / `cublas_available` 三态守卫（决定 CUDA 走 cuBLAS 或回退 dlight） |

---

## 当前状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| **M0** | 双专家图、`denoise_step`、loader、vs openpi 对拍 | ✅ |
| **M1** | `prefill` + `denoise_step_kv`、host Euler、单步 CUDA Graph、M1≡M0 | ✅ |
| **M1+** | 分段 `include`、prefix pad+mask、`LayerNormF32`、`denoise_loop_kv`、Chameleon `tvm_only` | ✅ |
| **量化** | group-quant 预设 + `compare_quant` / `bench_kv --quant` | ✅ 基建；端侧调优进行中 |
| **性能** | LIBERO 上 TVM fp16 精度近 TRT；延迟仍约 2–3×（prefill + denoise 均慢） | 🔍 优化中 |

端到端精度（LIBERO，动作 vs GT，详见 `docs/M1.md`）：

| 后端 | mean_cosine | mean_max_abs | 量级 latency/sample |
|------|-------------|--------------|---------------------|
| PyTorch (openpi) | 0.9992 | 0.058 | ~165ms |
| TensorRT | 0.9991 | 0.070 | ~170ms |
| **TVM fp16** | **0.9988** | **0.079** | **~480ms**（host 逐步；loop+Graph 后需重测） |

默认评测 dtype 用 **float16**（bf16 累加误差更大）。生产路径建议 `tvm_loop=true` + `tvm_cuda_graph=true`（Chameleon YAML）。

设计取舍：固定 shape + 显式 K/V（非 PagedKV）；注意力加性 mask 对齐 openpi；Vit 在 Chamleon 评测中仍可走 TRT，本仓已具备 `embed_image` 的 TVM 路径。

---

## 路线图

- **M0**（✅）：单步前向 + 数值对齐。
- **M1 / M1+**（✅）：KV 固化、loop、padding、Chameleon 集成与三路评测。
- **下一步**：
  1. `bench_kv` + nsys + TRT `trt-profile` 拆开 prefill / denoise 热点并优化 schedule / BYOC / 量化；
  2. Vit 全迁 TVM（去掉 TRT 依赖，便于非 NVIDIA）；
  3. 预编译 engine 落盘，避免 worker 每次 JIT；
  4. 端侧（Jetson Thor 等）延迟与功耗达标。

完整设计见 [`docs/arch.md`](docs/arch.md)，阶段细节见 [`docs/M0.md`](docs/M0.md)、[`docs/M1.md`](docs/M1.md)。

---

## 参考

- π0.5 参考实现：openpi（JAX/Flax 与 PyTorch）；本仓默认对齐 LIBERO checkpoint（`action_dim=32`，`action_horizon=10`，`prefix_len=968`）。
- 底座：`edgeLLM/tvm`（Apache TVM Unity/Relax）、`edgeLLM/mlc-llm`（垂直实现①，建模/量化范式参考）。
- 端到端评测与 bench：`edgeLLM/Chamleon`。
