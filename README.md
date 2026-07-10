# MLC-VLA

> **一句话定位**：MLC-VLA 是 **TVM 的第二个垂直领域实现**（与 [MLC LLM](https://github.com/mlc-ai/mlc-llm) 平级）——把 TVM 通用 ML 编译器底座针对 **VLA（Vision-Language-Action）/ 流匹配动作生成** 场景做满配。第一个落地目标是 **π0.5**。

```
TVM（通用编译器底座：Relax / TIRX / VM / Codegen / Disco）
  ├── MLC LLM   （垂直实现①：自回归文本生成）
  └── MLC-VLA   （垂直实现②：多模态 → 流匹配动作生成）   ← 本项目
```

设计原则：**不 fork 编译器内核，只在 TVM 扩展点上注册 VLA 专用内容；最大化复用、最小化新建。**

---

## 为什么需要它

π0.5 不是普通 LLM，而是 **VLM 主干（SigLIP 视觉 + Gemma 语言）+ 流匹配动作专家** 的复合体。
相比文本 LLM，它有三处现有 LLM 推理栈无法直接覆盖的特性，正是 MLC-VLA 需要新建的部分：

1. **双专家联合注意力（Mixture-of-Transformers）**：prefix（视觉/语言）与 suffix（动作）由不同专家权重处理，却共享一次注意力。
2. **flow-matching 去噪循环**：固定步数（N≈10）的连续动作向量迭代生成，而非自回归采样。
3. **连续动作 I/O 与实时控制 Engine**：输入机器人状态、输出连续动作 chunk，固定 shape、强调确定性低延迟。

而视觉编码与语言 prefill 段与 LLM 高度同构，可大量复用 TVM / MLC LLM 的建模与算子基建。

---

## 项目结构

```
mlc-vla/
├── README.md                       # 本文件（项目入口）
├── docs/
│   ├── arch.md                     # 架构设计（定位 / 模型拆解 / 复用矩阵 / 路线图）
│   ├── M0.md                       # M0 阶段落地细节与现状
│   └── M1.md                       # M1/M1+：prefix KV 固化 + suffix-only 去噪 + 端到端评测
└── python/
    ├── pyproject.toml
    └── mlc_vla/
        ├── compile.py              # 导出 IRModule + 编译 + 冒烟
        ├── compare.py              # 与 openpi 单步 v_t 数值对拍脚手架
        └── model/pi0/
            ├── pi0_config.py       # 双专家 Gemma + SigLIP + 动作维度配置
            ├── gemma_dual.py       # 双专家联合注意力 / RoPE / 门控 FFN / adaRMS
            ├── siglip.py           # SigLIP 视觉塔 + multi-modal projector
            ├── pi0_model.py        # embed_image / embed_language / denoise_step + 导出 spec
            └── pi0_loader.py       # openpi → MLC 权重映射（骨架 + 映射规则）
```

---

## 快速开始

前置：可用的 TVM（Unity/Relax，含 `tvm.relax.frontend.nn`）。

```bash
# 1) 配置 PYTHONPATH（指向已构建的 TVM 与本项目）
export PYTHONPATH=/home/zhangxa/codes/edgeLLM/tvm/python:$(pwd)/python:$PYTHONPATH

# 2) 打印导出的 relax IRModule（dummy 小尺寸，秒级）
python -m mlc_vla.compile --dummy --dump-ir

# 3) 冒烟：随机权重跑通单步去噪 denoise_step
python -m mlc_vla.compile --dummy --smoke --target c    # 已验证 ✅
python -m mlc_vla.compile --smoke --target c            # 全尺寸

# 4) 与 openpi 单步对拍（需装 openpi + 补全 loader）
python -m mlc_vla.compare --mode A
```

> **后端说明**：若 TVM 构建未注册 `target.build.llvm` 或无 CUDA 设备，CPU 验证走 `--target c`
> （`zero` pipeline 合法化 → 本机 gcc 编译 `.so` → 加载运行）。全尺寸 C 编译较慢，dummy 秒级。
> 链上 LLVM 或有 GPU 后可直接切 `--target llvm` / `--target cuda`。详见 `docs/M0.md`。

---

## 当前状态（M0）

**M0 目标**：π0.5 单步前向在 TVM 跑通（视觉 + prefix embedding + 双专家联合注意力 + 单步去噪），
并与 openpi 数值对齐。

| 项 | 状态 |
|----|------|
| 包骨架 / 配置（双专家 Gemma + SigLIP） | ✅ |
| 双专家联合注意力 / RoPE / adaRMS / 门控 FFN | ✅ |
| SigLIP 视觉塔 + projector | ✅ |
| `embed_image` / `embed_language` / `denoise_step` 计算图 + 导出 spec | ✅ |
| 编译 + 冒烟（dummy `denoise_step` 跑通） | ✅ `v_t (1,4,32) finite` |
| openpi → MLC 权重 loader | 🚧 骨架（键名待对照 checkpoint 补全） |
| 单步 `v_t` 与 openpi cosine ≥ 0.99 的数值 Gate | ⬜ 待做 |

设计取舍（M0 优先正确性）：不用 PagedKVCache，走显式拼接 + 加性 mask 的 eager 注意力逐算子对齐
openpi；固定 shape 下 RoPE 表与注意力 mask 作为常量 bake 进图；dtype=float32。
CUDA Graph / KV cache 复用 / bf16 / 调度优化留到后续阶段。

---

## 路线图

- **M0**（✅）：单步前向跑通 + 数值对齐 openpi。
- **M1 / M1+**（✅）：prefix KV 固化、suffix-only 解码；host Euler 10 步 + 单步 CUDA Graph；
  分段编译、bf16 LayerNorm 修复、prefix padding；Chameleon LIBERO 端到端评测。
  TVM fp16 精度逼近 TRT（cosine 0.9988 / max_abs 0.079）。
- **M2+**：去噪环整图化消除 host 往返、调度优化、量化、端侧（Jetson Thor 等）部署。

完整设计与各阶段 DoD 见 [`docs/arch.md`](docs/arch.md)，M0 细节见 [`docs/M0.md`](docs/M0.md)，
M1/M1+ 与评测结论见 [`docs/M1.md`](docs/M1.md)。

---

## 参考

- π0.5 参考实现：`model_optimizer/third_party/openpi`（JAX/Flax 与 PyTorch 两套）。
- 底座：`edgeLLM/tvm`（Apache TVM Unity/Relax）、`edgeLLM/mlc-llm`（垂直实现①，建模范式参考）。
