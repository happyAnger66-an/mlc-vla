# TVM vs TensorRT：π0.5 LIBERO 延迟差距与对齐计划

> 目标：让 MLC-VLA 的 `prefill` / `denoise_step_kv`（及整环）**对齐甚至优于** Chamleon TRT 同机耗时，同时保持与 openpi 的数值门禁。
>
> 数据来源（本机 / Ada）：
> - `python -m mlc_vla.bench_kv --target cuda --dtype float16`
> - Chamleon `chameleon bench`（`configs/pi05/pi05_libero_bench*.yaml`）
> - Chamleon `trt-profile` → `output/pi05_libero_trt/profiles/{llm,denoise}.profile.json`
>
> 复现命令见 Chamleon `docs/optimizer/pi05/trt_tvm_profile.md`、`scripts/profile_pi05_trt_tvm.sh`。

---

## 1. 实测对照（权威数字）

形状：`prefix_len=968`，`action_horizon=10`，`num_steps=10`，TVM `float16`。

### 1.0 Phase A nsys 结论（2026-07-12）

报告：`Chamleon/output/pi05_libero_profile/nsys/tvm_bench_kv_fp16_stats.txt`  
（nsys 下墙钟略高：prefill≈205 / step≈23，属 profiling 扰动；符号下载 Ctrl-C 不影响 `.nsys-rep`。）

**GPU kernel（cuda_gpu_kern_sum）**：

| 排名 | Kernel | 占比 | 平均 | 含义 |
|------|--------|------|------|------|
| 1 | `matmul10_kernel` | **27.2%** | ~3.65 ms | dlight 大 GEMM（FFN） |
| 2 | `transpose13_kernel` | **21.8%** | ~2.92 ms | **布局转置，几乎赶上 matmul** |
| 3 | `fused_matmul11_add11` | **13.8%** | ~1.97 ms | matmul+bias |
| 4 | `transpose14_kernel` | **9.6%** | ~1.28 ms | 又一次大 transpose |

粗算：**matmul 类 ~50%+，transpose 类 ~35–40%**。TRT MLP 是直接 GEMM，无此「转置税」。

**CUDA API**：`cuLaunchKernelEx` 81.5%（次数多）；`cudaMemcpyAsync` 仅 HtoD ~4.7%（非主因）。

**决策 → 先做 Phase B（cuBLAS + FuseTransposeMatmul）**：默认 CUDA pipeline 只有 `dlight.gpu.Matmul`，没有 cuBLAS；长 matmul/transpose（数 ms）说明是 **GEMM/布局问题**，不是先上 Graph。

B 完成验收：nsys Top 出现 `cublas`/`cutlass` 或 `matmul10`/`transpose13` 大幅下降；`bench_kv` prefill 目标先 ≤120 ms，再冲 G1（≤104）。

### 1.0.1 Phase B 结果（2026-07-12，cuBLAS + FuseTransposeMatmul，已落地）✅

实现：`compile.py:apply_gemm_prepasses()` 在 CUDA legalize 前插入
`partition_for_cublas → RunCodegen → FuseTransposeMatmul`；`bench_kv/compare_kv/sample`
全部支持 `--cublas`。开关一开，**一把过全部 gate 且反超 TRT**：

| 指标 | TRT layer-sum | TVM dlight（B 前） | TVM **cuBLAS**（B 后） | 结果 |
|------|---------------|--------------------|------------------------|------|
| prefill | 86.6 ms | 209 ms | **87.7 ms** | ≈TRT，G1(≤104) ✅ |
| denoise_step_kv | 5.5 ms | 22.0 ms | **4.87 ms** | **优于 TRT**，G2(≤7.2) ✅ |
| prefill+10×step | ~142 ms | 429 ms | **136 ms** | **优于 TRT**，G3(≤170) ✅ |

- prefill **2.4×**、单步 **4.5×**、整环 **3.15×** 提速；单步已低于 TRT。
- 数值：cuBLAS vs dlight `denoise_step_kv` **cosine=0.999943**（max_abs 2.5e-2，fp16 舍入），G5 精度 ✅。
- cuBLAS 的 `matmul_transposed` pattern 直接吃掉显式 transpose，nsys 里的「转置税」（`transpose13/14` 合计 ~31%）随之消除。

命令（复现）：

```bash
cd /home/zhangxa/codes/edgeLLM/Chamleon && source scripts/tvm_env.sh
$MLC_VLA_PY -m mlc_vla.bench_kv  --target cuda --dtype float16 --steps 10 --iters 30          # 基线
$MLC_VLA_PY -m mlc_vla.bench_kv  --target cuda --dtype float16 --steps 10 --iters 30 --cublas # cuBLAS
$MLC_VLA_PY -m mlc_vla.compare_kv --target cuda --dtype float16 --cublas                       # M1≡M0
```

**下一步**：把 `cublas=True` 设为 CUDA 默认（或在 Chamleon worker/`PiZeroRunner` 默认开启），
再验证 Chamleon 端到端（G4）；Phase C（denoise 融合）与量化转为「锦上添花」。

### 1.1 Stage / 孤立算子

| 指标 | TRT | TVM | 倍率 (TVM/TRT) | 来源 |
|------|-----|-----|----------------|------|
| Prefill | **86.6 ms**（layer 合计） / ~85 ms（bench） | **184.5 ms**（`bench_kv`） | **~2.1×** | trt `llm.profile` / `bench_kv` |
| Denoise 单步 | **5.5 ms**（layer） / ~5.1 ms（bench/10） | **20.6 ms** | **~3.8×** | trt `denoise.profile` / `bench_kv` |
| Prefill + 10×step | **~142 ms** | **~391 ms** | **~2.8×** | 上两行推算 |
| e2e（含 Vit/IPC） | ~180 ms | ~477 ms | ~2.7× | Chamleon `bench` |

结论：

1. **`bench_kv` ≈ Chamleon TVM worker 墙钟** → 差距在 TVM 图 / kernel，**不是** Vit、pickle IPC、跨进程。
2. **TRT layer 合计 ≈ Chamleon TRT stage**（86≈85，5.5×10≈55≈51）→ 基线可信。
3. M1 路径正确：`denoise_step_kv` 相对 M0 联合 `denoise_step` 约 **10.6×**（218→20.6 ms/step）。

### 1.2 CUDA Graph（本次负向）

| | 无 `--cuda-graph` | 有 `--cuda-graph` |
|--|-------------------|-------------------|
| prefill | 184.5 ms | 205.0 ms |
| denoise_step_kv | 20.6 ms/step | 23.0 ms/step |

`--cuda-graph` **变慢**。与先前「`tvm_loop` ≈ 逐步总和」一致：当前默认 dlight 路径下，**launch 不是第一瓶颈**，或 Graph 未真正覆盖热点 kernel。下一阶段以 **kernel 质量** 为主，Graph 作为验证项而非主杠杆。

### 1.3 TRT layer 结构（优化靶心）

**LLM prefill（~87 ms）—— FFN 主导（compute-bound）**

| 类别 | 约占 | 说明 |
|------|------|------|
| MLP up+gate | **~53%** | 每层 ~2.5 ms 级大 GEMM |
| MLP down | **~27%** | 每层 ~1.1–1.4 ms |
| MHA (`_gemm_mha_v2`) | ~7% | 每层 ~0.30 ms |
| QKV / O proj | ~7% | |
| Norm / 其它 | <2% | |

→ TVM prefill 要对齐 TRT，**必须先把宽 FFN GEMM 拉到 TRT/cuBLAS 同级**，而不是先抠 attention。

**Denoise 单步（~5.5 ms）—— 短序列 + 小算子（偏 launch / 小 GEMM）**

| 类别 | 约占 | 说明 |
|------|------|------|
| MHA | **~31%** | 每层 ~0.086 ms；suffix 短，访存/启动敏感 |
| 融合 adaRMS dense×36 | **~22%** | 单条 myelin 融合块 ~1.2 ms |
| MLP up+gate / down | ~22% | 比 prefill 轻得多 |
| QKV/O / 其它 | ~25% | RoPE、mask、cast 等 |

→ TVM 单步 20.6 ms ≈ TRT×3.8：更像 **小 shape 上 dlight 调度差 + 融合不足**，叠 10 步放大。

### 1.4 其它现象

- `pi0_model.py` 中 `neg_inf.astype(float16)` 的 overflow **RuntimeWarning**：与延迟无关，可另修（用 fp16 可表示的大负数或保持 mask 在 fp32）。
- 评测精度已近 TRT（cosine ~0.9988）；本文件只谈 **延迟**。

---

## 2. 差距归因（工作假设）

| 段 | TRT 特征 | TVM 现状（推断） | 主杠杆 |
|----|----------|------------------|--------|
| Prefill | Myelin 大 GEMM + 强融合 | dlight 通用 matmul，算术强度未吃满 | **高性能 GEMM BYOC**（cuBLAS/CUTLASS）+ 权重布局 |
| Prefill | FP16/TF32 tactic | fp16 dlight | 对齐累加精度与 tactic |
| Denoise | 短 seq FMHA + 融合 adaRMS | 多 kernel、小 GEMM | **融合**（adaRMS/RoPE/FFN epilogue）+ 小 GEMM/GEMV 专用 schedule |
| Denoise×10 | 单 engine×10 enqueue | 逐步或 loop，Graph 未生效 | 先内核，再确认 `RewriteCUDAGraph` 真捕获 |
| 整机 | Vit 也是 TRT | Vit 仍 TRT 时 e2e 差 ≈ prefill+denoise | 本计划聚焦 **llm/denoise**；Vit 迁 TVM 另项 |

**对齐判据（DoD）**：同机、同 shape、同 dtype：

| Gate | 指标 | 目标 |
|------|------|------|
| G1 | `bench_kv` prefill | ≤ **1.2×** TRT llm layer-sum（≤ ~104 ms） |
| G2 | `bench_kv` denoise_step_kv | ≤ **1.3×** TRT denoise（≤ ~7.2 ms） |
| G3 | prefill+10×step | ≤ **1.2×** TRT 合计（≤ ~170 ms） |
| G4 | Chamleon `bench` e2e（TVM） | ≤ **1.15×** TRT e2e（同 Vit） |
| G5 | 精度 | vs openpi / GT：cosine ≥ 0.998，max_abs 不劣于当前 fp16 |

**超越 TRT**：在 G1–G4 达标后，对 prefill 做 **W4/W8 量化**（权重带宽）或对 denoise 做 **手写/BYOC FMHA**；端侧（Thor）再叠加平台 tactic。

---

## 3. 下一步实现计划

原则：**先量后改**；每步用 `bench_kv` 回归，重大改动跑 `compare_kv` / `compare --mode B --kv`。

### Phase A — 证据补齐（0.5–1 天）

不改模型，确认「算力 vs launch」。

| # | 任务 | 方法 | 期望产出 |
|---|------|------|----------|
| A1 | nsys 包 `bench_kv`（无 CG） | `scripts/profile_pi05_trt_tvm.sh --run nsys` | Top kernels、`cudaLaunchKernel` 占比、GPU gap |
| A2 | 对比有/无 CG 的 nsys | 同上 + `--cuda-graph` | 证明 Graph 是否捕获 denoise 静态区 |
| A3 | 导出 TVM IR / TIR 热点 | `compile` dump 或 `mod.show()`；对 top matmul 看 schedule | 确认是否走 `matmul`/`nt_matmul`/`cublas` |
| A4 | 把本表数字写入 CI/笔记 | 本文件 + Chamleon profile 路径 | 基线冻结 |

**分支决策**：

- 若 A1 显示 **少量长 GEMM 占墙钟** → 走 Phase B（GEMM BYOC）。
- 若 A1 显示 **大量短 kernel + 高 launch** → Phase C 融合/Graph 权重上调。
- 实测倾向：**prefill = B；denoise = B+C 混合**。

### Phase B — Prefill 对齐 TRT（主收益，约 2.1×→≤1.2×）

目标：把 ~80% 的 MLP 时间打到 TRT 同级。

| # | 任务 | 实现入口 | 验收 |
|---|------|----------|------|
| B1 | CUDA 目标默认走 **BLAS/CUTLASS Dispatch** | `compile.py`：在 `relax.build` 前插入与 MLC LLM 类似的 `FuseTransposeMatmul` + `BLASDispatch` / `CutlassDispatch`（按本机 TVM 已启用扩展选） | `bench_kv` prefill 下降；nsys 出现 `cublas`/`cutlass` 名 |
| B2 | 权重布局 NK/KN 与 epilogue 融合 | 对照 TRT `up+gate` 融合；TVM 侧 `FuseOps` / pattern 合并 gate-up | 每层两次大 GEMM → 一次或共享读权重 |
| B3 | Prefill 注意力 | 若 B1 后 MHA 仍慢：接 TVM FMHA / FlashInfer BYOC（`DispatchDualExpertAttention` 的 prefill 半边） | MHA 合计接近 TRT ~6 ms |
| B4 | （可选）W4A16 / W8 量化 prefill | 已有 `quant.py` 预设 `q4f16_1`；`bench_kv --quant` + `compare_quant` | 带宽受限时超越 fp16 TRT；精度过 gate |

**建议顺序**：B1 → 重测 → B2 → B3；量化放在对齐 fp16 之后，作为「超越」手段。

### Phase C — Denoise 单步对齐（~3.8×→≤1.3×）

| # | 任务 | 实现入口 | 验收 |
|---|------|----------|------|
| C1 | **adaRMS / time 调制预计算** | arch 中的 `PrecomputeDenoiseModulation`：`time_emb` 仅依赖步序 → 编译期或 host 预计算表，砍掉每步 time MLP + 部分 dense | 对标 TRT 那条 22% 融合块；单步降数 ms |
| C2 | 小 shape GEMM/GEMV schedule | dlight `GEMV` / `LowBatchGEMM` 规则；或 CUTLASS 小 tile | expert FFN/QKV 接近 TRT ~0.05 ms 级 |
| C3 | Suffix-only attention 融合 | 固定 `Tq=action_horizon`、`Tk=prefix+suffix` 的自定义 / FMHA BYOC；减少 RoPE+mask+SDPA 碎 kernel | nsys 中 attn 相关 launch 数明显下降 |
| C4 | Epilogue 融合 | RMSNorm + residual + SiLU 链 FuseTIR | denoise layer 数接近 TRT 量级（TRT ~280 已含大量融合名） |
| C5 | 验证 `RewriteCUDAGraph` | 确认 `PassContext` 生效；nsys 见 `cudaGraphLaunch`；捕获失败则修静态区（去掉每步 host tensor 重建） | Graph 后 **≤** 无 Graph（至少不倒退） |

### Phase D — 整环与产品路径（对齐 Chamleon e2e）

| # | 任务 | 说明 |
|---|------|------|
| D1 | 默认 `denoise_loop_kv` + 生效的 CUDA Graph | Chamleon `tvm_loop` / `tvm_cuda_graph`；worker 与 `PiZeroRunner.sample_graph` 一致 |
| D2 | Engine 落盘 | 避免每次 worker JIT；冷启动与稳态分离 |
| D3 | 端到端 `chameleon bench` | G4；Vit 仍 TRT 时只比 llm+denoise 段 |
| D4 | （可选）Vit 迁 TVM | 去掉 TRT 依赖；不阻塞 B/C |

### Phase E — 超越 TRT（可选）

在 G1–G4 满足后：

1. **Prefill 权重量化**（int4/int8）+ 激活 fp16：带宽换算力，Ada/Thor 上常可低于 fp16 TRT。
2. **Denoise 手写 runtime / 单 Graph 手写 kernel 序列**（arch §8 诚实取舍：编译器拿 ~80%，余量 BYOC）。
3. **MetaSchedule / 自调优** 对残留 TIR 热点调 tile。
4. 精度允许时 **FP8 KV / 激活**（需单独数值 gate）。

---

## 4. 工作分解（建议排期）

```text
Week 1:  A1–A4 + B1（BLAS/CUTLASS）→ 期望 prefill 184→<120
Week 2:  B2–B3 + C1–C2           → 期望 prefill≤100，step 20→<12
Week 3:  C3–C5 + D1–D3            → G1–G4；写回归数字进本文件附录
之后:    E 量化 / 手写 BYOC        → 挑战 TRT 以下
```

每里程碑更新下表（复制一行填实测）：

| 日期 | prefill | step | ×10 合计 | 备注 |
|------|---------|------|----------|------|
| 2026-07-12 | 184.5 | 20.6 | 391 | 基线 fp16，无有效 CG |
| 2026-07-12 | 209.0 | 22.0 | 429 | dlight 基线（本轮同机复测，无 CG） |
| 2026-07-12 | **87.7** | **4.87** | **136** | **Phase B：cuBLAS+FuseTransposeMatmul，过 G1/G2/G3，反超 TRT** |

---

## 4.1 操作命令手册（按顺序复制执行）

以下默认在开发机；路径按本仓布局。Thor 上改 `TVM_HOME` / `MLC_VLA_PY` 即可。

### 0. 环境（每次新开 shell 先跑）

```bash
cd /home/zhangxa/codes/edgeLLM/Chamleon
source scripts/tvm_env.sh

export CHAM_PY=/home/zhangxa/codes/edgeLLM/Chamleon/models/openpi/.venv/bin/python
export OUT=/home/zhangxa/codes/edgeLLM/Chamleon/output/pi05_libero_profile
mkdir -p "$OUT"/{nsys,bench_kv,trt}

# 自检
$MLC_VLA_PY -c "import tvm_ffi, tvm, mlc_vla; print('ok', tvm.__file__)"
which nsys
```

---

### A. 证据补齐（今天就能跑完）

#### A1. 再确认墙钟基线（可选，你已跑过）

```bash
cd /home/zhangxa/codes/edgeLLM/Chamleon
bash scripts/profile_pi05_trt_tvm.sh --run kv
# 产出：
#   $OUT/bench_kv/fp16_steps.txt
#   $OUT/bench_kv/fp16_steps_cg.txt
```

或手工：

```bash
$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 \
  | tee "$OUT/bench_kv/fp16_steps.txt"

$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 --cuda-graph \
  | tee "$OUT/bench_kv/fp16_steps_cg.txt"
```

#### A2. nsys 采 TVM（无 Graph）— **下一步优先跑这个**

```bash
cd /home/zhangxa/codes/edgeLLM/Chamleon
source scripts/tvm_env.sh
OUT=${OUT:-output/pi05_libero_profile}
mkdir -p "$OUT/nsys"

nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  --force-overwrite=true \
  -o "$OUT/nsys/tvm_bench_kv_fp16" \
  -- "$MLC_VLA_PY" -m mlc_vla.bench_kv \
       --target cuda --dtype float16 --steps 10 --iters 20

nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" \
  | tee "$OUT/nsys/tvm_bench_kv_fp16_stats.txt"

# 看 GPU 空泡（launch 饿 GPU 时有用）
nsys analyze -r gpu_gaps,gpu_time_util \
  "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" \
  | tee "$OUT/nsys/tvm_bench_kv_fp16_analyze.txt"
```

一键等价：

```bash
bash scripts/profile_pi05_trt_tvm.sh --run nsys
```

#### A3. nsys 采 TVM（有 Graph）— 对照 A2

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  --force-overwrite=true \
  -o "$OUT/nsys/tvm_bench_kv_fp16_cg" \
  -- "$MLC_VLA_PY" -m mlc_vla.bench_kv \
       --target cuda --dtype float16 --steps 10 --iters 20 --cuda-graph

nsys stats -r cuda_gpu_kern_sum,cuda_api_sum \
  "$OUT/nsys/tvm_bench_kv_fp16_cg.nsys-rep" \
  | tee "$OUT/nsys/tvm_bench_kv_fp16_cg_stats.txt"

# 若几乎没有 cudaGraphLaunch，说明 RewriteCUDAGraph 未生效
grep -E "cudaGraph|GraphLaunch|tvmgen" "$OUT/nsys/tvm_bench_kv_fp16_cg_stats.txt" || true
```

#### A4. 汇总已有 TRT layer JSON（不必重跑 trt-profile）

```bash
python3 << 'PY'
import json
from pathlib import Path
from collections import defaultdict

def sum_profile(path):
    data = json.loads(Path(path).read_text())
    layers = [x for x in data if "name" in x]
    total = sum(x["averageMs"] for x in layers)
    cats = defaultdict(float)
    for x in layers:
        n, a = x["name"], x["averageMs"]
        if "_gemm_mha" in n: cats["MHA"] += a
        elif "mlp/up" in n or "mlp/gate" in n: cats["MLP_up_gate"] += a
        elif "mlp/down" in n: cats["MLP_down"] += a
        elif "q_proj" in n or "k_proj" in n or "v_proj" in n: cats["QKV"] += a
        elif "o_proj" in n: cats["O"] += a
        else: cats["other"] += a
    print(path)
    print(f"  sum(averageMs) = {total:.3f} ms")
    for k,v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v:8.3f} ms ({100*v/total:5.1f}%)")
    print("  top5:")
    for x in sorted(layers, key=lambda z: -z["averageMs"])[:5]:
        print(f"    {x['averageMs']:7.3f}  {x['name'][:88]}")

base = Path("/home/zhangxa/codes/edgeLLM/Chamleon/output/pi05_libero_trt/profiles")
sum_profile(base / "llm.profile.json")
sum_profile(base / "denoise.profile.json")
PY
```

若要重跑 trt-profile（llm 约几分钟，denoise 更久，别 Ctrl-C）：

```bash
bash scripts/profile_pi05_trt_tvm.sh --run trt
# 或
$CHAM_PY -m chameleon.cli trt-profile \
  --config configs/pi05/pi05_libero_trt_profile.yaml -v
```

#### A5. 读 nsys 时看什么

```bash
# Top GPU kernel（算力热点）
nsys stats -r cuda_gpu_kern_sum "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" | head -40

# Top CUDA API（launch / sync 开销）
nsys stats -r cuda_api_sum "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" | head -40
```

判读：

- Top 是长名 `matmul` / `nt_matmul` / `gemm` → **Phase B（换 BLAS/CUTLASS）**
- Top 是大量短 kernel + `cudaLaunchKernel` 很高 → **Phase C（融合/Graph）**
- CG 报告里几乎无 `cudaGraphLaunch` → Graph 没挂上，先修捕获再谈 Graph

GUI（可选）：

```bash
nsys-ui "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep"
```

---

### B/C 改代码后的回归命令（实现时每次都跑）

```bash
# 1) 延迟
$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 \
  | tee "$OUT/bench_kv/after_change.txt"

# 2) 数值：M1 == M0
$MLC_VLA_PY -m mlc_vla.compare_kv --target cuda --dtype float16

# 3) 相对 openpi（有权重时）
$MLC_VLA_PY -m mlc_vla.compare --mode B --kv --target cuda --dtype float16

# 4) Chamleon 端到端 stage（改完较大补丁后）
cd /home/zhangxa/codes/edgeLLM/Chamleon
source scripts/tvm_env.sh
$CHAM_PY -m chameleon.cli bench --config configs/pi05/pi05_libero_bench_steps.yaml -v \
  | tee "$OUT/bench_kv/cham_steps_after.txt"
```

量化试探（Phase E，对齐 fp16 之后再做）：

```bash
$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --quant q4f16_1 --steps 10 --iters 30
$MLC_VLA_PY -m mlc_vla.compare_quant --target cuda --quant q4f16_1
```

---

### 今天推荐最小路径（3 条命令）

```bash
cd /home/zhangxa/codes/edgeLLM/Chamleon && source scripts/tvm_env.sh
bash scripts/profile_pi05_trt_tvm.sh --run nsys
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum output/pi05_libero_profile/nsys/tvm_bench_kv_fp16.nsys-rep | head -50
```

把 `cuda_gpu_kern_sum` 前 20 行贴回来，即可定 Phase B 还是 C 先动手。

---

## 5. 风险与非目标

| 项 | 说明 |
|----|------|
| 只优化 Graph | 本次已证明无效/负收益；禁止作为唯一手段 |
| bf16 | 精度与部分 GPU 上更慢；默认继续 **fp16** |
| 与 TRT 逐 kernel 同构 | 不必要；对齐 **墙钟与类别耗时** 即可 |
| 改 openpi 训练图 | 不在范围；只改 mlc-vla 编译/运行时 |
| denoise 量化 | launch-bound 段量化收益差；优先融合与小 GEMM |

---

## 6. 相关入口

| 路径 | 作用 |
|------|------|
| `python/mlc_vla/bench_kv.py` | 孤立 prefill/step 测速 |
| `python/mlc_vla/compile.py` | pipeline / `cuda_graph` 开关 |
| `python/mlc_vla/quant.py` | group 量化预设 |
| `python/mlc_vla/sample.py` | `PiZeroRunner` 生产路径 |
| `docs/arch.md` §6–8 | DualExpert / DenoiseLoop / BYOC 设计 |
| `docs/M1.md` | 精度与 Chameleon 集成 |
| Chamleon `docs/optimizer/pi05/trt_tvm_profile.md` | nsys / trt-profile 命令 |
| Chamleon `output/pi05_libero_trt/profiles/*.profile.json` | 本机 TRT layer 基线 |

---

## 7. 一句话策略

> **Prefill 用工业级 GEMM（对齐 TRT 那 80% MLP）；Denoise 用融合 + 小算子 schedule（砍 3.8×）；Graph/量化是锦上添花。先让 `bench_kv` 过 G1/G2，再谈整机超越。**
