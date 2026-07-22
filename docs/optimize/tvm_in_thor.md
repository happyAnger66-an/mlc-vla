# TVM on Jetson Thor（Blackwell sm_110a）：TRT vs TVM 差距与优化

> 姊妹篇见 `tvm_vs_trt.md`（Ada/RTX4070）。Thor 上 TVM 相对 TRT 差距**远大于** Ada，
> 根因不同：Ada 是 GEMM/转置税（已由 cuBLAS 解决）；Thor 上 **cuBLAS GEMM 已接上 Blackwell**，
> 剩下的差距在 **dlight 调度的非 GEMM 小算子 + fp32 SIMT attention**（未做 Blackwell 专调/融合），
> 叠加 **仅 20 个 SM** 被放大。

## 0. TL;DR（本轮诊断结论，2026-07-12）

两个实验指向同一件事：

1. **fp16 attention 在 Thor 上大赢**：孤立 `bench_kv --cublas` prefill `92.15 → 79.58 ms`
   （**-13.6%**），正好吃掉那颗占 14.7% 的 `cutlass_80_simt_sgemm`（fp32 QK^T 走 SIMT 而非
   tensor-core）。denoise `12.33 → 12.71 ms/step` 基本不变（suffix 仅 10 query）。
   → Thor 专属高收益（Ada 上只值 2.6%，故 Ada 不默认开）。
2. **target / cuBLAS 都正常**：`arch: sm_110a`、`cc 11.0`、`max_threads 1024`、`warp 32` 全对；
   nsys 里已有 `nvjet_sm110_*`（cuBLAS Blackwell GEMM）。**不是 arch 误判，也不是库不支持 Thor。**
   但 `sm 20`（只有 20 个 SM，Ada 是 46）是关键：dlight「一堆独立小 kernel」在少 SM 上吃亏被放大。

**根因（与 Ada 不同）**：Ada 差距是 GEMM/转置税（cuBLAS 已解决）；**Thor 剩下的差距是
非 GEMM 路径**——dlight 生成的 GELU/RMSNorm/elementwise（通用 heuristic，无 Blackwell 专调）
+ 故意走 fp32 SIMT 的 QK^T——而 TRT 重度融合 + Blackwell 专调。

**动作**：① fp16 attention 已全链路打通并在 Thor bench 配置默认开（待真权重验精度）；
② 下一步啃 GELU/RMSNorm/elementwise 融合与 dlight 热点调优（§4）。

**精度门禁**：fp16 QK 随机权重测过是病态的（cosine 负），必须真权重对拍：给
`pi05_libero_tvm_compare.yaml` 的 `model_overrides` 加 `tvm_attn_logits_dtype: float16` 跑
PT vs TVM，cosine ≥ 0.998 才放心默认开；不过则删掉该行回退 fp32。

## 1. 设备与基线（2026-07-12）

`python -c "import tvm; d=tvm.cuda(0); ..."`：

| 项 | 值 | 说明 |
|----|----|------|
| compute cap | **11.0**（arch `sm_110a`） | TVM 正确识别 Blackwell，arch **无误判** |
| max_num_threads | 1024 | 正常 |
| thread_warp_size | 32 | 正常 |
| max_shared_memory_per_block | 49152（48KB） | 默认档（Blackwell 实际可 opt-in 更大） |
| **multiprocessor_count** | **20** | **SM 很少**（Ada RTX4070 是 46）→ 多小 kernel 吃亏 |

**Chamleon `bench`（`pi05_libero_bench_thor.yaml`，loop+CUDA Graph）**：

| stage | trt_p50 | tvm_p50 | delta |
|-------|---------|---------|-------|
| llm_prefill | 56.23 | 91.52 | **+35.28** |
| denoise_total | 90.72 | 114.61 | +23.89 |
| e2e | 185.83 | 262.15 | +76.32 |

对比 Ada：TVM prefill 在 Thor(91.5) ≈ Ada(92)，**GEMM 以外的部分没吃到 Blackwell 红利**；而 TRT 56<Ada 87。

## 2. 根因：两条后端，不是「CUTLASS 未适配 Thor」

### 2.1 TVM 实际走什么

Phase B 之后，图被拆成两条路径（**不是整图都走 CUTLASS**）：

| 算子 | 后端 | 说明 |
|------|------|------|
| 匹配到的 `matmul` / `matmul_transposed` | **cuBLAS BYOC**（`partition_for_cublas`） | 默认开启；Thor 上出 `nvjet_sm110_*` |
| GELU / RMSNorm / add / cast / split… | **dlight** 生成的 TIR CUDA kernel | 默认 pipeline 的启发式 schedule |
| fp32 QK^T（`attn_logits_dtype=float32`） | 多为 **SIMT sgemm** | 名字可带 `cutlass_*`，但是 `simt` + `80` 风格，**不是** Blackwell tensor-core |

**dlight**（`tvm.dlight`）：给 TIR 打启发式 GPU schedule（tile / thread binding）。Phase B 前大 GEMM 也走它；Phase B 后只剩 **cuBLAS 没吃掉的非 GEMM 算子**。

当前开的是 **cuBLAS Dispatch**，不是完整 CutlassDispatch。nsys 里的 `cutlass_80_simt_sgemm` ≠「CUTLASS 已上 Thor TC」——那是 **fp32 SIMT** 路径。

### 2.2 nsys 证据（`--cublas`）

| 占比 | kernel | 判读 |
|------|--------|------|
| 18.5%+16.3% | `nvjet_sm110_*` | **cuBLAS Blackwell GEMM，正常快**（库/arch 已接上） |
| **16.3%** | `fused_split4_gelu_tanh_multiply6` | **dlight GELU**，~1ms/次（应 ~0.1ms 量级）→ 通用 schedule，无专调 |
| **14.7%** | `cutlass_80_simt_sgemm_128x128` | **fp32 QK^T 走 SIMT**（非 tensor-core）；不是「CUTLASS 不支持 Thor」 |
| ~10% | `add/cast/fused_divide…rms/square_sum` | dlight 小 kernel 链，少 SM 上更亏 |

**结论**：cuBLAS 已适配 Blackwell；瓶颈是 **非 GEMM 的 dlight kernel + fp32 SIMT attention** 未做 Blackwell 专调/融合（且 20 SM 放大多 kernel 开销）。TRT 重度融合，故快。

> 勿写成「dlight/CUTLASS 未适配 Blackwell」——易误解为 GEMM 库没接上。准确说法是：
> **GEMM（cuBLAS）已接上；剩余慢点在 dlight 非 GEMM + fp32 SIMT attention。**

## 3. 已落地：fp16 attention（Thor 专属高收益）✅

- **动机**：那颗 fp32 SIMT sgemm（QK^T）在 Thor 占 **14.7%**（Ada 仅 2.6%，故 Ada 上放弃）。
- **实测**（孤立 `bench_kv --cublas`）：prefill **92.15 → 79.58 ms（-13.6%）**；denoise 基本不变
  （suffix 仅 10 query）。
- **实现**：复用 `Pi0Config.attn_logits_dtype`（见 `tvm_vs_trt.md` §4.2[B-attn]）。贯通到 e2e：
  - `bench_kv --attn-logits-dtype float16`
  - `worker.py --attn-logits-dtype` → `Pi0Config.from_openpi_config(attn_logits_dtype=...)`
  - runner 读 `model_overrides.tvm_attn_logits_dtype` → `TvmWorkerClient(attn_logits_dtype=...)`
  - `pi05_libero_bench_thor.yaml` 默认设 `tvm_attn_logits_dtype: float16`
- **待办（精度门禁）**：fp16 QK 需**真权重** compare 验证再信任（随机权重测试病态，见 §4[B-attn]）：
  ```bash
  # Thor 上，PT vs TVM 真权重对拍，给 TVM 开 fp16 attention
  PYTHONPATH=. <openpi_py311> -m chameleon.cli eval \
    --config configs/pi05/pi05_libero_tvm_compare.yaml   # 该配置加 model_overrides.tvm_attn_logits_dtype: float16
  ```
  cosine ≥ 0.998 才在 Thor 默认开。

## 4. 待做（更大工程，按 ROI）

| # | 项 | 目标 | 难度 |
|---|----|------|------|
| T1 | **GELU 融合 / schedule** | 16.3% 的 `gelu_tanh_multiply` ~1ms→~0.1ms；融进 cuBLAS epilogue 或补更好的 dlight/手写融合 | 中高 |
| T2 | **RMSNorm/elementwise 融合** | `square_sum`+`divide…`+`cast`+`add` 链合并，减 kernel 数（少 SM 上尤其值） | 中 |
| T3 | **denoise 融合** | denoise 12ms/step（Ada 4.8）——多为 dlight 小算子；同 T1/T2 | 中 |
| T4 | dlight / MetaSchedule 调优残留 TIR | 对 GELU/norm 热点按 sm_110a + 20 SM 调 tile/occupancy | 高 |
| T5 | fp8 / nvfp4（Thor 支持） | 对齐 TRT 可能用的低精度 tactic；需数值 gate | 高 |

**一句话**：Thor 上 **cuBLAS GEMM 已到位**；差距在 **dlight 非 GEMM 小算子 + fp32 SIMT attention**（缺融合/专调）。
先吃 fp16 attention（-14% prefill，已落地待验精度），再啃 GELU/norm 融合。

## 5. 复现命令

```bash
# Thor 环境
source scripts/tvm_thor.sh && export MLC_VLA_PY=<thor python3.12> CHAM_PY=<thor openpi py311>

# 孤立 A/B（fp32 vs fp16 attention）
$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --cublas --steps 10 --iters 30
$MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --cublas --steps 10 --iters 30 --attn-logits-dtype float16

# e2e（引擎须先在 Thor build）
bash scripts/profile_pi05_trt_tvm.sh --run thor      # deploy + bench

# nsys：应同时看到 nvjet_sm110_*（cuBLAS 快）与 fused_*gelu* / cutlass_*simt*（慢点）
nsys profile -t cuda -s none -o /tmp/tvm_thor -- \
  $MLC_VLA_PY -m mlc_vla.bench_kv --target cuda --dtype float16 --cublas --steps 10 --iters 20 --attn-logits-dtype float16
nsys stats -r cuda_gpu_kern_sum /tmp/tvm_thor.nsys-rep | head -30
```
