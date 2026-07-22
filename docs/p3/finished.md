# plannn3 TVM 推理：已完成工作总结

本文件汇总在 mlc-vla 上支持 plannn3 TVM 推理已交付的工作（M0–M2.5），
逐 milestone 详情见同目录 `M0.md` / `M1.md` / `M2.md` / `M3.md`，总体方案见 `arch.md`。

## 关键边界澄清（决定"还差什么"）

复核参考实现（`resource/plannn3/`）后确认两点，据此界定 TVM 侧交付范围：

1. **`TrajTokenizer.ids_to_embed` 就是 `embed_tokens(ids)` 查表**（`trajectory_encoder_v2.py:289`）。
   即 M0 的 `embed` 占位**结构本就正确**，不是替身，只差接入真实权重 `traj_encoder.embed_tokens.weight`。
2. **encode 外层编排按 arch.md 设计属宿主/PVA 预处理，不进 TVM 固定-shape 图**：
   多相机 DINOv3 外层拼接、view_token/PE、时序 `hist_img_feat` 缓存（`visual_encoder.py`）、
   navi 车道位标解码（`navigation_encoder_2.py`）、history 码本最近邻 `cdist.argmin`（`trajectory_encoder.py`）、
   数据相关 crop/resize——这些是数据相关、非定尺寸的宿主逻辑。
   **TVM engine 只负责 DINOv3 backbone（`embed_visual`，M1 已交付）+ 主干 prefill/decode。**

## 已交付内容

### 模型与图（M0 / M1 / M2）

| 模块 | 文件 | 说明 |
|------|------|------|
| GPT 主干 + KV-cache | `model/plannn3/plannn3_model.py: embed_token / prefill / decode_step` | relax nn 重写，interleaved RoPE，定长 KV buffer + `valid_kv_len` + `where` 写入 |
| 视觉 tokenizer | `model/plannn3/dinov3.py` + `embed_visual` | DINOv3 backbone（TVM engine 的视觉阶段） |
| 图内整段解码环 | `model/plannn3/plannn3_model.py: decode_loop_kv` | prefill + 固定 `pred_times-1` 步 AR 下沉进图，逐步 RoPE/mask/onehot 编译期 bake，图内 `topi.argmax` + embed，可整段 CUDA Graph 重放 |
| 编译 pipeline | `plannn3_compile.py` | cuBLAS BYOC + `FuseTransposeMatmul` + CUDA Graph 三态开关；`resolve_cublas` / `apply_gemm_prepasses`；dlight 兜底；`smoke_loop` |
| 主机/图 runner | `plannn3_runner.py: Plannn3Runner` | 宿主逐步环 `generate` 与图内环 `generate_graph` |
| 主干/头 loader | `model/plannn3/plannn3_loader.py` | `transfomer.h.*` / `traj_head.*` → relax nn 参数命名；`.safetensors`/`.bin` 读取 |

### 端到端闭环与对拍（M2.5，即 M3.md）

| 模块 | 文件 | 说明 |
|------|------|------|
| 真实 traj 嵌入权重 | `plannn3_loader.py: build_name_map` | 新增 `embed.weight` ← `traj_encoder.embed_tokens.weight`；源缺失时退回 `allow_missing` |
| 输出反解 | `plannn3_decode.py`（新增） | 纯 numpy 复刻 `PCATokenizer.decode` + `TrajTokenizer.decode`：`traj_ids → 增量轨迹 → 累积重建绝对 (x,y,yaw)` |
| 端到端编排 | `plannn3_runner.py: run / decode_waypoints` | `token_embeds → traj_ids（TVM）→ waypoints（宿主）`，宿主环/图内环可切换 |
| 数值对拍 | `compare_p3.py`（新增） | 自包含 PyTorch 参考主干 vs TVM 解码核，同权重同输入校验 `traj_ids` **bit-exact**（逐步环 + 图内环） |

## 验证结果（CPU `target=c`，无需 GPU/NFS/权重）

```bash
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1     # 关键：绕过 torch C-DLPack JIT，import ~1.7s
python -m mlc_vla.compare_p3 --dummy --target c --graph
```

```
[compare] host-step: ref=[23, 9, 3, 20]
[compare] host-step: tvm=[23, 9, 3, 20]
[compare] traj_ids bit-exact (ref vs TVM host-step): True
[compare] graph : tvm=[23, 9, 3, 20]
[compare] traj_ids bit-exact (ref vs TVM decode_loop_kv): True
PASS
```

- **`traj_ids` bit-exact**：PyTorch 参考主干（`network.py` GPT 自包含复刻：interleaved RoPE / LayerNorm-no-bias /
  GELU-erf / causal，解码走 `Net.decode` 式每步重算全序列）与 TVM 逐步环、TVM 图内 `decode_loop_kv` 三者完全一致。
- **waypoint 重建**：`reconstruct_waypoints` vs torch 端口 max_abs_diff = `2.2e-16`（机器精度）。
- **端到端 `run`**：宿主环与图内环 `traj_ids` 一致；`meta/main` 按 `18 = 3 + 15` 正确拆分。
- **loader**：`embed.weight` 已随参数一并加载生效。

## 遗留（明确非本轮 TVM 交付范围）

- encode 外层编排（多相机/时序/navi/history/crop）在宿主侧实现——按 arch.md 设计，非 TVM engine。
- GPU 侧真机验证（CUDA + CUDA Graph + cuBLAS）与真实 checkpoint 整网 golden 复核——
  需 NVIDIA GPU + NFS 上的 checkpoint / PCA tokenizer json / 码本。对拍脚手架已就位，
  硬件到位即可用同一 `token_embeds` 接 `infer.py` 的 `Net.decode` 做整网复核。
