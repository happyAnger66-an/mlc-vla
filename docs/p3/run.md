# plannn3 运行指南

本文汇总 plannn3 TVM 推理的运行方式：环境准备、各阶段冒烟测试、数值对拍、端到端调用。
背景与设计见 `arch.md`，已完成范围见 `finished.md`。

## 1. 环境准备

需要一份已编译的开源 TVM（Unity/Relax）。运行任何命令前先设置环境变量：

```bash
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:/home/zhangxa/codes/edgeLLM/mlc-vla/python

# 关键：绕过 tvm-ffi 对 torch C-DLPack 插件的 JIT 编译（否则 import 会卡在 FileLock）
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

> 设了 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 后，`import tvm` 约 1.7s；否则可能长时间卡住。

- **CPU 冒烟**：`--target c`（zero pipeline + 本机 gcc 编译为 `.so`），无需 GPU/权重，秒级。
- **GPU 运行**：`--target cuda`，可加 `--cuda-graph`、`--cublas`/`--no-cublas`（cuBLAS 不可用时自动回退 dlight）。
- **`--dummy`**：用小尺寸配置，加速验证；去掉则用真实 `Plannn3Config`。

## 2. 编译 / 冒烟测试（`mlc_vla.plannn3_compile`）

```bash
cd /home/zhangxa/codes/edgeLLM/mlc-vla/python

# 主干：prefill + decode_step（KV-cache 单步）
python -m mlc_vla.plannn3_compile --dummy --smoke --target c

# 视觉塔：DINOv3 embed_visual
python -m mlc_vla.plannn3_compile --dummy --visual --target c

# 宿主 18 步 AR 环
python -m mlc_vla.plannn3_compile --dummy --generate --target c

# 图内整段解码环 decode_loop_kv（可整段 CUDA Graph）
python -m mlc_vla.plannn3_compile --dummy --loop --target c

# 仅导出 relax IRModule 文本
python -m mlc_vla.plannn3_compile --dummy --dump-ir

# 仅编译（不跑）
python -m mlc_vla.plannn3_compile --dummy --target c
```

GPU + 优化开关示例：

```bash
python -m mlc_vla.plannn3_compile --loop --target cuda --cuda-graph            # cuBLAS 自动探测
python -m mlc_vla.plannn3_compile --loop --target cuda --cuda-graph --no-cublas # 强制走 dlight
```

CLI 参数：`--target`（默认 `c`）、`--smoke`、`--visual`、`--generate`、`--loop`、
`--dump-ir`、`--dummy`、`--cuda-graph`、`--cublas`/`--no-cublas`。

## 3. 数值对拍（`mlc_vla.compare_p3`）

TVM 解码核 vs 自包含 PyTorch 参考主干，校验 `traj_ids` **bit-exact**（无需 GPU/NFS/权重）：

```bash
python -m mlc_vla.compare_p3 --dummy --target c --graph
```

预期输出 `PASS`，逐步环与图内 `decode_loop_kv` 均与参考一致。

## 4. 端到端调用（`Plannn3Runner`）

```python
import numpy as np
from mlc_vla.model.plannn3 import Plannn3Config, load_params, load_state_dict, to_tvm_params
from mlc_vla.plannn3_runner import Plannn3Runner
from mlc_vla.plannn3_decode import PCATrajDecoder

cfg = Plannn3Config()                     # 或 Plannn3Config.dummy() 冒烟
runner = Plannn3Runner(cfg, target="cuda", cuda_graph=True)   # CPU 用 target="c"

# 加载真实权重（GPT 主干 + 轨迹头 + traj 嵌入）
src = load_state_dict("/path/to/plannn3_checkpoint.safetensors")
params = load_params(cfg, src, named_params=runner.named_params, dtype=cfg.dtype)
runner.set_params(to_tvm_params(runner.named_params, params, runner.dev))
# 冒烟时可用 runner.random_params() 代替上面三行

# token_embeds 来自 encode 阶段：
#   - DINOv3 backbone 走 TVM embed_visual（见 §2 --visual）
#   - 多相机/时序/navi/history 外层编排在宿主侧拼接（按 arch.md 设计）
token_embeds = np.random.randn(1, cfg.prompt_len, cfg.n_embd).astype(cfg.dtype)

# 端到端：token_embeds -> traj_ids(TVM) -> waypoints(宿主)
pca = PCATrajDecoder("/path/to/pred_dxdydyaw_pca_tokenizer.json")  # 无则传 None，仅拆分不反解
out = runner.run(token_embeds, use_graph=True, pca=pca)
# out = {traj_ids, meta_action_ids, main_action_ids, [wp_delta, waypoints]}
```

- `use_graph=True` 走图内 `decode_loop_kv`（可 CUDA Graph）；`False` 走宿主逐步环，二者 `traj_ids` 一致。
- 只要离散 `traj_ids`（对齐 golden）时可省略 `pca`；需连续轨迹时传入 PCA tokenizer json。

## 5. 常见问题

- **`import tvm` 卡住**：确认已 `export TVM_FFI_DISABLE_TORCH_C_DLPACK=1`；必要时清理残留卡锁进程。
- **cuBLAS 报错/回退**：本机 TVM 未编入 `relax.ext.cublas` 时，`--cublas` 会告警并回退 dlight，属正常。
- **`--target cuda` 需要 GPU**：无 GPU 时用 `--target c` 做 CPU 冒烟与对拍。
