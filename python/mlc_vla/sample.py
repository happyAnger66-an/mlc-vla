"""M1 host 去噪环 driver：prefill 一次 + N 步 suffix-only 去噪（flow-matching Euler 积分）。

对齐 openpi ``Pi0.sample_actions``：
    dt = -1/N;  x = noise;  t = 1.0
    for _ in range(N):
        v = velocity(x, t);  x = x + dt*v;  t += dt
    return x            # 即 t=0 处的动作

用法（宿主侧编排）：
    runner = PiZeroRunner(config, target)          # 编译 prefill + denoise_step_kv
    actions = runner.sample(params, prefix_embs)   # [1, horizon, action_dim]

其中 prefix_embs 由 embed_image/embed_language 在宿主侧拼好（M0 已提供这些子函数）。
"""

from __future__ import annotations

import numpy as np

from mlc_vla.compile import _device_for, compile_model
from mlc_vla.model.pi0 import Pi0Config
from mlc_vla.openpi_ref import sinusoidal_time_emb

_KV_FUNCS = ["prefill", "denoise_step_kv", "denoise_loop_kv"]


def make_time_embs(num_steps: int, ae_width: int) -> list[np.ndarray]:
    """预计算每步的正弦时间嵌入（t = 1, 1+dt, ...），[1, ae_width] fp32。"""
    dt = -1.0 / num_steps
    t = 1.0
    embs = []
    for _ in range(num_steps):
        embs.append(sinusoidal_time_emb(t, ae_width))
        t += dt
    return embs


def make_rope_np(num_positions: int, head_dim: int, theta: float, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """宿主侧 RoPE cos/sin 表 [1, num_positions, 1, head_dim//2] fp32（与 gemma_dual._rope_tables 一致）。"""
    half = head_dim // 2
    positions = np.arange(offset, offset + num_positions, dtype=np.float64)
    freq_exponents = (2.0 / head_dim) * np.arange(half, dtype=np.float64)
    timescale = theta**freq_exponents
    radians = positions[:, None] / timescale[None, :]
    cos = np.cos(radians).reshape(1, num_positions, 1, half).astype("float32")
    sin = np.sin(radians).reshape(1, num_positions, 1, half).astype("float32")
    return cos, sin


def make_prefix_mask_np(prefix_len: int, num_valid: int | None = None,
                        pad_mask: np.ndarray | None = None, neg: float = -1e9) -> np.ndarray:
    """构造 prefix 加性 mask [1,1,1,prefix_len]（0=有效，neg=padding）。

    ``pad_mask``（[prefix_len] 或 [1,prefix_len] 布尔/0-1）优先；否则用 ``num_valid`` 表示前
    ``num_valid`` 个 token 有效（右侧 padding）。二者都不给则全有效。
    """
    valid = np.ones((prefix_len,), dtype=np.float32)
    if pad_mask is not None:
        valid = np.asarray(pad_mask, dtype=np.float32).reshape(-1)[:prefix_len]
    elif num_valid is not None:
        valid = np.zeros((prefix_len,), dtype=np.float32)
        valid[: int(num_valid)] = 1.0
    add = np.where(valid > 0, 0.0, neg).astype("float32")
    return add.reshape(1, 1, 1, prefix_len)


def euler_loop(step_fn, x0: np.ndarray, num_steps: int) -> np.ndarray:
    """给定每步速度回调 step_fn(x_t[np], step_idx)->v_t[np]，跑 Euler 积分。"""
    dt = -1.0 / num_steps
    x = x0.astype(np.float32)
    for i in range(num_steps):
        v = step_fn(x, i).astype(np.float32)
        x = x + dt * v
    return x


class PiZeroRunner:
    """编译 M1 KV 路径并在宿主侧编排去噪环。"""

    def __init__(self, config: Pi0Config, target: str = "cuda"):
        import tvm
        from tvm import relax

        self.config = config
        self.target = target
        self.ex, self.named_params = compile_model(config, target, functions=_KV_FUNCS)
        self.dev = _device_for(target)
        self.vm = relax.VirtualMachine(self.ex, self.dev)
        self._tvm = tvm

    def to_params(self, arrays: list[np.ndarray]):
        """把 numpy 权重列表搬到 device（顺序须与 named_params 对齐）。"""
        return [self._tvm.runtime.tensor(a, self.dev) for a in arrays]

    def _unpack(self, ret):
        if hasattr(ret, "numpy"):
            return [ret]
        return [ret[i] for i in range(len(ret))]

    def _prepare(self, params, prefix_embs: np.ndarray, prefix_pad: np.ndarray | None):
        """prefill 固化 K/V + 构造 prefix pad mask 与 suffix RoPE 表（device 常量）。

        返回 (keys, values, prefix_mask, suffix_cos, suffix_sin, num_valid)。
        """
        cfg = self.config
        tvm = self._tvm
        dt = cfg.dtype

        if prefix_pad is None:
            num_valid = cfg.prefix_len
            prefix_mask_np = make_prefix_mask_np(cfg.prefix_len)
        else:
            num_valid = int(np.asarray(prefix_pad).reshape(-1).sum())
            prefix_mask_np = make_prefix_mask_np(cfg.prefix_len, pad_mask=prefix_pad)
        prefix_mask = tvm.runtime.tensor(prefix_mask_np, self.dev)
        cos_np, sin_np = make_rope_np(cfg.suffix_len, cfg.vlm.head_dim, cfg.rope_theta, offset=num_valid)
        suffix_cos = tvm.runtime.tensor(cos_np, self.dev)
        suffix_sin = tvm.runtime.tensor(sin_np, self.dev)

        prefix_dev = tvm.runtime.tensor(prefix_embs.astype(dt), self.dev)
        kv = self._unpack(self.vm["prefill"](prefix_dev, prefix_mask, params))
        keys, values = kv[0], kv[1]
        return keys, values, prefix_mask, suffix_cos, suffix_sin, num_valid

    def _noise(self, noise, seed):
        cfg = self.config
        if noise is None:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal((1, cfg.action_horizon, cfg.action_dim)).astype(np.float32)
        return noise.astype(np.float32)

    def sample(self, params, prefix_embs: np.ndarray, noise: np.ndarray | None = None,
               num_steps: int = 10, seed: int = 0,
               prefix_pad: np.ndarray | None = None) -> np.ndarray:
        """宿主去噪环（逐步调用 ``denoise_step_kv``，每步 host↔device 往返）。

        ``prefix_pad``：[prefix_len] 布尔/0-1 有效位（openpi prefix_pad_masks）。None=全有效。
        suffix RoPE offset 取有效 prefix 长度（对齐 openpi ``sum(prefix_pad)``）。
        """
        cfg = self.config
        tvm = self._tvm
        dt = cfg.dtype
        keys, values, prefix_mask, suffix_cos, suffix_sin, _ = self._prepare(params, prefix_embs, prefix_pad)

        noise = self._noise(noise, seed)
        time_embs = make_time_embs(num_steps, cfg.action_expert.width)
        te_dev = [tvm.runtime.tensor(te.astype(dt), self.dev) for te in time_embs]

        def step_fn(x_np, i):
            x_dev = tvm.runtime.tensor(x_np.astype(dt), self.dev)
            v = self.vm["denoise_step_kv"](
                keys, values, x_dev, te_dev[i], suffix_cos, suffix_sin, prefix_mask, params
            )
            return (v.numpy() if hasattr(v, "numpy") else v[0].numpy())

        return euler_loop(step_fn, noise, num_steps)

    def sample_graph(self, params, prefix_embs: np.ndarray, noise: np.ndarray | None = None,
                     seed: int = 0, prefix_pad: np.ndarray | None = None) -> np.ndarray:
        """图内整段去噪环（``denoise_loop_kv``，单次调用跑完 N 步）。

        步数固定为 ``config.num_denoise_steps``（编译期已知），一次调用完成全部 Euler 迭代，
        消除每步 host↔device 往返与跨进程 IPC 开销，可整段 CUDA Graph 捕获。
        """
        cfg = self.config
        tvm = self._tvm
        dt = cfg.dtype
        n = cfg.num_denoise_steps
        keys, values, prefix_mask, suffix_cos, suffix_sin, _ = self._prepare(params, prefix_embs, prefix_pad)

        noise = self._noise(noise, seed)
        x0 = tvm.runtime.tensor(noise, self.dev)  # fp32
        # time_embs 堆成 [num_steps, ae_width]，dtype 与模型一致（喂 time MLP）
        time_embs = np.concatenate(make_time_embs(n, cfg.action_expert.width), axis=0).astype(dt)
        te_dev = tvm.runtime.tensor(time_embs, self.dev)

        out = self.vm["denoise_loop_kv"](
            keys, values, x0, te_dev, suffix_cos, suffix_sin, prefix_mask, params
        )
        return out.numpy() if hasattr(out, "numpy") else out[0].numpy()
