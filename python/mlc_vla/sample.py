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

_KV_FUNCS = ["prefill", "denoise_step_kv"]


def make_time_embs(num_steps: int, ae_width: int) -> list[np.ndarray]:
    """预计算每步的正弦时间嵌入（t = 1, 1+dt, ...），[1, ae_width] fp32。"""
    dt = -1.0 / num_steps
    t = 1.0
    embs = []
    for _ in range(num_steps):
        embs.append(sinusoidal_time_emb(t, ae_width))
        t += dt
    return embs


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

    def sample(self, params, prefix_embs: np.ndarray, noise: np.ndarray | None = None,
               num_steps: int = 10, seed: int = 0) -> np.ndarray:
        cfg = self.config
        tvm = self._tvm
        dt = cfg.dtype

        prefix_dev = tvm.runtime.tensor(prefix_embs.astype(dt), self.dev)
        kv = self._unpack(self.vm["prefill"](prefix_dev, params))
        keys, values = kv[0], kv[1]

        if noise is None:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal((1, cfg.action_horizon, cfg.action_dim)).astype(np.float32)
        time_embs = make_time_embs(num_steps, cfg.action_expert.width)
        te_dev = [tvm.runtime.tensor(te.astype(dt), self.dev) for te in time_embs]

        def step_fn(x_np, i):
            x_dev = tvm.runtime.tensor(x_np.astype(dt), self.dev)
            v = self.vm["denoise_step_kv"](keys, values, x_dev, te_dev[i], params)
            return (v.numpy() if hasattr(v, "numpy") else v[0].numpy())

        return euler_loop(step_fn, noise, num_steps)
