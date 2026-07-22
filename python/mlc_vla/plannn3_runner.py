"""plannn3 宿主侧 AR 解码环 driver（M1）。

对齐 `network.planner_generate` 与 mlc-vla `PiZeroRunner`：
    prefill(token_embeds) → 首 logits + 定长 KV buffer
    for step in range(pred_times-1):
        latest_embed = embed_token(argmax(上一步 logits))
        decode_step(latest_embed, step_rope, add_mask, write_onehot, kv) → logits, kv
    返回 pred_times 个离散轨迹 token id（宿主再走 PCA 反解得到 waypoints）。

`valid_kv_len` / 写入位置 / RoPE 位置全部在宿主按步推进（tensor 化交给图），
KV buffer 每步就地更新（图返回新 buffer，宿主持有句柄回传）。
"""

from __future__ import annotations

import numpy as np

from mlc_vla.model.plannn3 import Plannn3Config
from mlc_vla.model.plannn3.plannn3_model import _rope_tables_np


class Plannn3Runner:
    """编译 embed_token/prefill/decode_step 三入口，并在宿主编排 18 步自回归。"""

    def __init__(self, config: Plannn3Config, target: str = "c"):
        import tvm
        from tvm import relax

        from mlc_vla.plannn3_compile import _device_for, compile_model

        self._tvm = tvm
        self.config = config
        self.ex, self.named_params = compile_model(
            config, target, functions=["embed_token", "prefill", "decode_step"]
        )
        self.dev = _device_for(target)
        self.vm = relax.VirtualMachine(self.ex, self.dev)
        self.params = None

    def _t(self, arr):
        return self._tvm.runtime.tensor(arr, self.dev)

    def set_params(self, params):
        """params：与 named_params 同序的 tvm ndarray 列表。"""
        self.params = params

    def random_params(self):
        params = []
        for _name, p in self.named_params:
            shape = [int(s) for s in p.shape]
            if p.dtype.startswith("int"):
                arr = np.zeros(shape, dtype=p.dtype)
            else:
                arr = (0.02 * np.random.randn(*shape)).astype(p.dtype)
            params.append(self._t(arr))
        self.params = params
        return params

    @staticmethod
    def _first(ret):
        return ret if hasattr(ret, "numpy") else ret[0]

    def generate(self, token_embeds: np.ndarray):
        """token_embeds [1,prompt_len,n_embd] → 长度 pred_times 的 traj id 列表。"""
        assert self.params is not None, "先 set_params / random_params"
        cfg = self.config
        max_seq = cfg.max_seq_len

        ret = self.vm["prefill"](self._t(token_embeds.astype(cfg.dtype)), self.params)
        logits, kv = ret[0], ret[1]
        cur = int(np.argmax(logits.numpy()[0, -1]))
        ids = [cur]

        for step in range(cfg.pred_times - 1):
            pos = cfg.prompt_len + step
            emb = self._first(self.vm["embed_token"](self._t(np.array([[cur]], "int32")), self.params))
            cos, sin = _rope_tables_np(1, cfg.head_dim, cfg.rope_theta, offset=pos)
            idx = np.arange(max_seq)
            add = np.where(idx <= pos, 0.0, cfg.attn_neg_inf).astype("float32").reshape(1, 1, 1, max_seq)
            onehot = (idx == pos).astype(cfg.dtype).reshape(1, max_seq, 1)
            ret = self.vm["decode_step"](
                emb, self._t(cos), self._t(sin), self._t(add), self._t(onehot), kv, self.params
            )
            logits, kv = ret[0], ret[1]
            cur = int(np.argmax(logits.numpy()[0, -1]))
            ids.append(cur)
        return ids


def smoke_generate(config: Plannn3Config, target: str = "c"):
    """随机权重跑通完整 prefill + 18 步 AR 环，验证宿主编排自洽。"""
    runner = Plannn3Runner(config, target)
    runner.random_params()
    token_embeds = np.random.randn(1, config.prompt_len, config.n_embd).astype(config.dtype)
    ids = runner.generate(token_embeds)
    print(f"[smoke] AR loop OK, generated {len(ids)} ids (expect {config.pred_times}): {ids}")
    return ids
