"""plannn3 数值对拍：TVM 解码 vs PyTorch 参考（M3）。

验证 **TVM 固定-shape KV 解码核** 与 **PyTorch 参考主干** 在同权重、同 ``token_embeds`` 下
产出 **bit-exact 相同的 ``traj_ids``**（离散输出，对齐 ``resource/plannn3/infer.py`` 的贪心 argmax）。

参考主干是 ``network.py`` GPT 的自包含最小复刻（interleaved RoPE / LayerNorm-no-bias /
GELU-erf / causal），解码走 ``Net.decode`` 式「每步重算全序列」；TVM 走 KV-cache 固定 shape，
二者数值等价。**不依赖 NFS 权重 / 编码器**，CPU 即可跑（缺 torch/tvm 时优雅 skip）。

用法::

    python -m mlc_vla.compare_p3 --dummy --target c
    python -m mlc_vla.compare_p3 --dummy --target c --graph   # 同时校验图内 decode_loop_kv
"""

from __future__ import annotations

import argparse

import numpy as np

from mlc_vla.model.plannn3 import Plannn3Config


# --------------------------------------------------------------------------- #
# PyTorch 参考主干（network.py 的最小自包含复刻）
# --------------------------------------------------------------------------- #
def _build_torch_ref(cfg: Plannn3Config):
    import torch
    from torch import nn

    theta = cfg.rope_theta
    hd = cfg.head_dim

    def rope_tables(seq, offset=0):
        half = hd // 2
        pos = torch.arange(offset, offset + seq, dtype=torch.float64)
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        fr = torch.outer(pos, inv)
        return fr.cos()[None, None].float(), fr.sin()[None, None].float()

    def apply_rope(x, cos, sin):  # x [B,nh,T,hd]
        x1, x2 = x[..., ::2], x[..., 1::2]
        cf, sf = cos.to(x.dtype), sin.to(x.dtype)
        y1 = x1 * cf - x2 * sf
        y2 = x1 * sf + x2 * cf
        return torch.stack((y1, y2), dim=-1).flatten(-2)

    class LN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return nn.functional.layer_norm(x, self.weight.shape, self.weight, None, cfg.layer_norm_eps)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
            self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.c_attn(x).split(cfg.n_embd, dim=2)
            q = q.view(b, t, cfg.n_head, hd).transpose(1, 2)
            k = k.view(b, t, cfg.n_head, hd).transpose(1, 2)
            v = v.view(b, t, cfg.n_head, hd).transpose(1, 2)
            cos, sin = rope_tables(t)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
            return self.c_proj(y)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

        def forward(self, x):
            return self.c_proj(nn.functional.gelu(self.c_fc(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1, self.attn, self.ln_2, self.mlp = LN(cfg.n_embd), Attn(), LN(cfg.n_embd), MLP()

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))

    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([Block() for _ in range(cfg.n_layer)])
            self.ln = LN(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        def backbone(self, x):
            for blk in self.h:
                x = blk(x)
            return x

        def logits_last(self, x):
            return self.head(self.ln(self.backbone(x)[:, -1:])).float()

        @torch.no_grad()
        def generate(self, token_embeds):  # 复刻 infer.py：prefill + 每步重算全序列
            logits = self.logits_last(token_embeds)
            ids = [int(torch.argmax(logits[0, -1]))]
            for _ in range(cfg.pred_times - 1):
                dec = self.embed(torch.tensor([ids], dtype=torch.long))  # [1,k,C]
                x = torch.cat([token_embeds, dec], dim=1)
                logits = self.logits_last(x)
                ids.append(int(torch.argmax(logits[0, -1])))
            return ids

    torch.manual_seed(0)
    return Ref().eval().double()


def _ref_state_to_src(ref) -> dict:
    """把参考模型的 state_dict 转成 loader 认识的源键（transfomer.h.* / traj_head.* / traj_encoder.embed_tokens）。"""
    sd = ref.state_dict()
    src = {}
    for k, v in sd.items():
        arr = v.detach().cpu().numpy()
        if k.startswith("h."):
            src["transfomer." + k] = arr
        elif k == "ln.weight":
            src["traj_head.ln.weight"] = arr
        elif k == "head.weight":
            src["traj_head.head.weight"] = arr
        elif k == "embed.weight":
            src["traj_encoder.embed_tokens.weight"] = arr
    return src


def compare(cfg: Plannn3Config, target: str = "c", check_graph: bool = False):
    import torch

    from mlc_vla.model.plannn3 import load_params, to_tvm_params
    from mlc_vla.plannn3_runner import Plannn3Runner

    ref = _build_torch_ref(cfg)
    src = _ref_state_to_src(ref)

    fns = ["embed_token", "prefill", "decode_step"] + (["decode_loop_kv"] if check_graph else [])
    runner = Plannn3Runner(cfg, target, functions=fns)
    params = load_params(cfg, src, named_params=runner.named_params, dtype=cfg.dtype)
    runner.set_params(to_tvm_params(runner.named_params, params, runner.dev))

    rng = np.random.RandomState(1234)
    token_embeds = rng.randn(1, cfg.prompt_len, cfg.n_embd).astype(cfg.dtype)

    ref_ids = ref.generate(torch.from_numpy(token_embeds.astype("float64")))
    tvm_ids = runner.generate(token_embeds)
    ok = list(ref_ids) == list(map(int, tvm_ids))
    print(f"[compare] host-step: ref={ref_ids}")
    print(f"[compare] host-step: tvm={list(map(int, tvm_ids))}")
    print(f"[compare] traj_ids bit-exact (ref vs TVM host-step): {ok}")

    if check_graph:
        graph_ids = runner.generate_graph(token_embeds)
        okg = list(ref_ids) == list(map(int, graph_ids))
        print(f"[compare] graph : tvm={list(map(int, graph_ids))}")
        print(f"[compare] traj_ids bit-exact (ref vs TVM decode_loop_kv): {okg}")
        ok = ok and okg

    print("PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description="plannn3 TVM vs PyTorch 参考 bit-exact 对拍 (M3)")
    ap.add_argument("--target", default="c")
    ap.add_argument("--dummy", action="store_true", help="用 dummy 小尺寸（秒级）")
    ap.add_argument("--graph", action="store_true", help="同时校验图内 decode_loop_kv")
    args = ap.parse_args()
    cfg = Plannn3Config.dummy() if args.dummy else Plannn3Config()
    ok = compare(cfg, args.target, check_graph=args.graph)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
