"""Minimal runnable inference example for the plannn3 benchmark model.

This is the reference delivery that algorithm teams should mirror: a flat, inference-only
package plus a script that loads the model, runs the real inference path, and validates
its outputs against the golden reference data.

The model runs in three stages (no simple ``model(*inputs)`` call):

    1. ``trace_encoder_forward(*6 inputs)`` -> token embeddings + ``next_hist_img_feat``
    2. ``prefill(...)``                     -> first-step logits
    3. 18x ``decode(...)``                  -> autoregressive trajectory token loop

Run from the repository root (the script's own directory is placed on ``sys.path`` by
the interpreter, so ``import model.*`` resolves without any path manipulation)::

    python3 resource/plannn3/infer.py

Weights and golden pickles live on NFS (paths in ``data.json``); nothing large is
committed to the repository.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from model.common import NetConfig
from model.network import Net
from model.weights import load_weight

from lasermodel_exporter.loaders import load_golden_data

HERE = Path(__file__).resolve().parent
PRED_TIMES = 18  # 3 meta-action tokens + 15 PCA trajectory tokens
SEED = 1024

# io_spec.json is the single source of truth for the ordered input/output names. The
# golden inputs are an ordered list (input_0..input_N) matching these names, which in
# turn matches Net.trace_encoder_forward's signature.
IO_SPEC = json.loads((HERE / "io_spec.json").read_text())
INPUT_NAMES = [entry["name"] for entry in IO_SPEC["inputs"]]


def set_determinism() -> None:
    """Match the reference environment (seeds + TF32) so golden outputs reproduce."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_model(config_path: Path, weights_path: str) -> Net:
    """Construct the model from the local config and restore NFS weights."""
    config_dict = json.loads(config_path.read_text())
    config = NetConfig(**config_dict)
    model = Net(config)
    load_weight(model, weights_path)
    model.cuda().eval()
    return model


@torch.no_grad()
def run_inference(model: Net, inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run the encoder -> prefill -> decode autoregressive loop."""
    token_embeds, position_embeds, next_hist_img_feat = model.trace_encoder_forward(*inputs)

    actions_list: list[torch.Tensor] = []
    actions = None
    for step in range(PRED_TIMES):
        if step == 0:
            logits = model.prefill(token_embeds, position_embeds)
        else:
            index = torch.arange(1, step + 1, dtype=torch.long, device=actions.device).unsqueeze(0)
            actions_with_index = torch.stack([actions, index], dim=2)  # B, L, 2
            logits = model.decode(token_embeds, position_embeds, actions_with_index)
        label_next = torch.argmax(logits, dim=-1)  # B, 1
        actions_list.append(label_next)
        actions = torch.cat(actions_list, dim=1)  # B, L

    return {
        "next_hist_img_feat": next_hist_img_feat,
        "traj_ids": torch.cat(actions_list, dim=1),
    }


# traj_ids are discrete decision tokens and must match the golden reference bit-for-bit.
# next_hist_img_feat is a continuous feature cache produced by the conv/attention-heavy
# dinov3 backbone; with TF32 matmuls enabled (as in the reference) its relative precision
# is ~1e-3, so an exact float match across torch versions/GPUs is not expected. We require
# a small relative error instead, and report the exact figures for transparency.
FEATURE_REL_TOL = 1e-2


def compare_exact(name: str, predicted: torch.Tensor, golden: torch.Tensor) -> bool:
    """Require a bit-exact match (used for discrete token outputs)."""
    pred = predicted.detach().cpu().numpy()
    ref = golden.detach().cpu().numpy()
    if pred.shape != ref.shape:
        print(f"  [FAIL] {name}: shape mismatch predicted {pred.shape} vs golden {ref.shape}")
        return False
    ok = bool(np.array_equal(pred, ref))
    max_abs = float(np.max(np.abs(pred.astype(np.float64) - ref.astype(np.float64)))) if pred.size else 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: shape {pred.shape}, exact match, max abs diff {max_abs:.3e}")
    return ok


def compare_relative(name: str, predicted: torch.Tensor, golden: torch.Tensor, rel_tol: float) -> bool:
    """Require a small max relative error (used for continuous feature outputs)."""
    pred = predicted.detach().cpu().numpy().astype(np.float64)
    ref = golden.detach().cpu().numpy().astype(np.float64)
    if pred.shape != ref.shape:
        print(f"  [FAIL] {name}: shape mismatch predicted {pred.shape} vs golden {ref.shape}")
        return False
    max_abs = float(np.max(np.abs(pred - ref))) if pred.size else 0.0
    magnitude = float(np.max(np.abs(ref))) if pred.size else 0.0
    max_rel = max_abs / magnitude if magnitude > 0 else max_abs
    ok = max_rel <= rel_tol
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}: shape {pred.shape}, "
        f"max abs diff {max_abs:.3e}, golden max |x| {magnitude:.3e}, "
        f"max rel err {max_rel:.3e} (tol {rel_tol:.0e})"
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example but is not available.")

    torch.set_num_threads(8)
    set_determinism()

    data_cfg = json.loads((HERE / "data.json").read_text())

    print("Loading model ...")
    model = build_model(HERE / "model_config.json", data_cfg["weights_path"])
    print("Model loaded.")

    print("Loading golden data ...")
    golden = load_golden_data(HERE / "data.json", device="cuda")
    input_keys = golden.inputs.keys()
    if len(input_keys) != len(INPUT_NAMES):
        raise ValueError(f"Expected {len(INPUT_NAMES)} inputs, got {len(input_keys)}: {input_keys}")
    inputs = [golden.inputs[f"input_{i}"] for i in range(len(INPUT_NAMES))]

    print("Inputs:")
    for name, tensor in zip(INPUT_NAMES, inputs):
        print(f"  {name}: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")

    print("Running inference ...")
    outputs = run_inference(model, inputs)
    print("Outputs:")
    for name, tensor in outputs.items():
        print(f"  {name}: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")

    print("Validating against golden:")
    results = [
        compare_exact("traj_ids", outputs["traj_ids"], golden.outputs["traj_ids"]),
        compare_relative(
            "next_hist_img_feat",
            outputs["next_hist_img_feat"],
            golden.outputs["next_hist_img_feat"],
            rel_tol=FEATURE_REL_TOL,
        ),
    ]

    if all(results):
        print("PASS: all outputs match golden data.")
        return 0
    print("FAIL: one or more outputs do not match golden data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
