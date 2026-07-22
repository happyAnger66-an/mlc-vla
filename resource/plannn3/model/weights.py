"""Minimal checkpoint-loading helper extracted from the algorithm team's training
utilities (``plannn3/utils/train_utils.py``).

Only ``load_weight`` is needed for inference -- the dinov3 encoder uses it to load its
timm backbone checkpoint during construction. The surrounding distributed-training,
optimiser, and checkpoint-saving helpers are intentionally omitted from this minimal
example.
"""

from __future__ import annotations

import glob
import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_weight(model: torch.nn.Module, checkpoint_dir: str | None) -> None:
    """Load weights into ``model`` from a checkpoint directory or a single ``.bin`` file.

    Mirrors the reference loader: tolerant ``module.`` prefix handling, ``gamma`` ->
    ``weight`` renaming, shape-mismatch skipping, and a non-strict ``load_state_dict`` so
    partial backbones load cleanly.

    Args:
        model: Target module to receive the weights.
        checkpoint_dir: Either a directory containing ``pytorch_model-*.bin`` shards (or a
            ``model`` sub-directory of such), or a path to a single ``.bin`` file. ``None``
            leaves the model at its initialised values.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If a directory is given but contains no checkpoint shards.
    """
    if checkpoint_dir is None:
        logger.info("No checkpoint dir provided, leaving model at initialised values.")
        return

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_dir} does not exist.")

    if os.path.isdir(checkpoint_dir):
        resume_ckpt = checkpoint_dir if checkpoint_dir.endswith("model") else os.path.join(checkpoint_dir, "model")
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"Model checkpoint directory {resume_ckpt} does not exist.")
        logger.info(f"load checkpoint from {resume_ckpt}")
        ckpt_bin_list = glob.glob(f"{resume_ckpt}/pytorch_model-*.bin")
        if len(ckpt_bin_list) == 0:
            raise RuntimeError(f"No checkpoint shard found in {resume_ckpt}.")
    else:
        logger.info(f"load checkpoint from {checkpoint_dir}")
        ckpt_bin_list = [checkpoint_dir]

    state_dict: dict[str, torch.Tensor] = {}
    for ckpt_bin_path in ckpt_bin_list:
        state_dict.update(torch.load(ckpt_bin_path, map_location="cpu"))

    model_state_dict = model.state_dict()
    model_has_prefix = list(model_state_dict.keys())[0].startswith("module.")
    ckpt_has_prefix = list(state_dict.keys())[0].startswith("module.")
    if not model_has_prefix and ckpt_has_prefix:
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    elif model_has_prefix and not ckpt_has_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    mismatch_keys = []
    for key in list(state_dict.keys()):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            mismatch_keys.append(key)
            state_dict.pop(key)
        if key not in model_state_dict and key.replace("gamma", "weight") in model_state_dict:
            state_dict[key.replace("gamma", "weight")] = state_dict.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Mismatch keys: {mismatch_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")
