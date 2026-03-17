"""Checkpoint utilities for saving and loading model state."""

import os
import torch
import string
import random
from datetime import datetime
from dataclasses import asdict


def generate_checkpoint_path(config):
    """Generate unique checkpoint path based on config, timestamp, and random ID.

    Creates checkpoint_dir if needed. Path format: {dataset}_{YYMMDD_HHMM}_{random8}.pt
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    run_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    ckpt_name = f"{config.dataset}_{timestamp}_{run_id}".replace("/", "_")
    return os.path.join(config.checkpoint_dir, f"{ckpt_name}.pt")


def save_checkpoint(path, model, optimizer, epoch, best_val, config):
    """Save model, optimizer state, epoch, best metric, and config to path."""
    config_dict = {
        k: v if not callable(v) else v.__name__ for k, v in asdict(config).items()
    }
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    """Load checkpoint, handling compiled/non-compiled model mismatch.

    :param path: Path to checkpoint file.
    :param model: Model to load state into.
    :param optimizer: Optimizer to load state into (None to skip).
    :param device: torch device.
    :return: (epoch, best_val) tuple.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt["model"]

    # Handle mismatch between compiled and non-compiled models
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in ckpt_keys)

    if ckpt_has_prefix and not model_has_prefix:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    elif not ckpt_has_prefix and model_has_prefix:
        state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return ckpt["epoch"], ckpt.get("best_val", ckpt.get("best_val_loss"))
