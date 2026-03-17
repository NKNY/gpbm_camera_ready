"""Shared utilities for OPE pipeline."""

import dataclasses
import glob as glob_module
import importlib.util
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def expand_user_paths(config):
    """Expand ~ in all string fields of a dataclass that look like filesystem paths.

    Mutates ``config`` in place and returns it for convenience.
    Only expands fields whose names end with common path suffixes.
    """
    _PATH_SUFFIXES = ("_dir", "_path", "_root", "_source", "_from")
    for f in dataclasses.fields(config):
        if not f.name.endswith(_PATH_SUFFIXES):
            continue
        val = getattr(config, f.name)
        if isinstance(val, str) and "~" in val:
            setattr(config, f.name, os.path.expanduser(val))
    return config


def load_config_from_path(path, inject_helpers=True):
    """Load config from a Python file.

    The config file must define a module-level `config` variable.

    If inject_helpers=True, injects `load_training_config(rel_path)` into the module
    namespace, which resolves paths relative to the config root (parent of logging/F/eval).
    """
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)

    if inject_helpers:
        config_dir = Path(path).parent

        def load_training_config(rel_path):
            base = config_dir
            while (
                base.name not in ("logging", "F", "eval", "training")
                and base.parent != base
            ):
                base = base.parent
            abs_path = base.parent / rel_path
            return load_config_from_path(str(abs_path), inject_helpers=False)

        module.load_training_config = load_training_config

    sys.modules["config_module"] = module
    spec.loader.exec_module(module)
    config = module.config
    if dataclasses.is_dataclass(config):
        expand_user_paths(config)
    return config


def get_repeat_dirs(logging_dir, n_repeats=None):
    """Get sorted repeat directories from logging_dir.

    Expects directories named "repeat_X" where X is an integer (e.g., repeat_0, repeat_1, ...).
    Sorting is done numerically by X.
    """
    dirs = sorted(
        [d for d in os.listdir(logging_dir) if d.startswith("repeat_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return dirs[:n_repeats] if n_repeats else dirs


def print_F_matrix(F, clear_lines=0):
    """Print F matrix with color coding.

    :param F: (K, K) tensor or numpy array
    :param clear_lines: if > 0, move cursor up and clear before printing (for overwriting);
        if < 0, print |clear_lines| blank lines before matrix
    :return: number of lines printed
    """
    if clear_lines > 0:
        print(f"\033[{clear_lines}A\033[J", end="")
    elif clear_lines < 0:
        print("\n" * (-clear_lines - 1))
    F_np = F.cpu().numpy() if isinstance(F, torch.Tensor) else F
    for row in F_np:
        cells = [
            f"\033[38;2;{int(255 * v)};0;{int(255 * (1 - v))}m{v:6.2f}\033[0m"
            for v in row
        ]
        print("  ".join(cells))
    return len(F_np)


def get_eval_metrics(config):
    """Ensure the monitored metric is included in evaluation metrics.

    Early stopping monitors a specific metric (e.g., "ndcg@5"). If that metric
    isn't in config.metrics, we add it so it gets computed during evaluation.

    :param config: training config with ``monitor`` (str) and ``metrics`` (list of (name, k) tuples)
    :return: list of (name, k) tuples including the monitored metric
    """
    monitor_in_metrics = config.monitor == "loss" or any(
        (f"{m}@{k}" if k else m) == config.monitor for m, k in config.metrics
    )
    if monitor_in_metrics:
        return config.metrics
    parts = config.monitor.split("@")
    return config.metrics + [(parts[0], int(parts[1]) if len(parts) > 1 else None)]


def load_tensor(path, device, n_workers=16):
    """Load tensor from .pt file or chunked directory.

    Supports:
    - path.pt file -> load directly
    - path.pt doesn't exist but path/ directory with part_*.pt -> load chunks in parallel

    :param path: Path to .pt file (will also check for chunked dir if file missing)
    :param device: torch device
    :param n_workers: threads for parallel chunk loading (EFS benefits from high values)
    """
    if os.path.isfile(path):
        return torch.load(path, weights_only=True).to(device)

    # Check for chunked directory (path without .pt suffix)
    chunk_dir = path[:-3] if path.endswith(".pt") else path
    if os.path.isdir(chunk_dir):
        parts = sorted(
            glob_module.glob(f"{chunk_dir}/part_*.pt"),
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
        )
        if parts:
            # Load to CPU first, concat, then move to device (avoids GPU OOM)
            if n_workers > 1 and len(parts) > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    tensors = list(ex.map(torch.load, parts))
            else:
                tensors = [torch.load(p, weights_only=True) for p in parts]
            result = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
            return result.to(device)

    raise FileNotFoundError(f"No file or chunked dir found: {path}")


def load_and_concat(paths, device, dim=0, n_workers=16):
    """Load tensors from multiple paths and concatenate.

    Each path can be a .pt file or chunked directory (see load_tensor).
    """
    tensors = [load_tensor(p, device, n_workers) for p in paths]
    return torch.cat(tensors, dim=dim) if len(tensors) > 1 else tensors[0]


def get_device(config_device=None):
    """Get torch device, preferring CUDA if available.

    :param config_device: Requested device string (e.g., "cuda", "cpu", "cuda:0", or None)
    :return: torch.device - Uses CUDA if available and requested (or None), else CPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if config_device is None:
        return torch.device("cuda")
    return torch.device(config_device)
