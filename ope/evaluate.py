"""Evaluate OPE estimators on logged ranking data."""

import argparse
import json
import os

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm.auto import tqdm

from ope.gpbm import GPBM
from ope.ensemble import (
    bootstrap_ci_gpu,
    bootstrap_ci_snips_gpu,
    apply_slope,
    compute_opera,
    compute_blue,
)
from ope.ipm import compute_item_position_bias
from ope.utils import (
    get_repeat_dirs,
    load_config_from_path,
    set_seed,
    load_and_concat,
    get_device,
)


@dataclass
class EvalConfig:
    """Configuration for OPE evaluation."""

    # Required paths
    logging_dir: (
        str  # directory with {subset}_propensities.pt and repeat_X/{subset}_rankings.pt
    )
    target_propensities_path: (
        str  # directory with {subset}_propensities.pt (or single .pt file)
    )
    position_bias: List[float]  # assumed position bias for estimators (length K)

    # Optional paths
    true_position_bias: Optional[List[float]] = None  # true bias for reward calculation
    labels_path: Optional[str] = None  # relevance labels for true reward calculation
    F_source: Optional[str] = None  # learned F matrix file or directory
    output_path: str = None  # where to save results JSON
    true_reward: Optional[float] = None  # if provided, skip reward calculation

    # Data selection
    subsets: List[str] = field(
        default_factory=lambda: ["test"]
    )  # subsets to load and concatenate
    n_repeats: Optional[int] = None  # max repeats to evaluate (None = all)
    n_samples_per_query: Optional[int] = (
        None  # max ranking samples per query (None = all)
    )
    seed: Optional[int] = None  # random seed for bootstrap sampling
    batch_size: int = 80000  # ranking samples per batch (for weight computation)

    # Estimators
    window_sizes: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )  # F window sizes (0=IPS)
    use_snips: bool = True  # compute SNIPS variants
    use_single_F: bool = False  # use first F for all repeats

    # SLOPE (baseline)
    use_slope: bool = True  # enable SLOPE selection
    slope_n_bootstrap: int = 500  # bootstrap folds for confidence intervals
    slope_ci_alpha: float = 0.01  # CI significance level (0.01 = 99% CI)

    # BLUE (baseline)
    use_blue: bool = False  # enable BLUE combination
    blue_configs: Optional[Dict[str, List[str]]] = None  # {name: [estimator_names]}

    # OPERA (baseline)
    use_opera: bool = False  # enable OPERA combination
    opera_configs: Optional[Dict[str, List[str]]] = None  # {name: [estimator_names]}
    opera_n_bootstrap: int = 100  # bootstrap folds for MSE matrix estimation
    opera_subsample_exp: float = 0.6  # fold size = n^exp (0.6-0.7 recommended)

    # IPM
    ipm_alpha_bound: float = 1.4  # alpha bound for item-specific position bias

    # Runtime
    prop_clamp_min: float = 1e-10  # clamp logging propensities from below
    max_clip: float = torch.inf  # clip importance weights (for numerical stability)
    device: str = "cuda"  # "cuda" or "cpu"


def _load_propensities(config, device):
    """Load propensities for logging and target policies.

    Supports both single .pt files and chunked directories (part_*.pt).
    """
    P1 = load_and_concat(
        [
            os.path.join(config.logging_dir, f"{s}_propensities.pt")
            for s in config.subsets
        ],
        device,
    )
    P2 = load_and_concat(
        [
            os.path.join(config.target_propensities_path, f"{s}_propensities.pt")
            for s in config.subsets
        ],
        device,
    )
    n_queries, n_docs, K = P1.shape
    P1, P2 = P1.clamp(min=config.prop_clamp_min), P2.clamp(min=0)
    return P1, P2, K, n_queries


def _load_F_matrices(config, device):
    """Load F matrices, returning (F_shared, F_matrices_dict, optim_mse_values).

    If config.F_source is None, returns (None, {}, []) - no learned F, only window-based estimators.

    F files should be named either as a single .pt file or as repeat_{i}.pt files.
    in a directory. Files contain {"F": tensor, "mse": float, ...} dicts.
    Sorting prioritizes numeric suffixes (repeat_0.pt before repeat_10.pt).
    """
    F_matrices, F_shared, optim_mse_values = {}, None, []
    if not config.F_source:
        return F_shared, F_matrices, optim_mse_values

    if config.F_source.endswith(".pt"):
        F_data = torch.load(config.F_source, map_location=device, weights_only=True)
        F_shared = F_data["F"].to(device)
        if "mse" in F_data:
            optim_mse_values.append(F_data["mse"])
    else:
        pt_files = sorted(
            [f for f in os.listdir(config.F_source) if f.endswith(".pt")],
            key=lambda x: int(x.replace(".pt", "").split("_")[-1]),
        )
        if len(pt_files) == 1 or config.use_single_F:
            F_data = torch.load(
                os.path.join(config.F_source, pt_files[0]),
                map_location=device,
                weights_only=True,
            )
            F_shared = F_data["F"].to(device)
            if "mse" in F_data:
                optim_mse_values.append(F_data["mse"])
        else:
            for f in pt_files:
                F_data = torch.load(
                    os.path.join(config.F_source, f),
                    map_location=device,
                    weights_only=True,
                )
                F_matrices[f.replace(".pt", "")] = F_data["F"].to(device)
                if "mse" in F_data:
                    optim_mse_values.append(F_data["mse"])

    return F_shared, F_matrices, optim_mse_values


def _compute_true_reward(config, P2, true_position_bias, device):
    """Compute true reward from labels if not provided.

    True reward = E[sum_k P2[d,k] * pb[k] * rel[d]] = expected clicks under target policy.

    For IPM: pb is item-specific, loaded from {subset}_alphas.pt files.
    For PBM: pb is global (true_position_bias).

    Note: For per-fold evaluation, true_reward should be provided directly.
    """
    if config.true_reward is not None:
        return config.true_reward

    if os.path.isdir(config.labels_path):
        labels = load_and_concat(
            [
                os.path.join(config.labels_path, f"{s}_labels.pt")
                for s in config.subsets
            ],
            device,
        )
    else:
        labels = torch.load(config.labels_path, weights_only=True).to(
            device
        )  # (n_queries, n_docs)
    labels_masked = labels.masked_fill(labels == -1, 0)  # (n_queries, n_docs)

    # Check for IPM (item-specific position bias)
    alpha_path = os.path.join(config.logging_dir, f"{config.subsets[0]}_alphas.pt")
    if os.path.exists(alpha_path):
        # IPM: load item alphas and compute item-specific position bias
        alphas_list = []
        for s in config.subsets:
            alpha_file = os.path.join(config.logging_dir, f"{s}_alphas.pt")
            alphas_list.append(
                torch.load(alpha_file, map_location=device, weights_only=True)
            )
        alphas = torch.cat(alphas_list, dim=0)  # (n_queries, n_docs)
        K = P2.shape[-1]
        item_pb = compute_item_position_bias(
            alphas.cpu().numpy().flatten(),
            K,
            config.ipm_alpha_bound,
            true_position_bias.cpu().numpy(),
        )
        item_pb = torch.tensor(item_pb, dtype=P2.dtype, device=device).reshape_as(P2)
        # P2: (n_queries, n_docs, K), item_pb: (n_queries, n_docs, K), labels_masked: (n_queries, n_docs)
        return (
            (P2 * item_pb * labels_masked.unsqueeze(-1)).sum(dim=(1, 2)).mean().item()
        )
    else:
        # PBM: use global position bias
        # P2: (n_queries, n_docs, K), true_position_bias: (K,), labels_masked: (n_queries, n_docs)
        return (
            (P2 * true_position_bias * labels_masked.unsqueeze(-1))
            .sum(dim=(1, 2))
            .mean()
            .item()
        )


def _compute_summary(results, true_reward, config, optim_mse_values):
    """Compute summary metrics from results."""
    summary = {"config": vars(config), "metrics": {}, "true_reward": true_reward}

    for name, vals in results.items():
        if name.endswith("_weights"):
            summary[name] = vals
            continue
        if vals:
            v = np.array(vals)
            summary["metrics"][name] = {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "mse": float(((v - true_reward) ** 2).mean()),
                "bias": float(v.mean() - true_reward),
            }

    if optim_mse_values:
        summary["metrics"]["optim_MSE"] = float(max(optim_mse_values))

    return summary


def run_evaluation(config: EvalConfig, true_reward: Optional[float] = None):
    """Run OPE evaluation with multiple estimators across logged data repeats.

    Evaluates window-based estimators (w0, w1, ...) and optionally learned F matrix.
    Supports IPS and SNIPS variants, SLOPE selection, BLUE, and OPERA combination.

    :param config: Evaluation configuration.
    :param true_reward: Ground truth reward (computed from labels if not provided).
    :return: Dict with per-estimator results, summary metrics, and true_reward.
    """
    device = get_device(config.device)
    if config.seed is not None:
        set_seed(config.seed)

    P1, P2, K, n_queries = _load_propensities(config, device)
    F_shared, F_matrices, optim_mse_values = _load_F_matrices(config, device)

    # Load position bias
    position_bias = torch.tensor(
        config.position_bias, dtype=torch.float32, device=device
    )

    # true_position_bias only needed if computing true_reward from labels
    if config.true_position_bias is not None:
        true_position_bias = torch.tensor(
            config.true_position_bias, dtype=torch.float32, device=device
        )
    elif true_reward is None and config.true_reward is None:
        raise ValueError(
            "true_position_bias is required when true_reward is not provided"
        )
    else:
        true_position_bias = None

    if true_reward is None:
        true_reward = _compute_true_reward(config, P2, true_position_bias, device)

    repeat_dirs = get_repeat_dirs(config.logging_dir, config.n_repeats)

    print(
        f"K: {K}, n_queries: {n_queries}, repeats: {len(repeat_dirs)}, true_reward: {true_reward:.4f}"
    )

    # Build estimator names
    est_names = [f"w{w}" for w in config.window_sizes]
    if config.F_source:
        est_names.append("learned_F")
    if config.use_snips:
        est_names += [f"{n}_snips" for n in est_names]

    non_learned = [n for n in est_names if "learned_F" not in n]
    results = {n: [] for n in est_names}
    if config.use_slope:
        results["SLOPE"], results["SLOPE_SNIPS"] = [], []

    # Build estimators
    estimators = {}
    for w in config.window_sizes:
        estimators[f"w{w}"] = GPBM(
            K=K, window_size=w, position_bias=position_bias, device=device
        )

    iterator = tqdm(repeat_dirs, desc="Evaluating")
    for repeat_name in iterator:
        F_learned = F_shared if F_shared is not None else F_matrices.get(repeat_name)
        if config.F_source and F_learned is None:
            continue  # skip repeats without F matrix

        repeat_path = os.path.join(config.logging_dir, repeat_name)
        rankings = load_and_concat(
            [os.path.join(repeat_path, f"{s}_rankings.pt") for s in config.subsets],
            device,
            dim=1,
        )
        clicks = load_and_concat(
            [os.path.join(repeat_path, f"{s}_clicks.pt") for s in config.subsets],
            device,
            dim=1,
        )

        n_samples = rankings.shape[0]
        if config.n_samples_per_query:
            n_samples = min(n_samples, config.n_samples_per_query)
            rankings, clicks = rankings[:n_samples], clicks[:n_samples]

        if F_learned is not None:
            estimators["learned_F"] = GPBM(
                K=K, position_bias=position_bias, F=F_learned, device=device
            )

        # IPS: D = (1/n) sum w*r;  SNIPS: D = (sum w*r) / (sum w) * K
        # Store per-sample (wr_sum, w_sum) for bootstrap
        data = {n: {"wr_sum": [], "w_sum": []} for n in est_names}

        for est_name, estimator in estimators.items():
            if est_name == "learned_F" and F_learned is None:
                continue

            rankings_flat = rankings.reshape(-1, K)  # (n_samples * n_queries, K)
            clicks_flat = clicks.reshape(-1, K)  # (n_samples * n_queries, K)
            n_total = rankings_flat.shape[0]
            reward_flat = torch.maximum(
                clicks_flat, torch.zeros(1, device=device)
            )  # (n_samples * n_queries, K)

            for start in range(0, n_total, config.batch_size):
                end = min(start + config.batch_size, n_total)
                chunk_idx = (
                    torch.arange(start, end, device=device) % n_queries
                )  # (batch,)

                batch = {
                    "logging_actions": rankings_flat[start:end],  # (batch, K)
                    "rewards": clicks_flat[start:end],  # (batch, K)
                    "logging_propensities": P1[chunk_idx],  # (batch, n_docs, K)
                    "target_propensities": P2[chunk_idx],  # (batch, n_docs, K)
                }

                weights = estimator.compute_ips_weights_torch(batch)  # (batch, K)
                weights = torch.clip(weights, max=config.max_clip)  # (batch, K)
                wr_sum_batch = (weights * reward_flat[start:end]).sum(
                    dim=-1
                )  # (batch,)
                w_sum_batch = weights.sum(dim=-1)  # (batch,)

                data[est_name]["wr_sum"].append(wr_sum_batch)
                data[est_name]["w_sum"].append(w_sum_batch)

                if config.use_snips:
                    snips_name = f"{est_name}_snips"
                    data[snips_name]["wr_sum"].append(wr_sum_batch)
                    data[snips_name]["w_sum"].append(w_sum_batch)

        # Compute estimates
        repeat_est = {}
        for name in est_names:
            if "learned_F" in name and F_learned is None:
                continue
            wr_sum = torch.cat(data[name]["wr_sum"])
            if "_snips" in name:
                w_sum = torch.cat(data[name]["w_sum"])
                total_w = w_sum.sum()
                # Multiply by K because to calculate per-ranking reward instead of per-position.
                repeat_est[name] = (
                    (wr_sum.sum() / total_w * K).item() if total_w > 0 else 0
                )
            else:
                repeat_est[name] = wr_sum.mean().item()
            results[name].append(repeat_est[name])

        if config.use_slope:
            # Generate shared bootstrap indices once
            n = n_samples * n_queries
            bootstrap_idx = torch.randint(
                0, n, (config.slope_n_bootstrap, n), device=device
            )

            ci_bounds = {}
            for name in non_learned:
                wr_sum = torch.cat(data[name]["wr_sum"])
                w_sum = torch.cat(data[name]["w_sum"])
                if "_snips" in name:
                    ci_bounds[name] = bootstrap_ci_snips_gpu(
                        wr_sum, w_sum, K, bootstrap_idx, config.slope_ci_alpha
                    )
                else:
                    ci_bounds[name] = bootstrap_ci_gpu(
                        wr_sum, bootstrap_idx, config.slope_ci_alpha
                    )

            ordered = [f"w{w}" for w in config.window_sizes]
            slope_estimator_name = apply_slope(
                ordered, {n: ci_bounds[n] for n in ordered}
            )
            results["SLOPE"].append(repeat_est[slope_estimator_name])
            if config.use_snips:
                ordered_snips = [f"w{w}_snips" for w in config.window_sizes]
                slope_snips_estimator_name = apply_slope(
                    ordered_snips, {n: ci_bounds[n] for n in ordered_snips}
                )
                results["SLOPE_SNIPS"].append(repeat_est[slope_snips_estimator_name])
            iterator.set_description_str(
                f"SLOPE: {slope_estimator_name}, SLOPE_SNIPS: {slope_snips_estimator_name}"
            )

        if config.use_blue:
            blue_configs = config.blue_configs or {"BLUE": non_learned}
            for blue_name, blue_est_names in blue_configs.items():
                est, var, weights = compute_blue(blue_est_names, data, K)
                if est is not None:
                    results.setdefault(blue_name, []).append(est)
                    results.setdefault(f"{blue_name}_weights", []).append(weights)

        if config.use_opera:
            opera_configs = config.opera_configs or {"OPERA": non_learned}
            for opera_name, opera_est_names in opera_configs.items():
                opera_est, opera_weights = compute_opera(
                    opera_est_names,
                    data,
                    K,
                    n_bootstrap=config.opera_n_bootstrap,
                    subsample_exp=config.opera_subsample_exp,
                    device=device,
                )
                if opera_est is not None:
                    results.setdefault(opera_name, []).append(opera_est)
                    results.setdefault(f"{opera_name}_weights", []).append(
                        {n: float(w) for n, w in zip(opera_est_names, opera_weights)}
                    )

    # Summary
    summary = _compute_summary(results, true_reward, config, optim_mse_values)

    if config.output_path:
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        with open(config.output_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nTrue reward: {true_reward:.4f}")
    for name, m in summary["metrics"].items():
        if isinstance(m, dict):
            print(
                f"  {name}: mean={m['mean']:.4f}, bias={m['bias']:.4f}, std={m.get('std', 0):.4f}, mse={m.get('mse', 0):.7f}"
            )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--n_repeats", type=int)
    parser.add_argument("--n_samples_per_query", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_snips", type=int)
    parser.add_argument("--use_slope", type=int)
    parser.add_argument("--use_opera", type=int)
    parser.add_argument(
        "--opera_estimators",
        type=str,
        help="Comma-separated list of estimators for OPERA",
    )
    parser.add_argument(
        "--blue_estimators",
        type=str,
        help="Comma-separated list of estimators for BLUE",
    )
    parser.add_argument(
        "--slope_ci_method", type=str, choices=["bootstrap", "bernstein"]
    )
    parser.add_argument("--max_clip", type=float)
    parser.add_argument("--prop_clamp_min", type=float)
    parser.add_argument("--true_reward", type=float)
    parser.add_argument("--F_source", type=str, help="Override F_source path")
    parser.add_argument("--opera_subsample_exp", type=float)
    parser.add_argument(
        "--no_output", action="store_true", help="Disable writing results to file"
    )
    args = parser.parse_args()

    config = load_config_from_path(args.config)
    if args.no_output:
        config.output_path = None
    for k in [
        "n_repeats",
        "n_samples_per_query",
        "device",
        "seed",
        "use_snips",
        "use_slope",
        "use_opera",
        "slope_ci_method",
        "max_clip",
        "prop_clamp_min",
        "true_reward",
        "F_source",
        "opera_subsample_exp",
    ]:
        if getattr(args, k, None) is not None:
            setattr(config, k, getattr(args, k))
    if args.opera_estimators:
        config.opera_estimators = args.opera_estimators.split(",")
    if args.blue_estimators:
        config.blue_estimators = args.blue_estimators.split(",")

    run_evaluation(config, true_reward=args.true_reward)
