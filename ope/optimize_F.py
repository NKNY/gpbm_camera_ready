"""F matrix optimization for GPBM off-policy evaluation."""

import argparse
import os

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union
from tqdm.auto import tqdm

from ope.utils import (
    load_config_from_path,
    get_repeat_dirs,
    print_F_matrix,
    set_seed,
    load_and_concat,
    get_device,
)


@dataclass
class FOptimizationConfig:
    """Configuration for F matrix optimization.

    Optimizes F to minimize MSE = Var + Bias^2 using bootstrap estimation.
    """

    # Required paths
    logging_dir: str  # directory with {subset}_propensities.pt and repeat_X subdirs
    target_propensities_path: str  # .pt file or directory with target propensities
    output_dir: str  # where to save optimized F matrices
    position_bias: List[float]  # assumed position bias (length K)

    # Data selection
    subsets: List[str] = field(
        default_factory=lambda: ["test"]
    )  # subsets to load and concatenate
    n_repeats: Optional[int] = None  # max repeats to process (None = all)
    n_samples_per_query: Optional[int] = (
        None  # max ranking samples per query (None = all)
    )
    single_repeat: Union[bool, int] = (
        False  # False=all, True=repeat_0, int=specific index
    )
    seed: Optional[int] = None  # random seed for bootstrap sampling

    # Optimization loop
    max_outer: int = 200  # max optimization iterations
    lr: float = 0.1  # Adam learning rate for F parameters
    patience: int = 50  # iterations without improvement before stopping
    init_F: str = "middle"  # F initialization: "middle" (sigmoid=0.5) or float value

    # Variance estimation (bootstrap subsampling)
    n_bootstrap_var: int = 50  # number of bootstrap folds for variance
    variance_sample_size: Optional[int] = (
        None  # samples per fold (None = use subsample_ratio)
    )
    subsample_ratio: float = (
        0.6  # fold size = n^ratio when variance_sample_size is None
    )

    # Bias estimation (Taylor expansion for position bias uncertainty)
    n_bootstrap_bias: int = 50  # number of bootstrap folds for bias gradient
    bias_sample_size: Optional[int] = (
        None  # samples per fold (None = use bias_subsample_ratio)
    )
    bias_subsample_ratio: float = (
        0.6  # fold size = n^ratio when bias_sample_size is None
    )
    eps_k: Optional[List[float]] = (
        None  # position bias uncertainty bounds (length K, 0 = skip bias)
    )

    # Memory/batching
    chunk_size: int = 10000  # samples per forward pass (reduce if OOM)
    bootstrap_chunk: int = 10  # bootstrap folds computed in parallel
    weight_clamp: float = float(
        "inf"
    )  # clip importance weights (for numerical stability)

    # Runtime
    device: str = "cuda"  # "cuda" or "cpu"
    verbose: bool = True  # show progress bar and F matrix


def optimize_F(P1, P2, p, rankings, clicks, config: FOptimizationConfig, eps_k=None):
    """Optimize F matrix to minimize MSE bound = Var + Bias^2.

    :param P1: (n_queries, n_docs, K) - logging policy propensities
    :param P2: (n_queries, n_docs, K) - target policy propensities
    :param p: (K,) - position bias estimates (assumed known or estimated)
    :param rankings: (n_total, K) - flattened observed rankings where n_total = n_samples * n_queries
    :param clicks: (n_total, K) - flattened observed clicks
    :param config: optimization hyperparameters
    :param eps_k: (K,) - position bias uncertainty for Taylor bias bound (0 = no uncertainty)

    Expects rankings and clicks to be flattened such that query q's samples are at
    indices [q, q + n_queries, q + 2*n_queries, ...]. This is achieved by reshaping
    (n_samples, n_queries, K) -> (n_samples * n_queries, K) with default row-major order.
    """
    n_queries, n_docs, K = P1.shape
    n_total = rankings.shape[0]
    device = P1.device
    dtype = P1.dtype

    if eps_k is None:
        eps_k = torch.zeros(K, device=device, dtype=dtype)

    # Initialize F
    init_val = 0.0 if config.init_F == "middle" else float(config.init_F)
    F_raw = torch.full(
        (K, K), init_val, device=device, dtype=dtype, requires_grad=True
    )  # (K, K)

    query_idx = torch.arange(n_total, device=device) % n_queries  # (n_total,)

    # Precompute propensities at ranking positions for all samples
    rank_exp = rankings.unsqueeze(-1).expand(-1, -1, K)  # (n_total, K, K)
    P1_at_rank = P1[query_idx].gather(1, rank_exp)  # (n_total, K, K)
    P2_at_rank = P2[query_idx].gather(1, rank_exp)  # (n_total, K, K)
    P1p_at_rank = P1_at_rank * p  # (n_total, K, K)
    P2p_at_rank = P2_at_rank * p  # (n_total, K, K)
    del rank_exp  # free memory

    optimizer = torch.optim.Adam([F_raw], lr=config.lr)

    def get_F():
        """Convert raw parameters to valid F matrix via sigmoid + row normalization."""
        F_sigmoid = torch.sigmoid(F_raw)  # (K, K)
        ret = F_sigmoid / F_sigmoid.amax(-1, keepdim=True)  # (K, K)
        return ret

    # W[k] = sum_j F[j,k] * P2[d,j] * pb[j] / sum_k' F[j,k'] * P1[d,k'] * pb[k']
    # Importance weight for doc d at position k under GPBM
    def compute_W_for_idx(F, idx):
        """Compute importance weights for given sample indices.

        :param F: (K, K) - F matrix
        :param idx: (bs,) - sample indices
        :return: W tensor of shape (bs, K) - importance weights
        """
        P1p_sub = P1p_at_rank[idx]  # (bs, K, K)
        P2p_sub = P2p_at_rank[idx]  # (bs, K, K)
        denom = (
            (F.unsqueeze(0).unsqueeze(0) * P1p_sub.unsqueeze(2))
            .sum(dim=-1)
            .clamp(min=1e-20)
        )  # (bs, K, K)
        num = F.T.unsqueeze(0) * P2p_sub  # (bs, K, K)
        W = (num / denom).sum(dim=-1)  # (bs, K)
        return W.clamp(max=config.weight_clamp)

    if config.bias_sample_size is not None:
        bias_sample_size_actual = min(config.bias_sample_size, n_total)
    else:
        bias_sample_size_actual = int(n_total**config.bias_subsample_ratio)
    if config.variance_sample_size is not None:
        n1 = min(config.variance_sample_size, n_total)
    else:
        n1 = int(
            n_total**config.subsample_ratio
        )  # subsample size for variance estimation
    chunk_size_actual = min(config.chunk_size, n_total)

    def compute_delta_full(F):
        """Compute delta_full."""
        total = 0.0
        for start in range(0, n_total, chunk_size_actual):
            end = min(start + chunk_size_actual, n_total)
            W_chunk = compute_W_for_idx(F, torch.arange(start, end, device=device))
            total = total + (W_chunk * clicks[start:end]).sum(dim=1).sum()
        return total / n_total

    # Variance via subsampling: Var(D) ~ (n1/n) * E[(D_boot - D_full)^2]
    def compute_variance(F):
        b_chunk = min(config.bootstrap_chunk, config.n_bootstrap_var)

        delta_boots = []
        for b_start in range(0, config.n_bootstrap_var, b_chunk):
            b_end = min(b_start + b_chunk, config.n_bootstrap_var)
            cur_b = b_end - b_start

            idx_block = torch.randint(
                0, n_total, (cur_b, n1), device=device
            )  # (cur_b, n1)
            idx_flat = idx_block.reshape(-1)  # (cur_b * n1,)
            W = compute_W_for_idx(F, idx_flat)  # (cur_b * n1, K)
            contrib = (W * clicks[idx_flat]).sum(dim=1)  # (cur_b * n1,)
            boot_sums = contrib.view(cur_b, n1).sum(dim=1)  # (cur_b,)
            delta_boots.append(boot_sums / n1)  # (cur_b,)

        delta_boots = torch.cat(delta_boots)  # (n_bootstrap_var,)
        center = compute_delta_full(F).detach()
        var = ((delta_boots - center) ** 2).mean()
        return var * (n1 / n_total)  # scalar

    def compute_bias_taylor(F):
        """Compute Taylor expansion bias bound for position bias uncertainty.

        When true position bias p* differs from assumed p by at most eps_k per position,
        the bias is bounded by |grad_p D(p)| * eps_k (first-order Taylor bound).
        Uses bootstrap to estimate the gradient at the assumed p.
        """
        if eps_k.max() == 0:
            return torch.tensor(0.0, device=device)

        p_param = p.detach().clone().requires_grad_(True)  # (K,)
        b_chunk = min(config.bootstrap_chunk, config.n_bootstrap_bias)

        delta_boots = []
        for b_start in range(0, config.n_bootstrap_bias, b_chunk):
            b_end = min(b_start + b_chunk, config.n_bootstrap_bias)
            cur_b = b_end - b_start

            boot_idx = torch.randint(
                0, n_total, (cur_b, bias_sample_size_actual), device=device
            )  # (cur_b, bs)
            click_sub = clicks[boot_idx]  # (cur_b, bs, K)

            P1p_sub = P1_at_rank[boot_idx] * p_param  # (cur_b, bs, K, K)
            P2p_sub = P2_at_rank[boot_idx] * p_param  # (cur_b, bs, K, K)

            denom = (
                (F.unsqueeze(0).unsqueeze(0) * P1p_sub.unsqueeze(3))
                .sum(dim=-1)
                .clamp(min=1e-20)
            )  # (cur_b, bs, K, K)
            num = F.T.unsqueeze(0).unsqueeze(0) * P2p_sub  # (cur_b, bs, K, K)
            W_observed = (num / denom).sum(dim=-1)  # (cur_b, bs, K)
            delta_boots.append(
                (W_observed * click_sub).sum(dim=2).mean(dim=1)
            )  # (cur_b,)

        Delta_hat = torch.cat(delta_boots).mean()  # scalar
        grad_p = torch.autograd.grad(Delta_hat, p_param, create_graph=True)[0]  # (K,)
        return (grad_p.abs() * eps_k).sum()  # scalar

    def compute_loss():
        F = get_F()  # (K, K)
        var = compute_variance(F)  # scalar
        bias = (
            compute_bias_taylor(F)
            if eps_k.max() > 0
            else torch.tensor(0.0, device=device)
        )  # scalar
        return var + bias**2, var, bias

    best_loss = float("inf")
    best_F = None
    patience_counter = 0
    F_display_lines = 0
    iterator = tqdm(
        range(config.max_outer), disable=not config.verbose, desc="Optimizing F"
    )

    for epoch in iterator:
        optimizer.zero_grad()
        loss, var, bias = compute_loss()
        loss_val = loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_F = get_F().detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        loss.backward()
        optimizer.step()

        F_cur = get_F().detach()
        F_mean = F_cur.mean().item()
        iterator.set_postfix(
            mse=f"{loss_val:.6f}",
            var=f"{var.item():.6f}",
            bias_sq=f"{bias.item() ** 2:.6f}",
            F_mean=f"{F_mean:.6f}",
        )

        if config.verbose and epoch % 10 == 0:
            iterator.clear()
            F_display_lines = print_F_matrix(F_cur, F_display_lines)
            iterator.refresh()

        if patience_counter >= config.patience:
            if config.verbose:
                tqdm.write(
                    f"Early stopping: no improvement for {config.patience} iterations"
                )
            break

    # Use best F
    F_fixed = best_F if best_F is not None else get_F().detach()
    if config.verbose:
        print_F_matrix(F_fixed, -1)
    with torch.no_grad():
        var = compute_variance(F_fixed)

    bias = compute_bias_taylor(F_fixed).item() if eps_k.max() > 0 else 0.0
    mse = var.item() + bias**2

    return F_fixed, mse, var.item(), bias, epoch + 1


def run_optimization(config: FOptimizationConfig):
    """Run F matrix optimization from config, saving results to output_dir.

    Loads propensities and logged data, runs optimize_F for each repeat,
    and saves F matrices with MSE metrics.
    """
    device = get_device(config.device)

    if config.seed is not None:
        set_seed(config.seed)

    # Load and concatenate propensities for all subsets
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

    # Load position bias
    p = torch.tensor(config.position_bias, dtype=P1.dtype, device=device)
    eps_k = (
        torch.tensor(config.eps_k, dtype=P1.dtype, device=device)
        if config.eps_k
        else None
    )

    os.makedirs(config.output_dir, exist_ok=True)

    repeat_dirs = get_repeat_dirs(config.logging_dir)
    if config.single_repeat is not False:
        idx = 0 if config.single_repeat is True else config.single_repeat
        repeat_dirs = [repeat_dirs[idx]]
    elif config.n_repeats:
        repeat_dirs = repeat_dirs[: config.n_repeats]

    print(f"Logging: {config.logging_dir}")
    print(f"Target: {config.target_propensities_path}")
    print(f"Output: {config.output_dir}")
    print(f"Subsets: {config.subsets}")
    print(
        f"P1: {P1.shape}, P2: {P2.shape}, repeats: {len(repeat_dirs)}, single_repeat: {config.single_repeat}"
    )

    for repeat_name in tqdm(repeat_dirs, desc="Repeats"):
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
        if config.n_samples_per_query is not None:
            n_samples = min(n_samples, config.n_samples_per_query)
            rankings = rankings[:n_samples]
            clicks = clicks[:n_samples]
        rankings_flat = rankings.reshape(-1, rankings.shape[-1])
        clicks_flat = clicks.reshape(-1, clicks.shape[-1])

        F_fixed, mse, var, bias, n_steps = optimize_F(
            P1,
            P2,
            p,
            rankings_flat,
            clicks_flat,
            config,
            eps_k=eps_k,
        )

        torch.save(
            {"F": F_fixed.cpu(), "mse": mse, "var": var, "bias": bias},
            os.path.join(config.output_dir, f"{repeat_name}.pt"),
        )
        tqdm.write(
            f"{repeat_name}: mse={mse:.6f}, var={var:.6f}, bias^2={bias**2:.6f}, steps={n_steps}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--n_samples_per_query", type=int, help="Override n_samples_per_query"
    )
    parser.add_argument(
        "--n_repeats", type=int, help="Override n_repeats (disables single_repeat)"
    )
    parser.add_argument(
        "--single_repeat",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Use single repeat (default: 0, or specify index)",
    )
    parser.add_argument(
        "--bias_sample_size", type=int, help="Override bias_sample_size"
    )
    parser.add_argument(
        "--variance_sample_size", type=int, help="Override variance_sample_size"
    )
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    for k in [
        "n_samples_per_query",
        "n_repeats",
        "single_repeat",
        "bias_sample_size",
        "variance_sample_size",
    ]:
        if getattr(args, k) is not None:
            setattr(config, k, getattr(args, k))
    if args.n_repeats is not None:
        config.single_repeat = False  # CLI n_repeats overrides single_repeat

    run_optimization(config)
