"""Ensemble OPE estimators: SLOPE, BLUE, OPERA."""

import numpy as np
import torch
import cvxpy as cp


def bootstrap_ci_gpu(wr_sum, idx, alpha=0.05):
    """GPU bootstrap CI with doubled width (conservative).

    :param wr_sum: (n,) - per-query estimates
    :param idx: (n_bootstrap, n) - bootstrap indices (pre-generated for consistency across estimators)
    :param alpha: confidence level
    :return: (ci_low, ci_high) - CI bounds doubled from sample mean (conservative for SLOPE)
    """
    pred = wr_sum.mean()  # scalar
    means = wr_sum[idx].mean(dim=1)  # (n_bootstrap,)
    ci_low = torch.quantile(means, alpha / 2)  # scalar
    ci_high = torch.quantile(means, 1 - alpha / 2)  # scalar
    d_ci_low, d_ci_high = torch.abs(pred - ci_low), torch.abs(pred - ci_high)
    return pred - 2 * d_ci_low, pred + 2 * d_ci_high


def bootstrap_ci_snips_gpu(wr_sum, w_sum, K, idx, alpha=0.05):
    """GPU bootstrap CI for SNIPS (self-normalized IPS) with doubled width.

    SNIPS estimate = (sum wr_sum) / (sum w_sum) * K, where the K factor normalizes
    for the number of positions (each position contributes to the sum).

    :param wr_sum: (n,) - weighted rewards per query (sum_k w_k * r_k for each query)
    :param w_sum: (n,) - sum of weights per query (sum_k w_k for each query)
    :param K: number of positions (normalization factor)
    :param idx: (n_bootstrap, n) - bootstrap indices (pre-generated)
    :param alpha: confidence level
    :return: (ci_low, ci_high) - CI bounds doubled from sample mean
    """
    total_w = w_sum.sum()  # scalar
    pred = (wr_sum.sum() / total_w * K) if total_w > 0 else 0  # scalar
    total_w_boot = w_sum[idx].sum(dim=1)  # (n_bootstrap,)
    total_wr_boot = wr_sum[idx].sum(dim=1)  # (n_bootstrap,)
    ests = torch.where(
        total_w_boot > 0,
        total_wr_boot / total_w_boot * K,
        torch.zeros_like(total_w_boot),
    )  # (n_bootstrap,)
    ci_low = torch.quantile(ests, alpha / 2)  # scalar
    ci_high = torch.quantile(ests, 1 - alpha / 2)  # scalar
    d_ci_low, d_ci_high = torch.abs(pred - ci_low), torch.abs(pred - ci_high)
    return pred - 2 * d_ci_low, pred + 2 * d_ci_high


def apply_slope(ordered_names, ci_bounds):
    """SLOPE estimator selection via confidence interval intersection.

    Walks through estimators from lowest to highest variance, selecting the
    highest-variance estimator whose CI still intersects all previous CIs.
    This balances bias-variance tradeoff adaptively.

    Based on: Y. Su, P. Srinath and A. Krishnamurthy. "Adaptive Estimator Selection for Off-Policy Evaluation." ICML '20.

    """
    if not ordered_names:
        return None
    lo, hi = ci_bounds[ordered_names[0]]
    selected = ordered_names[0]
    for name in ordered_names[1:]:
        l, h = ci_bounds[name]
        new_lo, new_hi = max(lo, l), min(hi, h)
        if new_lo <= new_hi:
            lo, hi, selected = new_lo, new_hi, name
        else:
            break
    return selected


def compute_opera(
    estimator_names, data, K, n_bootstrap=100, subsample_exp=0.6, device="cpu"
):
    """OPERA: Offline Policy Evaluation with Re-weighted Aggregates.

    Combines multiple OPE estimators by minimizing bootstrap-estimated MSE.

    Based on Nie et al. "OPERA: Automatic Offline Policy Evaluation with Re-weighted Aggregates of Multiple Estimators". NeurIPS '24.

    :param estimator_names: List of estimator names to combine
    :param data: Dict mapping estimator names to {"wr_sum": [tensors], "w_sum": [tensors]}
        where each list contains per-batch tensors that get concatenated
    :param K: Number of positions (for SNIPS normalization)
    :param n_bootstrap: Number of bootstrap samples for MSE matrix estimation
    :param subsample_exp: exponent in n1 = n^exp - smaller values give more variance reduction
        but may increase bias in MSE estimate (paper recommends ~0.6-0.7)
    :param device: torch device
    :return: (opera_estimate, weights) tuple, or (None, None) if failed
    """
    n_est = len(estimator_names)
    if n_est == 0:
        return None, None

    # Collect per-query data
    wr_all = []  # list of (n,) tensors
    w_sum_all = []  # list of (n,) tensors
    snips_mask = []

    for name in estimator_names:
        wr_all.append(torch.cat(data[name]["wr_sum"]))
        w_sum_all.append(torch.cat(data[name]["w_sum"]))
        snips_mask.append("_snips" in name)

    wr_all = torch.stack(wr_all)  # (n_est, n)
    w_sum_all = torch.stack(w_sum_all)  # (n_est, n)
    snips_mask = torch.tensor(snips_mask, device=device)  # (n_est,)
    n = wr_all.shape[1]
    n1 = max(int(n**subsample_exp), 1)

    # Full-data estimates
    ips_est = wr_all.mean(dim=1)  # (n_est,)
    snips_est = wr_all.sum(dim=1) / w_sum_all.sum(dim=1) * K  # (n_est,)
    full_estimates = torch.where(snips_mask, snips_est, ips_est)  # (n_est,)

    # Generate all bootstrap indices at once
    idx = torch.randint(0, n, (n_bootstrap, n1), device=device)  # (n_bootstrap, n1)

    # Gather bootstrap samples: (n_est, n_bootstrap, n1)
    wr_boot = wr_all[:, idx]  # (n_est, n_bootstrap, n1)
    w_sum_boot = w_sum_all[:, idx]  # (n_est, n_bootstrap, n1)

    # Compute bootstrap estimates
    ips_boot = wr_boot.mean(dim=2)  # (n_est, n_bootstrap)
    snips_boot = wr_boot.sum(dim=2) / w_sum_boot.sum(dim=2) * K  # (n_est, n_bootstrap)
    boot_estimates = torch.where(
        snips_mask.unsqueeze(1), snips_boot, ips_boot
    ).T  # (n_bootstrap, n_est)

    # OPERA minimizes estimated MSE = E[(a^T est - true)^2] via bootstrap
    # A_hat estimates the MSE matrix: A[i,j] = E[(est_i - true)(est_j - true)]
    delta = boot_estimates - full_estimates  # (n_bootstrap, n_est)
    A_hat = (delta.T @ delta) / n_bootstrap * (n1 / n)  # (n_est, n_est)

    # Solve constrained optimization: min a^T A a  s.t. sum(a) = 1
    # Regularize if needed
    while not torch.all(torch.linalg.eigvalsh(A_hat) >= 0):
        A_hat = A_hat + torch.eye(n_est, device=A_hat.device) * 1e-6
    A_np = A_hat.cpu().numpy()
    x = cp.Variable(n_est)
    objective = cp.Minimize(cp.quad_form(x, A_np))
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if x.value is None:
        return None, None
    alpha = x.value

    opera_estimate = float((alpha * full_estimates.cpu().numpy()).sum())
    return opera_estimate, alpha


def compute_blue(estimator_names, data, K):
    """BLUE: Best Linear Unbiased Estimator.

    Combines estimators assuming they are all unbiased, weighting by inverse covariance
    to minimize variance. Uses delta method for SNIPS covariance estimation.

    Based on O. Jeunen. "Meta Off-Policy Estimation". RecSys '25.

    :param estimator_names: List of estimator names to combine (e.g., ["w0", "w1_snips"])
    :param data: Dict mapping estimator names to {"wr_sum": [tensors], "w_sum": [tensors]}
        where each list contains per-batch tensors that get concatenated
    :param K: Number of positions (for SNIPS normalization)
    :return: (blue_estimate, blue_variance, weights_dict) or (None, None, None) if failed

    Note: Assumes estimators are unbiased. If they have different biases, BLUE is invalid.
    """
    n_est = len(estimator_names)
    if n_est == 0:
        return None, None, None

    wr_sum_all = []  # list of (n,) tensors
    w_sum_all = []  # list of (n,) tensors

    for name in estimator_names:
        wr_sum_all.append(torch.cat(data[name]["wr_sum"]))
        w_sum_all.append(torch.cat(data[name]["w_sum"]))

    wr_sum_all = torch.stack(wr_sum_all)  # (n_est, n)
    w_sum_all = torch.stack(w_sum_all)  # (n_est, n)
    n = wr_sum_all.shape[1]

    ips_means = wr_sum_all.mean(dim=1)  # (n_est,)
    sn_means = w_sum_all.mean(dim=1)  # (n_est,)

    # Compute means (IPS vs SNIPS)
    means = torch.zeros(n_est, device=wr_sum_all.device)
    for i, name in enumerate(estimator_names):
        if "_snips" in name:
            total_w = w_sum_all[i].sum()
            means[i] = (wr_sum_all[i].sum() / total_w * K) if total_w > 0 else 0
        else:
            means[i] = ips_means[i]

    # Compute covariances
    stacked = torch.cat([wr_sum_all, w_sum_all], dim=0)  # (2*n_est, n)
    full_cov = torch.cov(stacked)  # (2*n_est, 2*n_est)

    cov_rr = full_cov[:n_est, :n_est]  # (n_est, n_est)
    cov_ww = full_cov[n_est:, n_est:]  # (n_est, n_est)
    cov_rw = full_cov[:n_est, n_est:]  # (n_est, n_est)

    snips_mask = torch.tensor(
        ["_snips" in name for name in estimator_names], device=wr_sum_all.device
    )

    # Delta method covariance for SNIPS: Var(r/w) ~ (1/mu_w)^2 Var(r) - 2(mu_r/mu_w^3) Cov(r,w) + (mu_r^2/mu_w^4) Var(w)
    cov_matrix = torch.zeros((n_est, n_est), device=wr_sum_all.device)

    for i in range(n_est):
        for j in range(n_est):
            si, sj = snips_mask[i], snips_mask[j]
            if not si and not sj:
                cov_matrix[i, j] = cov_rr[i, j]
            elif si and not sj:
                cov_matrix[i, j] = (
                    cov_rr[i, j] / sn_means[i]
                    - ips_means[i] / sn_means[i] ** 2 * cov_rw[j, i]
                ) * K
            elif not si and sj:
                cov_matrix[i, j] = (
                    cov_rr[i, j] / sn_means[j]
                    - ips_means[j] / sn_means[j] ** 2 * cov_rw[i, j]
                ) * K
            else:
                cov_matrix[i, j] = K**2 * (
                    cov_rr[i, j] / (sn_means[i] * sn_means[j])
                    - ips_means[j] / (sn_means[i] * sn_means[j] ** 2) * cov_rw[i, j]
                    - ips_means[i] / (sn_means[i] ** 2 * sn_means[j]) * cov_rw[j, i]
                    + ips_means[i]
                    * ips_means[j]
                    / (sn_means[i] ** 2 * sn_means[j] ** 2)
                    * cov_ww[i, j]
                )

    cov_matrix = cov_matrix / n

    # Regularize if needed
    while not torch.all(torch.linalg.eigvalsh(cov_matrix) >= 0):
        cov_matrix = cov_matrix + torch.eye(n_est, device=cov_matrix.device) * 1e-6

    # BLUE formula: a = Cov^{-1} 1 / (1^T Cov^{-1} 1)
    ones = torch.ones(n_est, device=cov_matrix.device)
    inv_cov = torch.linalg.inv(cov_matrix)
    denom = ones @ inv_cov @ ones
    weights = inv_cov @ ones / denom
    blue_mean = weights @ means
    blue_var = 1.0 / denom

    return (
        blue_mean.item(),
        blue_var.item(),
        {name: w.item() for name, w in zip(estimator_names, weights)},
    )
