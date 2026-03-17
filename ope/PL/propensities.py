"""Plackett-Luce propensity computation via Gauss-Legendre quadrature.

Based on: N. Knyazev and H. Oosterhuis. "Sample-Free Almost-Exact Estimation of Plackett-Luce Propensities for
Off-Policy Ranking". ECIR '26."""

import numpy as np
import torch
from functools import lru_cache


@lru_cache(maxsize=8)
def _get_legendre_np(N):
    """Cached Gauss-Legendre quadrature points (numpy)."""
    return np.polynomial.legendre.leggauss(N)


def _get_legendre_cached(N, device, dtype):
    """Get Gauss-Legendre points on specified device/dtype."""
    xi_np, wi_np = _get_legendre_np(N)
    return torch.tensor(xi_np, device=device, dtype=dtype), torch.tensor(
        wi_np, device=device, dtype=dtype
    )


def sample_rankings_batched(logits, n_samples, mask, K, batch_size=1000):
    """Sample rankings from Plackett-Luce model via Gumbel-top-k trick, batched over queries.

    :param logits: (n_queries, n_docs) - scores from ranking model
    :param n_samples: number of ranking samples per query
    :param mask: (n_queries, n_docs) - valid document mask
    :param K: number of positions to rank
    :param batch_size: queries per batch
    :return: rankings tensor of shape (n_samples, n_queries, K)
    """
    n_queries, n_docs = logits.shape
    device, dtype = logits.device, logits.dtype
    log_scores = logits.masked_fill(~mask, float("-inf"))

    rankings_list = []
    for i in range(0, n_queries, batch_size):
        batch_log_scores = log_scores[i : i + batch_size]
        batch_size_actual = batch_log_scores.shape[0]
        gumbel = -torch.log(
            -torch.log(
                torch.empty(
                    n_samples, batch_size_actual, n_docs, device=device, dtype=dtype
                ).uniform_(torch.finfo(dtype).tiny, 1 - torch.finfo(dtype).eps)
            )
        )
        _, batch_rankings = (batch_log_scores + gumbel).topk(K, dim=2)
        rankings_list.append(batch_rankings)

    return torch.cat(rankings_list, dim=1)


def compute_pl_propensities_mpl(logits, K, N=200, percentile=1e-8):
    """Compute Plackett-Luce placement probabilities P(doc d at position k).

    Uses Gauss-Legendre quadrature over Gumbel random variables.

    :param logits: (batch_size, n_docs) - document scores (-inf for padding).
    :param K: Number of positions.
    :param N: Quadrature points.
    :param percentile: Gumbel quantile bounds.
    :return: Tensor of shape (batch_size, n_docs, K) - P(y_{k+1} = d) for each doc d, position k.
    """
    device = logits.device
    dtype = logits.dtype

    # Gumbel quantiles for integration bounds
    c1 = -np.log(-np.log(percentile))
    c2 = -np.log(-np.log(1 - percentile))

    # Gauss-Legendre quadrature (cached)
    xi, wi = _get_legendre_cached(N, device, dtype)

    # Integration bounds
    valid_mask = logits > -float("inf")
    logits_for_min = torch.where(
        valid_mask, logits, torch.full_like(logits, float("inf"))
    )
    logits_for_max = torch.where(
        valid_mask, logits, torch.full_like(logits, -float("inf"))
    )
    a = logits_for_min.min(dim=1, keepdim=True).values + c1
    b = logits_for_max.max(dim=1, keepdim=True).values + c2

    alpha = (b - a) / 2
    beta = (a + b) / 2
    x_points = alpha * xi + beta  # (batch_size, N)

    # z = m_d - x: (batch_size, n_docs, N)
    z = logits.unsqueeze(-1) - x_points.unsqueeze(1)

    # Gumbel PDF: log f_d(x) = z - exp(z)
    log_f_d = z - torch.exp(z)

    # Gumbel survival function: log(1 - CDF) = log(1 - exp(-exp(z)))
    log_cdf = -torch.exp(z)
    log_sf = log1mexp(log_cdf)

    # Poisson binomial excluding each document
    log_prob_S = compute_poisson_binomial_excluding_d_vectorized(log_sf, K, valid_mask)

    # Equation 12: integrate
    log_wi = torch.log(wi)  # (N,)
    log_alpha = torch.log(alpha)  # (batch_size, 1)

    # (batch_size, n_docs, K, N)
    log_integrand = log_wi + log_f_d.unsqueeze(2) + log_prob_S

    log_probs = log_alpha.unsqueeze(-1) + torch.logsumexp(log_integrand, dim=-1)

    # Mask padding
    log_probs = torch.where(
        valid_mask.unsqueeze(-1), log_probs, torch.full_like(log_probs, -float("inf"))
    )

    return torch.exp(log_probs)


def compute_poisson_binomial_excluding_d_vectorized(log_sf, K, valid_mask):
    """Compute P(S = k | d excluded) for Poisson binomial via forward-backward convolution.

    S = number of docs with Gumbel max > threshold x. Uses DP to compute P(S=k) excluding
    each doc d, needed for placement probability integration.
    """
    batch_size, n_docs, N = log_sf.shape
    device = log_sf.device
    dtype = log_sf.dtype

    log_p = torch.where(
        valid_mask.unsqueeze(-1), log_sf, torch.full_like(log_sf, -float("inf"))
    )
    log_1mp = torch.where(
        valid_mask.unsqueeze(-1), log1mexp(log_sf), torch.zeros_like(log_sf)
    )

    # Forward pass
    c_forward_list = []
    log_conv = torch.zeros(batch_size, 1, N, device=device, dtype=dtype)
    c_forward_list.append(log_conv)

    for i in range(n_docs):
        log_conv = convolve_log_vectorized(
            log_conv, log_1mp[:, i, :], log_p[:, i, :], K
        )
        c_forward_list.append(log_conv)

    # Backward pass
    c_backward_list = [None] * (n_docs + 1)
    log_conv = torch.zeros(batch_size, 1, N, device=device, dtype=dtype)
    c_backward_list[n_docs] = log_conv

    for i in range(n_docs - 1, -1, -1):
        log_conv = convolve_log_vectorized(
            log_conv, log_1mp[:, i, :], log_p[:, i, :], K
        )
        c_backward_list[i] = log_conv

    # Combine via anti-diagonal sum
    log_prob_S = torch.full(
        (batch_size, n_docs, K, N), -float("inf"), device=device, dtype=dtype
    )

    for d in range(n_docs):
        fwd = c_forward_list[d]  # (batch_size, len_fwd, N)
        bwd = c_backward_list[d + 1]  # (batch_size, len_bwd, N)
        len_fwd = fwd.shape[1]
        len_bwd = bwd.shape[1]

        for k in range(K):
            log_sum = torch.full(
                (batch_size, N), -float("inf"), device=device, dtype=dtype
            )
            for i in range(min(k + 1, len_fwd)):
                j = k - i
                if 0 <= j < len_bwd:
                    log_sum = torch.logaddexp(log_sum, fwd[:, i, :] + bwd[:, j, :])
            log_prob_S[:, d, k, :] = log_sum

    return log_prob_S


def convolve_log_vectorized(log_conv, log_1mp, log_p, K):
    """Convolve distribution with [1-p, p] in log space."""
    batch_size, current_len, N = log_conv.shape
    new_len = min(current_len + 1, K + 1)
    device = log_conv.device
    dtype = log_conv.dtype

    # Pad conv to new_len
    if current_len < new_len:
        padded = torch.full(
            (batch_size, new_len, N), -float("inf"), device=device, dtype=dtype
        )
        padded[:, :current_len, :] = log_conv
    else:
        padded = log_conv[:, :new_len, :]

    # Shift for p * conv[j-1]
    shifted = torch.full(
        (batch_size, new_len, N), -float("inf"), device=device, dtype=dtype
    )
    shifted[:, 1:, :] = (
        padded[:, : new_len - 1, :] if new_len > 1 else shifted[:, 1:, :]
    )

    # new[j] = logsumexp(log_1mp + conv[j], log_p + conv[j-1])
    term1 = log_1mp.unsqueeze(1) + padded
    term2 = log_p.unsqueeze(1) + shifted

    return torch.logaddexp(term1, term2)


def log1mexp(x):
    """Compute log(1 - exp(x)) numerically stable. x should be <= 0."""
    return torch.where(
        x < -0.693147, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
    )
