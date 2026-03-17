"""Item-specific Position Bias Model (IPM).

Implements:
1. Item-specific position bias: pb[item, k] = dcg_discount(k)^alpha[item]
2. Click simulation under IPM
"""

import torch
import numpy as np


def dcg_discount(K):
    """Standard DCG discount: 1/log2(k+2) for k=0..K-1."""
    return 1.0 / np.log2(np.arange(2, K + 2))


def generate_item_alphas(n_items, seed=None):
    """Generate per-item interpolation weights uniformly in [0, 1].

    :param n_items: Number of unique items
    :param seed: Random seed
    :return: alphas array of shape (n_items,) with values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, n_items)


def generate_item_alphas_from_features(features, feature_frac=0.1, seed=None):
    """Generate per-item interpolation weights from features using a random nonlinear mapping.

    Maps features to weights in [0, 1] via: random feature subset -> random MLP -> quantile

    :param features: (n_queries, n_docs, n_feat) or (n_items, n_feat) feature tensor
    :param feature_frac: fraction of features to use (e.g., 0.1 = 10%)
    :param seed: Random seed
    :return: alphas array of shape (n_queries, n_docs) or (n_items,) with values in [0, 1]
    """
    torch.manual_seed(seed if seed is not None else 0)

    orig_shape = features.shape[:-1]
    n_feat = features.shape[-1]
    X = features.reshape(-1, n_feat)  # (n_items, n_feat)
    n_items = X.shape[0]

    # Select random subset of features
    n_select = max(1, int(n_feat * feature_frac))
    feat_idx = torch.randperm(n_feat)[:n_select]
    X_sub = X[:, feat_idx]  # (n_items, n_select)

    # Random MLP: n_select -> 16 -> 1
    model = torch.nn.Sequential(
        torch.nn.Linear(n_select, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )

    with torch.no_grad():
        scores = model(X_sub).squeeze(-1)  # (n_items,)

    # Map to quantiles [0, 1]
    ranks = scores.argsort().argsort().float()
    alphas = ranks / (n_items - 1)

    return alphas.reshape(orig_shape).numpy()


def generate_item_alphas_from_model(model, features):
    """Generate per-item alphas from a trained ranking model's scores via quantile normalization."""
    orig_shape = features.shape[:-1]
    with torch.no_grad():
        scores = model(features.reshape(-1, features.shape[-1])).squeeze(-1)
    ranks = scores.argsort().argsort().float()
    alphas = ranks / (len(ranks) - 1)
    return alphas.reshape(orig_shape).cpu().numpy()


def compute_item_position_bias(alphas, K, alpha_bound=1.4, position_bias=None):
    """Compute position bias curves for each item with symmetric deviation.

    pb[i, k] = base[k] + (2*alpha[i] - 1) * (base[k] - base[k]^alpha_bound)

    where alpha[i] in [0, 1]:
    - alpha=0: pb = base^alpha_bound (lowest)
    - alpha=0.5: pb = base (no deviation)
    - alpha=1: pb = 2*base - base^alpha_bound (highest)

    :param alphas: (n_items,) per-item values in [0, 1]
    :param K: Number of positions
    :param alpha_bound: Exponent defining max deviation
    :param position_bias: (K,) base position bias curve. If None, uses DCG discount.
    :return: pb array of shape (n_items, K), position bias per item per position
    """
    base = np.asarray(position_bias) if position_bias is not None else dcg_discount(K)
    deviation = base - base**alpha_bound  # (K,)
    # alpha=0 -> -1, alpha=0.5 -> 0, alpha=1 -> +1
    weights = 2 * alphas[:, None] - 1  # (n_items, 1)
    return base[None, :] + weights * deviation[None, :]  # (n_items, K)


def simulate_clicks_ipm(rankings, labels, item_pb):
    """Simulate clicks using Item-specific Position Bias Model.

    Click probability = P(click|d,k) = pb[d,k] * rel[d]

    :param rankings: (n_samples, n_queries, K) - document indices per position
    :param labels: (n_queries, n_docs) - relevance labels in [0,1], -1 for padding
    :param item_pb: (n_queries, n_docs, K) - position bias per item per position
    :return: clicks tensor of shape (n_samples, n_queries, K) - binary click indicators
    """
    n_samples, n_queries, K = rankings.shape

    labels_masked = labels.masked_fill(labels == -1, 0)  # (n_queries, n_docs)

    # Get relevance at each ranking position
    rel_at_pos = torch.gather(
        labels_masked.unsqueeze(0).expand(n_samples, -1, -1), 2, rankings
    )  # (n_samples, n_queries, K)

    # Get position bias at each ranking position
    q_idx = torch.arange(n_queries).view(1, -1, 1)
    k_idx = torch.arange(K).view(1, 1, -1)
    pb_at_pos = item_pb[q_idx, rankings, k_idx]  # (n_samples, n_queries, K)

    click_prob = rel_at_pos * pb_at_pos
    return torch.bernoulli(click_prob)
