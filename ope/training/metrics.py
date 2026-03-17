"""Ranking metrics: NDCG and score spread."""

import torch


def compute_ndcg(scores, labels, k=None, dataset_normalized=True):
    """Normalized Discounted Cumulative Gain.

    :param scores: (batch, n_docs) - model scores
    :param labels: (batch, n_docs) - relevance labels (-1 for padding)
    :param k: cutoff position (None = use all)
    :param dataset_normalized: if True, sum DCG/IDCG across batch then divide (default);
        if False, compute per-query NDCG then average
    """
    mask = labels != -1
    scores = scores.masked_fill(~mask, float("-inf"))
    k = k or scores.size(1)
    _, idx = scores.topk(min(k, scores.size(1)), dim=1)
    gains = torch.gather(labels.clamp(min=0), 1, idx)
    discounts = 1.0 / torch.log2(
        torch.arange(gains.size(1), device=gains.device).float() + 2
    )
    dcg = (gains * discounts).sum(dim=1)
    ideal, _ = (
        labels.clamp(min=0).masked_fill(~mask, 0).topk(min(k, labels.size(1)), dim=1)
    )
    idcg = (ideal * discounts[: ideal.size(1)]).sum(dim=1)
    if dataset_normalized:
        return (dcg.sum() / idcg.sum().clamp(min=1e-10)).item()
    valid = idcg > 0
    return (dcg[valid] / idcg[valid]).mean().item() if valid.any() else 0.0


def compute_score_spread(scores, labels, k=None):
    """Score spread - difference between max and min scores (for diagnostics)."""
    mask = labels != -1
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    max_scores = masked_scores.max(dim=1).values
    masked_scores = scores.masked_fill(~mask, float("inf"))
    min_scores = masked_scores.min(dim=1).values
    return (max_scores - min_scores).mean().item()


METRICS = {"ndcg": compute_ndcg, "score_spread": compute_score_spread}


class BestMetricsTracker:
    """Track best values for each metric across epochs.

    :param minimize: tuple of metric name substrings to minimize (default: ("loss",))
    """

    def __init__(self, minimize=("loss",)):
        self.best = {}
        self.minimize = minimize

    def update(self, metrics, epoch):
        for k, v in metrics.items():
            is_min = any(m in k for m in self.minimize)
            if k not in self.best or (
                v < self.best[k][0] if is_min else v > self.best[k][0]
            ):
                self.best[k] = (v, epoch)

    def get_best(self):
        return {k: v for k, (v, _) in self.best.items()}
