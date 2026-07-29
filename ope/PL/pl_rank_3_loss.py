"""Plackett-Luce ranking loss with gradient estimation (PL-Rank-3).

Implements differentiable ranking loss using Gumbel-softmax sampling for gradient
estimation. Based on: H. Oosterhuis. "Learning-to-Rank at the Speed of Sampling:
Plackett-Luce Gradient Estimation With Minimal Computational Complexity." SIGIR '22.
"""

import torch
from torch.autograd import Function


def _sample_gumbel(shape, device, dtype):
    """Sample from Gumbel(0, 1) distribution."""
    finfo = torch.finfo(dtype)
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(
        finfo.tiny, 1 - finfo.eps
    )
    return -torch.log(-torch.log(u))


def sample_ndcg(scores, labels, rank_weights, n_samples=1, cutoff=None):
    """Estimate NDCG via Gumbel sampling (for loss value logging)."""
    batch_size, n_docs = scores.shape
    device, dtype = scores.device, scores.dtype
    cutoff = min(cutoff or n_docs, n_docs)

    mask = (labels != -1) & torch.isfinite(scores)
    log_scores = scores.masked_fill(~mask, float("-inf"))

    gumbel = _sample_gumbel((n_samples, batch_size, n_docs), device, dtype)
    _, rankings = (log_scores + gumbel).topk(cutoff, dim=2)

    sampled_labels = torch.gather(
        labels.unsqueeze(0).expand(n_samples, -1, -1), 2, rankings
    ).clamp(min=0)
    dcg = (sampled_labels * rank_weights[:cutoff]).sum(dim=2).mean(dim=0)

    ideal_labels, _ = labels.clamp(min=0).masked_fill(~mask, 0).topk(cutoff, dim=1)
    idcg = (ideal_labels * rank_weights[:cutoff]).sum(dim=1).clamp(min=1e-20)

    return (dcg / idcg).mean()


class PLRank3Function(Function):
    """Autograd function for PL-Rank-3 loss with custom backward pass."""

    @staticmethod
    def forward(ctx, scores, labels, rank_weights, n_samples, cutoff, loss_value):
        ctx.save_for_backward(scores, labels, rank_weights)
        ctx.n_samples = n_samples
        ctx.cutoff = cutoff
        return loss_value

    @staticmethod
    def backward(ctx, grad_output):
        scores, labels, rank_weights = ctx.saved_tensors
        grad = pl_rank_3_gradient(
            scores, labels, rank_weights, ctx.n_samples, ctx.cutoff
        )
        return -grad * grad_output / scores.size(0), None, None, None, None, None


def pl_rank_3_gradient(
    scores, labels, rank_weights, n_samples=1, cutoff=None, _gumbel=None
):
    """Compute PL-Rank-3 gradient estimate via Gumbel sampling."""
    batch_size, n_docs = scores.shape
    device, dtype = scores.device, scores.dtype
    cutoff = min(cutoff or n_docs, n_docs)

    mask = (labels != -1) & torch.isfinite(scores)
    if (mask.sum(dim=1) <= 1).all():
        return torch.zeros_like(scores)

    log_scores = (
        scores - scores.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True)[0]
    )
    log_scores = log_scores.masked_fill(~mask, float("-inf"))

    gumbel = (
        _gumbel
        if _gumbel is not None
        else _sample_gumbel((n_samples, batch_size, n_docs), device, dtype)
    )
    _, rankings = (log_scores + gumbel).topk(cutoff, dim=2)

    labels_exp = labels.unsqueeze(0).expand(n_samples, -1, -1)
    sampled_labels = torch.gather(labels_exp, 2, rankings).clamp(min=0)
    cumsum_labels = (
        (sampled_labels * rank_weights[:cutoff]).flip([2]).cumsum(dim=2).flip([2])
    )

    result = torch.zeros(batch_size, n_docs, device=device, dtype=dtype)
    if cutoff > 1:
        flat_idx = (
            rankings[:, :, :-1]
            + torch.arange(batch_size, device=device).view(1, -1, 1) * n_docs
        ).reshape(-1)
        result.view(-1).scatter_add_(0, flat_idx, cumsum_labels[:, :, 1:].reshape(-1))
        result /= n_samples

    log_total = torch.logsumexp(log_scores, dim=1, keepdim=True)  # (batch, 1)
    sampled_log_scores = torch.gather(
        log_scores.unsqueeze(0).expand(n_samples, -1, -1), 2, rankings
    )

    cumsum_log_exp = torch.logcumsumexp(sampled_log_scores, dim=2)
    ratio = cumsum_log_exp[:, :, :-1] - log_total.unsqueeze(0)  # negative values
    log_one_minus_ratio = torch.log1p(
        -torch.exp(ratio.clamp(max=-1e-7))
    )  # log(1 - exp(ratio))
    log_denom = torch.cat(
        [
            log_total.unsqueeze(0).expand(n_samples, -1, -1),
            log_total.unsqueeze(0) + log_one_minus_ratio,
        ],
        dim=2,
    )

    inv_denom = torch.exp(-log_denom)  # 1/denom
    cumsum_weight_denom = (rank_weights[:cutoff] * inv_denom).cumsum(dim=2)
    cumsum_reward_denom = (cumsum_labels * inv_denom).cumsum(dim=2)

    cwd_last = cumsum_weight_denom[:, :, -1:]
    crd_last = cumsum_reward_denom[:, :, -1:]

    exp_scores = torch.exp(log_scores).masked_fill(~mask, 0.0)
    second_part = (
        -exp_scores * crd_last + (labels.clamp(min=0) * exp_scores * cwd_last) * mask
    )

    sampled_exp = torch.gather(
        exp_scores.unsqueeze(0).expand(n_samples, -1, -1), 2, rankings
    )
    sampled_diff = (
        sampled_labels * sampled_exp * cumsum_weight_denom
        - sampled_exp * cumsum_reward_denom
    )
    second_part.scatter_(2, rankings, sampled_diff)

    return (result + second_part.masked_fill(~mask, 0.0).mean(dim=0)).masked_fill(
        ~mask, 0.0
    )


class PLRank3Loss(torch.nn.Module):
    """Plackett-Luce ranking loss with unbiased gradient estimation.

    :param cutoff: Only consider top-k positions for loss (default: 10).
    :param n_samples: Number of Gumbel samples for gradient estimation (default: 1).
    :param n_samples_metric: If > 0, compute sampled NDCG as loss value for logging (default: 0).
    """

    def __init__(self, cutoff=10, n_samples=1, n_samples_metric=0):
        super().__init__()
        self.cutoff = cutoff
        self.n_samples = n_samples
        self.n_samples_metric = n_samples_metric
        self.register_buffer(
            "rank_weights",
            1.0 / torch.log2(torch.arange(cutoff, dtype=torch.float32) + 2),
        )

    def forward(self, scores, labels):
        if self.n_samples_metric > 0:
            with torch.no_grad():
                loss_value = -sample_ndcg(
                    scores,
                    labels,
                    self.rank_weights,
                    self.n_samples_metric,
                    self.cutoff,
                )
        else:
            loss_value = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return PLRank3Function.apply(
            scores, labels, self.rank_weights, self.n_samples, self.cutoff, loss_value
        )
