"""Generalized Position-Based Model (GPBM) for off-policy evaluation."""

import numpy as np
import torch

from dataclasses import dataclass


@dataclass
class GPBM:
    """Generalized Position-Based Model estimator.

    Attributes:
        position_bias: (K,) - examination probability at each position
        F: (K, K) - position dependency matrix. F[j,k] = influence of logging position k on target position j.
           If None, created from window_size as a banded diagonal matrix (INTERPOL).
        window_size: If F not provided, creates F with 1s within window_size of diagonal.
        K: Number of positions (required if using window_size without F).
        device: torch device to place tensors on.

    """

    position_bias: np.ndarray
    F: np.ndarray | torch.Tensor = None  # (K, K)
    window_size: int = None
    K: int = None
    device: str = None  # if set, moves F and position_bias to this device

    def __post_init__(self):
        if not isinstance(self.position_bias, torch.Tensor):
            self.position_bias = torch.tensor(self.position_bias)
        if self.F is None and self.window_size is not None:
            self.F = self.create_diagonal_matrix(self.K, self.window_size)
        elif self.F is not None and self.window_size is not None:
            print(f"Ignoring window size {self.window_size}, using provided matrix F.")
        elif self.F is None and self.window_size is None:
            raise ValueError("Must provide either F or window_size.")

        if self.device is not None:
            self.F = self.F.to(self.device)
            self.position_bias = self.position_bias.to(self.device)

    @staticmethod
    def create_diagonal_matrix(K, window_size):
        """Create banded diagonal matrix with 1s within window_size of diagonal."""
        return torch.tensor(
            sum(np.eye(K, k=d) for d in range(-window_size, window_size + 1))
        )

    def compute_ips_weights_torch(self, batch: dict) -> torch.Tensor:
        """Compute IPS weights for documents at their observed positions.

        :param batch: Dict with "logging_propensities", "target_propensities",
            "logging_actions", "rewards" tensors.
        :return: Tensor of shape (batch_size, K) with weights for each position.
        """
        bs, n_docs, K = batch["logging_propensities"].shape
        mask = batch["rewards"] >= 0

        logging_props = batch["logging_propensities"]  # (bs, n_docs, K)
        target_props = batch["target_propensities"]  # (bs, n_docs, K)
        logging_actions = batch["logging_actions"]  # (bs, K)

        target_props_d = target_props[
            torch.arange(bs).unsqueeze(1), logging_actions
        ]  # (bs, K, K)
        logging_props_d = logging_props[
            torch.arange(bs).unsqueeze(1), logging_actions
        ]  # (bs, K, K)

        # F is (K, K) shared across all queries/docs
        # For position k, we need F[j, k] for all j
        # num[b,k,j] = F[j,k] * target_props_d[b,k,j] * position_bias[j]
        F_expanded = self.F.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        num = (
            F_expanded
            * target_props_d.unsqueeze(-1)
            * self.position_bias.view((1, 1, K, 1))
        )  # (bs, K, K, K)
        num = torch.diagonal(num, dim1=1, dim2=3).transpose(
            -1, -2
        )  # (bs, K, K) - extract F[j,k] for each k

        # denom[b,k,j] = sum_k' F[j,k'] * logging_props_d[b,k,k'] * position_bias[k']
        denom = (
            (F_expanded * logging_props_d.unsqueeze(2) * self.position_bias)
            .sum(-1)
            .clamp(min=1e-20)
        )  # (bs, K, K)

        weight = (num / denom).sum(-1)  # (bs, K)
        weight = torch.where(mask, weight, 0.0)
        return weight

    def compute_all_weights_torch(self, batch: dict) -> torch.Tensor:
        """Compute weights for all documents at all positions.

        Unlike compute_ips_weights_torch which only computes weights for observed
        placements, this computes weights for every (document, position) pair.

        :param batch: Dict with "logging_propensities" and "target_propensities" tensors.
        :return: Tensor of shape (batch_size, n_docs, K).
        """
        bs, n_docs, K = batch["logging_propensities"].shape

        logging_props = batch["logging_propensities"]  # (bs, n_docs, K)
        target_props = batch["target_propensities"]  # (bs, n_docs, K)
        pb = self.position_bias  # (K,)

        # F is (K, K), expand to (1, 1, K, K) for broadcasting
        F = self.F[None, None]  # (1, 1, K, K)

        # num[b,d,j,k] = F[j,k] * target_props[b,d,j] * pb[j]
        num = (
            F * target_props.unsqueeze(-1) * pb.view((1, 1, K, 1))
        )  # (bs, n_docs, K, K)

        # denom[b,d,j] = sum_k' F[j,k'] * logging_props[b,d,k'] * pb[k']
        denom = (
            (F * logging_props.unsqueeze(2) * pb).sum(-1).clamp(min=1e-20)
        )  # (bs, n_docs, K)

        # weight[b,d,k] = sum_j num[b,d,j,k] / denom[b,d,j]
        weight = (num / denom.unsqueeze(-1)).sum(2)  # (bs, n_docs, K)

        return weight
