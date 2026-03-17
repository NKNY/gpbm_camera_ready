"""Neural ranking model for learning-to-rank."""

import torch
import torch.nn as nn

from ope.PL.pl_rank_3_loss import PLRank3Loss


class RankingModel(nn.Module):
    """MLP ranking model that outputs relevance scores for documents.

    :param n_feat_input: Number of input features per document.
    :param n_feat: Hidden layer size.
    :param n_layers: Number of hidden layers.
    :param activation: Activation function class (default: nn.ReLU).
    :param dropout: Dropout probability between layers (default: 0.0).
    :param bias: Whether to use bias in linear layers (default: True).
    :param output_scale: If set, applies output_scale * tanh(output) to bound scores.
    :param temperature: Multiplier for output scores, controls softmax sharpness (default: 1.0).
    :param loss_fn: Loss function class (default: PLRank3Loss).
    """

    def __init__(
        self,
        n_feat_input,
        n_feat,
        n_layers,
        activation=nn.ReLU,
        dropout=0.0,
        bias=True,
        output_scale=None,
        temperature=1.0,
        loss_fn=PLRank3Loss,
        *loss_args,
        **loss_kwargs,
    ):
        super().__init__()
        layers = [nn.Linear(n_feat_input, n_feat, bias=bias), activation()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_feat, n_feat, bias=bias), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(n_feat, 1, bias=bias))
        self.net = nn.Sequential(*layers)
        self.output_scale = output_scale
        self.temperature = temperature
        self.loss_fn = loss_fn(*loss_args, **loss_kwargs)

    def forward(self, x):
        out = self.net(x).squeeze(-1)
        if self.output_scale is not None:
            out = self.output_scale * torch.tanh(out)
        return out * self.temperature

    def compute_loss(self, x, labels):
        return self.loss_fn(self(x), labels)
