"""Training configuration dataclass."""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable, Optional, List

from ope.PL.pl_rank_3_loss import PLRank3Loss


@dataclass
class Config:
    """Training configuration for ranking models."""

    # Dataset
    dataset: str = "yahoo"  # dataset name: "yahoo", "mslr", "mslr_1", etc.
    n_docs_min: int = 1  # min docs per query (filter out smaller)
    n_docs_max: Optional[int] = None  # max docs per query (None = dataset default)

    # Feature selection
    feature_select_k: Optional[int] = None  # select top k features by correlation
    feature_select_pct: Optional[float] = None  # select top pct% features (0-100)
    feature_select_mode: str = "best"  # "best"/"worst"/"middle" by |correlation|
    feature_indices: Optional[List[int]] = (
        None  # manual feature indices (overrides above)
    )

    # Model architecture
    n_feat: Optional[int] = None  # input features (None = auto from dataset)
    n_hidden: int = 32  # hidden layer size
    n_layers: int = 2  # number of hidden layers
    activation: type = nn.Sigmoid  # activation function class
    dropout: float = 0.0  # dropout probability between layers
    bias: bool = True  # use bias in linear layers
    output_scale: Optional[float] = None  # if set, output = scale * tanh(output)
    normalize_features: bool = True  # min-max normalize features per query

    # Loss
    loss_fn: type = PLRank3Loss  # loss function class
    loss_kwargs: dict = field(
        default_factory=lambda: {"cutoff": 10}
    )  # loss function kwargs
    n_samples_grad: int = 100  # Gumbel samples for gradient estimation
    n_samples_train: int = 100  # samples for train loss computation
    n_samples_val: int = 100  # samples for validation loss
    n_samples_test: int = 1000  # samples for test loss

    # Dataloader
    batch_size: int = 128  # queries per training batch
    relevance_transform: Callable = field(
        default_factory=lambda: lambda x: (2**x - 1) / 15
    )  # label transform
    num_workers: int = 0  # dataloader workers
    pin_memory: bool = True  # pin memory for GPU transfer
    persistent_workers: bool = False  # keep workers alive between epochs

    # Training loop
    lr: float = 1e-3  # optimizer learning rate
    weight_decay: float = 0.0  # L2 regularization
    optimizer: type = torch.optim.Adam  # optimizer class
    max_epochs: int = 100  # maximum training epochs
    seed: Optional[int] = None  # random seed (affects data shuffling, dropout, etc.)
    device: Optional[str] = None  # "cuda"/"cpu" (None = auto-detect)
    compile_model: bool = False  # use torch.compile
    temperature: float = 1.0  # score multiplier (higher = sharper softmax)

    # Early stopping
    patience: int = 10  # epochs without val improvement before stopping
    min_delta: float = 0.0  # minimum change to count as improvement
    monitor: str = "loss"  # metric to monitor: "loss", "ndcg@5", etc.
    restore_best_weights: bool = True  # restore best checkpoint after training
    verbose_early_stopping: bool = True  # print early stopping message
    eval_every: int = 1  # evaluate every N epochs

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"  # directory for checkpoints
    save_best: bool = True  # save checkpoint on improvement
    keep_checkpoint: bool = True  # keep checkpoint after training
    resume_from: Optional[str] = None  # path to resume from

    # Evaluation
    eval_on_val: bool = True  # evaluate on validation set
    eval_on_test: bool = True  # evaluate on test set after training
    metrics: List[tuple] = field(
        default_factory=lambda: [("ndcg", 5), ("ndcg", 10), ("score_spread", None)]
    )  # (metric, k) pairs

    # UI
    show_progress: bool = True  # show tqdm progress bars

    # MLflow logging
    mlflow_tracking: bool = False  # enable MLflow tracking
    mlflow_tracking_uri: Optional[str] = None  # MLflow server URI
    mlflow_experiment: Optional[str] = None  # experiment name
    mlflow_run_name: Optional[str] = None  # run name
    mlflow_system_metrics: bool = False  # log CPU/GPU metrics
    mlflow_tags: dict = field(default_factory=dict)  # additional tags
    mlflow_note: Optional[str] = None  # run description
