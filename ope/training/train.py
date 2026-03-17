"""Training loop and evaluation for ranking models."""

import argparse
import os
from dataclasses import fields

import ast

import torch
from tqdm.auto import tqdm

from ope.training.ranking_model import RankingModel
from ope.training.config import Config
from ope.training.metrics import METRICS, BestMetricsTracker
from ope.data import create_dataloaders, get_dataset_tfds_name, select_top_features
from ope.training.early_stopping import EarlyStopping
from ope.training.checkpoint import (
    generate_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
)
from ope.training.mlflow_utils import create_logger
from ope.utils import set_seed, get_eval_metrics, load_config_from_path, get_device


def evaluate(model, dataloader, device, metrics, loss_fn=None):
    """Evaluate model on dataloader, computing loss and specified metrics.

    :param model: RankingModel instance.
    :param dataloader: DataLoader yielding (features, labels) batches.
    :param device: torch device.
    :param metrics: List of (metric_name, k) tuples, e.g. [("ndcg", 5), ("ndcg", 10)].
    :param loss_fn: Loss function (default: model.loss_fn).
    :return: Dict with "loss" and metric keys like "ndcg@5".
    """
    model.eval()
    loss_fn = loss_fn or model.loss_fn
    total_loss, n = 0.0, 0
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            scores = model(x)
            total_loss += loss_fn(scores, labels).item() * x.size(0)
            all_scores.append(scores)
            all_labels.append(labels)
            n += x.size(0)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    results = {"loss": total_loss / n}
    for name, k in metrics:
        key = f"{name}@{k}" if k else name
        results[key] = (
            METRICS[name](scores, labels, k) if k else METRICS[name](scores, labels)
        )
    return results


def train(config: Config, feature_indices=None, tags=None):
    """Train a ranking model with early stopping and checkpointing.

    :param config: Training configuration dataclass.
    :param feature_indices: Optional feature subset to use (overrides config.feature_indices).
    :param tags: Optional dict of tags for MLflow logging.
    :return: Tuple of (model, results_dict, checkpoint_path).
    """
    checkpoint_path = generate_checkpoint_path(config)

    if config.seed is not None:
        set_seed(config.seed)

    logger = create_logger(config, tags)
    device = get_device(config.device)

    # Feature selection
    if feature_indices is None:
        feature_indices = config.feature_indices
    if feature_indices is None and (
        config.feature_select_k or config.feature_select_pct
    ):
        dataset = get_dataset_tfds_name(config.dataset)
        feature_indices, _ = select_top_features(
            dataset,
            config.n_docs_min,
            config.n_docs_max,
            top_k=config.feature_select_k,
            top_pct=config.feature_select_pct,
            mode=config.feature_select_mode,
        )
        print(
            f"Selected {len(feature_indices)} features ({config.feature_select_mode})"
        )

    # Create dataloaders
    dataloaders, n_feat = create_dataloaders(
        config, config.n_feat, feature_indices=feature_indices
    )

    # Create model
    model = RankingModel(
        n_feat_input=n_feat,
        n_feat=config.n_hidden,
        n_layers=config.n_layers,
        activation=config.activation,
        dropout=config.dropout,
        bias=config.bias,
        output_scale=config.output_scale,
        temperature=config.temperature,
        loss_fn=config.loss_fn,
        n_samples=config.n_samples_grad,
        n_samples_metric=config.n_samples_train,
        **config.loss_kwargs,
    ).to(device)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Loss functions for val/test
    val_loss_fn = config.loss_fn(
        n_samples=config.n_samples_val,
        n_samples_metric=config.n_samples_val,
        **config.loss_kwargs,
    ).to(device)
    test_loss_fn = config.loss_fn(
        n_samples=config.n_samples_test,
        n_samples_metric=config.n_samples_test,
        **config.loss_kwargs,
    ).to(device)

    optimizer = config.optimizer(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    print(f"Checkpoint: {checkpoint_path}")
    logger.log_param("checkpoint_path", checkpoint_path)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        monitor=config.monitor,
        verbose=config.verbose_early_stopping,
    )

    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_epoch, best_val = load_checkpoint(
            config.resume_from, model, optimizer, device
        )
        early_stopping.best_val = best_val
        print(f"Resumed from epoch {start_epoch}")

    eval_metrics = get_eval_metrics(config)
    best_tracker = BestMetricsTracker()
    epoch_iter = range(start_epoch, config.max_epochs)
    if config.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training")

    for epoch in epoch_iter:
        # Training
        model.train()
        train_loss, n = 0.0, 0
        batch_iter = dataloaders["train"]
        if config.show_progress:
            batch_iter = tqdm(batch_iter, desc=f"Epoch {epoch + 1}", leave=False)

        for x, labels in batch_iter:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)
            if config.show_progress:
                batch_iter.set_postfix(loss=f"{train_loss / n:.4f}")

        log = f"Epoch {epoch + 1}: train_loss={train_loss / n:.4f}"
        logger.log_metrics({"train_loss": train_loss / n}, epoch)

        # Validation
        if config.eval_on_val and (epoch + 1) % config.eval_every == 0:
            val_metrics = evaluate(
                model, dataloaders["vali"], device, eval_metrics, val_loss_fn
            )
            best_tracker.update(val_metrics, epoch)
            log += ", " + ", ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            logger.log_metrics(val_metrics, epoch, prefix="val_")
            logger.log_metrics(best_tracker.get_best(), epoch, prefix="best_")

            if config.show_progress:
                epoch_iter.set_postfix(
                    **{k: f"{v:.4f}" for k, v in val_metrics.items()}
                )

            improved = not early_stopping(epoch, val_metrics)
            if improved and config.save_best:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    early_stopping.best_val,
                    config,
                )

            if early_stopping.should_stop:
                break

        if not config.show_progress:
            print(log)

    # Restore best weights
    if (
        config.restore_best_weights
        and config.save_best
        and os.path.exists(checkpoint_path)
    ):
        print("Restoring best checkpoint")
        load_checkpoint(checkpoint_path, model, None, device)

    results = {
        "best_val": early_stopping.best_val,
        "best_epoch": early_stopping.best_epoch,
        "monitor": config.monitor,
    }

    # Test evaluation
    if config.eval_on_test:
        results["test"] = evaluate(
            model, dataloaders["test"], device, config.metrics, test_loss_fn
        )
        print("Test: " + ", ".join(f"{k}={v:.4f}" for k, v in results["test"].items()))
        logger.log_metrics(results["test"], early_stopping.best_epoch, prefix="test_")

    logger.log_metrics(
        {"best_epoch": early_stopping.best_epoch, "best_val": early_stopping.best_val},
        early_stopping.best_epoch,
    )
    logger.close()

    if not config.keep_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        checkpoint_path = None

    return model, results, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file (e.g., configs/train.py)")
    parser.add_argument(
        "--override", "-o", nargs="*", default=[], help="Overrides as key=value"
    )
    parser.add_argument(
        "--tag", "-t", nargs="*", default=[], help="MLflow tags as key=value"
    )
    parser.add_argument("--note", "-n", help="MLflow run note/description")
    args = parser.parse_args()

    # Load config from file path
    config = load_config_from_path(args.config)

    # Apply overrides
    field_types = {f.name: f.type for f in fields(config)}
    for override in args.override:
        key, val = override.split("=", 1)
        if key in field_types:
            t = field_types[key]
            if t == bool:
                val = val.lower() in ("true", "1", "yes")
            elif t in (int, float, str) or (
                hasattr(t, "__origin__") and t.__origin__ is type(None)
            ):
                val = ast.literal_eval(val)
            setattr(config, key, val)

    # Parse tags
    tags = dict(t.split("=", 1) for t in args.tag) if args.tag else {}
    if args.note:
        tags["mlflow.note.content"] = args.note

    train(config, tags=tags)


if __name__ == "__main__":
    main()
