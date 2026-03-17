"""Click simulation for off-policy evaluation experiments."""

import argparse
import os

import torch
from dataclasses import dataclass, field, replace
from typing import List, Any
from tqdm.auto import tqdm

from ope.PL.propensities import (
    compute_pl_propensities_mpl as compute_pl_propensities,
    sample_rankings_batched,
)
from ope.data import (
    get_dataset_tfds_name,
    load_or_create_dataset,
    collate_fn,
    get_n_feat,
    get_max_docs,
    select_top_features,
)
from ope.training.ranking_model import RankingModel
from ope.training.checkpoint import load_checkpoint
from ope.ipm import (
    generate_item_alphas,
    generate_item_alphas_from_features,
    generate_item_alphas_from_model,
    compute_item_position_bias,
    simulate_clicks_ipm,
)
from ope.utils import set_seed, load_config_from_path, get_device


@dataclass
class SimulationConfig:
    """Configuration for click simulation."""

    # Required
    model_config: Any  # training Config used to train the ranking model
    output_dir: str  # base output directory

    # Output structure
    config_name: str = None  # subfolder name (None = save directly to output_dir)

    # Model
    model_path: str = None  # checkpoint path (None = latest in checkpoint_dir)
    config_overrides: dict = field(
        default_factory=dict
    )  # override model_config params (e.g., temperature)

    # Simulation
    n_samples: int = 100  # ranking samples per query
    n_repeats: int = 1  # independent repeat folders
    position_bias: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625]
    )  # examination prob per position
    subsets: List[str] = field(
        default_factory=lambda: ["vali", "test"]
    )  # dataset subsets
    propensities_only: bool = False  # skip ranking/click sampling
    seed: int = None  # random seed

    # Data filtering
    n_docs_max_drop: int = None  # drop queries with more docs
    n_docs_max_truncate: int = None  # truncate docs per query

    # Propensity computation
    batch_size: int = 96  # queries per batch
    propensity_n_quadrature: int = 1000  # Gauss-Legendre points
    propensity_percentile: float = 1e-10  # Gumbel quantile bounds
    propensity_dtype: str = None  # "float64" for precision (None = model dtype)

    # Item-specific Position Bias Model (IPM)
    click_model: str = "pbm"  # "pbm" (global) or "ipm" (item-specific)
    ipm_alpha_bound: float = (
        1.4  # alpha bound: pb deviation is pb ± (pb - pb^alpha), alpha in [1, bound]
    )
    ipm_alpha_seed: int = None  # seed for alpha generation (None = use main seed)
    ipm_alpha_from_features: bool = (
        True  # derive alphas from features (vs uniform random)
    )
    ipm_alpha_from_model: bool = False  # derive alphas from ranking model scores
    ipm_model_config: Any = (
        None  # separate training Config for IPM alpha model (None = use logging policy)
    )
    ipm_feature_frac: float = 0.1  # fraction of features to use for alpha generation


@dataclass
class SimulationContext:
    """Runtime context populated by load_model_and_config."""

    model: Any  # loaded RankingModel
    device: Any  # torch device
    dataset: str  # tfds dataset name
    n_docs_max: int  # max docs per query
    n_docs_truncate: int  # truncation limit
    feature_indices: Any  # selected feature indices
    K: int  # number of positions
    position_bias: Any  # position bias tensor
    base_dir: str  # output directory
    model_config: Any  # resolved model config


def simulate_clicks_pbm(rankings, labels, position_bias):
    """Simulate clicks using Position-Based Model.

    Click probability = P(click|d,k) = P(examine|k) * P(relevant|d) = pb[k] * rel[d]

    :param rankings: (n_samples, n_queries, K) - sampled rankings (document indices per position)
    :param labels: (n_queries, n_docs) - relevance labels in [0,1], -1 for padding
    :param position_bias: (K,) - examination probability at each position
    :return: clicks tensor of shape (n_samples, n_queries, K) - binary click indicators
    """
    n_samples, n_queries, K = rankings.shape

    # Get relevance at each position (mask padded docs once)
    labels_masked = labels.masked_fill(labels == -1, 0)  # (n_queries, n_docs)
    rel_at_pos = torch.gather(
        labels_masked.unsqueeze(0).expand(n_samples, -1, -1), 2, rankings
    )  # (n_samples, n_queries, K)

    # Click probability = position_bias * relevance
    position_bias = position_bias.to(rankings.device)[:K]  # (K,)
    clicks = torch.bernoulli(rel_at_pos * position_bias)  # (n_samples, n_queries, K)

    return clicks


def get_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in directory."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(
        checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    return os.path.join(checkpoint_dir, latest)


def _load_model_from_config(
    model_config, dataset, n_docs_max, device, checkpoint_path=None
):
    """Load a RankingModel from a training config.

    :param model_config: training Config dataclass
    :param dataset: TFDS dataset name (already resolved)
    :param n_docs_max: max docs per query (for feature selection)
    :param device: torch device
    :param checkpoint_path: explicit checkpoint path (None = latest in checkpoint_dir)
    :return: (model, feature_indices)
    """
    feature_indices = model_config.feature_indices
    if feature_indices is not None:
        print(f"Using {len(feature_indices)} explicit feature indices")
    elif model_config.feature_select_k or model_config.feature_select_pct:
        feature_indices, _ = select_top_features(
            dataset,
            model_config.n_docs_min,
            n_docs_max,
            top_k=model_config.feature_select_k,
            top_pct=model_config.feature_select_pct,
            mode=model_config.feature_select_mode,
        )
        print(
            f"Selected {len(feature_indices)} features ({model_config.feature_select_mode})"
        )

    n_feat = (
        len(feature_indices) if feature_indices is not None else get_n_feat(dataset)
    )
    model = RankingModel(
        n_feat_input=n_feat,
        n_feat=model_config.n_hidden,
        n_layers=model_config.n_layers,
        activation=model_config.activation,
        dropout=model_config.dropout,
        bias=model_config.bias,
        output_scale=model_config.output_scale,
        temperature=model_config.temperature,
        loss_fn=model_config.loss_fn,
        **model_config.loss_kwargs,
    ).to(device)
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(model_config.checkpoint_dir)
    load_checkpoint(checkpoint_path, model, None, device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, feature_indices


def load_model_and_config(sim_config) -> SimulationContext:
    """Load model and prepare configuration for simulation."""
    model_config = replace(sim_config.model_config, **sim_config.config_overrides)

    if sim_config.seed is not None:
        set_seed(sim_config.seed)

    base_dir = (
        os.path.join(sim_config.output_dir, sim_config.config_name)
        if sim_config.config_name
        else sim_config.output_dir
    )

    device = get_device(model_config.device)
    dataset = get_dataset_tfds_name(model_config.dataset)

    n_docs_max = (
        sim_config.n_docs_max_drop or model_config.n_docs_max or get_max_docs(dataset)
    )
    n_docs_truncate = sim_config.n_docs_max_truncate

    model, feature_indices = _load_model_from_config(
        model_config,
        dataset,
        n_docs_max,
        device,
        sim_config.model_path,
    )
    print(f"Output path: {base_dir}")

    position_bias = torch.tensor(sim_config.position_bias, dtype=torch.float32)
    K = len(position_bias)

    os.makedirs(base_dir, exist_ok=True)
    print(f"K: {K}, n_docs_max: {n_docs_max}, n_docs_truncate: {n_docs_truncate}")

    return SimulationContext(
        model,
        device,
        dataset,
        n_docs_max,
        n_docs_truncate,
        feature_indices,
        K,
        position_bias,
        base_dir,
        model_config,
    )


def compute_and_save_propensities(sim_config, ctx: SimulationContext):
    """Compute PL propensities for all subsets and save to disk.

    :return: Tuple of (subset_data, features_data) where subset_data maps subset names
        to (scores, labels, mask) tuples and features_data maps subset names to features tensors.
    """
    prop_dtype = (
        getattr(torch, sim_config.propensity_dtype)
        if sim_config.propensity_dtype
        else None
    )
    print(
        f"Propensity params: N={sim_config.propensity_n_quadrature}, percentile={sim_config.propensity_percentile}"
    )

    subset_data = {}
    features_data = {}
    for subset in sim_config.subsets:
        data = load_or_create_dataset(
            ctx.dataset, subset, ctx.model_config.n_docs_min, ctx.n_docs_max
        )
        raw_n_feat = get_n_feat(ctx.dataset)

        # Get selected features for model
        features, labels = collate_fn(
            data,
            ctx.n_docs_max,
            raw_n_feat,
            ctx.model_config.relevance_transform,
            ctx.model_config.normalize_features,
            ctx.feature_indices,
        )

        # Also get full features (no feature selection)
        features_full, _ = collate_fn(
            data,
            ctx.n_docs_max,
            raw_n_feat,
            ctx.model_config.relevance_transform,
            ctx.model_config.normalize_features,
            None,
        )

        features, labels = features.to(ctx.device), labels.to(ctx.device)
        features_full = features_full.to(ctx.device)

        if ctx.n_docs_truncate is not None and ctx.n_docs_truncate < features.shape[1]:
            features = features[:, : ctx.n_docs_truncate, :]
            features_full = features_full[:, : ctx.n_docs_truncate, :]
            labels = labels[:, : ctx.n_docs_truncate]

        mask = labels != -1

        with torch.no_grad():
            scores = ctx.model(features)
            scores_masked = scores.masked_fill(~mask, float("-inf"))
            original_dtype = scores_masked.dtype

            per_query_max = (
                scores_masked.masked_fill(~mask, float("-inf")).max(dim=1).values
            )
            per_query_min = (
                scores_masked.masked_fill(~mask, float("inf")).min(dim=1).values
            )
            print(
                f"{subset} max logit spread: {(per_query_max - per_query_min).max().item():.4f}"
            )

            scores_for_prop = (
                scores_masked.to(prop_dtype) if prop_dtype else scores_masked
            )

            propensities_list = []
            for i in tqdm(
                range(0, scores_for_prop.shape[0], sim_config.batch_size),
                desc=f"{subset} propensities",
            ):
                batch_scores = scores_for_prop[i : i + sim_config.batch_size]
                batch_prop = compute_pl_propensities(
                    batch_scores,
                    ctx.K,
                    N=sim_config.propensity_n_quadrature,
                    percentile=sim_config.propensity_percentile,
                )
                propensities_list.append(batch_prop.to(original_dtype).cpu())
            propensities = torch.cat(propensities_list, dim=0)

        assert (propensities >= 0).all() and (propensities <= 1).all(), (
            "Propensities must be in [0, 1]"
        )
        pos_sums = propensities.sum(dim=1)
        valid_sums = (pos_sums < 1e-5) | ((pos_sums > 1 - 1e-5) & (pos_sums < 1 + 1e-5))
        assert valid_sums.all(), (
            f"Position sums should be ~0 or ~1, got min={pos_sums.min():.4f}, max={pos_sums.max():.4f}"
        )
        print(
            f"{subset} propensity sums: min={pos_sums.min():.4f}, max={pos_sums.max():.4f}"
        )

        torch.save(
            propensities, os.path.join(ctx.base_dir, f"{subset}_propensities.pt")
        )
        torch.save(labels.cpu(), os.path.join(ctx.base_dir, f"{subset}_labels.pt"))
        torch.save(features.cpu(), os.path.join(ctx.base_dir, f"{subset}_features.pt"))
        torch.save(
            features_full.cpu(),
            os.path.join(ctx.base_dir, f"{subset}_features_full.pt"),
        )
        print(
            f"{subset} propensities: {propensities.shape}, features: {features.shape}, features_full: {features_full.shape}"
        )

        subset_data[subset] = (scores, labels, mask)
        features_data[subset] = features

    return subset_data, features_data


def sample_and_save_clicks(
    sim_config, subset_data, ctx: SimulationContext, features_data=None
):
    """Sample rankings from PL model and simulate clicks for each repeat.

    :param sim_config: Simulation configuration
    :param subset_data: Dict mapping subset -> (scores, labels, mask)
    :param ctx: Simulation context
    :param features_data: Dict mapping subset -> features tensor (required for IPM)
    """
    use_ipm = sim_config.click_model == "ipm"

    if use_ipm:
        # Load separate IPM model if configured
        ipm_model, ipm_feature_indices = (None, None)
        if sim_config.ipm_alpha_from_model and sim_config.ipm_model_config is not None:
            ipm_model, ipm_feature_indices = _load_model_from_config(
                sim_config.ipm_model_config,
                ctx.dataset,
                ctx.n_docs_max,
                ctx.device,
            )

        # Pre-generate alphas for each subset (consistent across repeats)
        subset_alphas = {}
        subset_item_pb = {}
        for subset in sim_config.subsets:
            _, labels, _ = subset_data[subset]
            features = features_data[subset]
            n_queries, n_docs = labels.shape

            alpha_seed = sim_config.ipm_alpha_seed or sim_config.seed
            if alpha_seed is not None:
                alpha_seed = alpha_seed + hash(subset) % (2**31)

            if sim_config.ipm_alpha_from_model:
                if ipm_model is not None:
                    # Use separate IPM model with its own feature subset
                    features_full = torch.load(
                        os.path.join(ctx.base_dir, f"{subset}_features_full.pt"),
                        weights_only=True,
                    ).to(ctx.device)
                    ipm_features = (
                        features_full[:, :, ipm_feature_indices]
                        if ipm_feature_indices is not None
                        else features_full
                    )
                    alphas = generate_item_alphas_from_model(ipm_model, ipm_features)
                else:
                    # Fall back to logging policy model
                    alphas = generate_item_alphas_from_model(ctx.model, features)
                alphas = torch.tensor(alphas, dtype=labels.dtype, device=labels.device)
            elif sim_config.ipm_alpha_from_features:
                # Derive alphas from features via random nonlinear mapping
                alphas = generate_item_alphas_from_features(
                    features.cpu(), sim_config.ipm_feature_frac, alpha_seed
                )
                alphas = torch.tensor(alphas, dtype=labels.dtype, device=labels.device)
            else:
                # Uniform random alphas
                alphas = generate_item_alphas(n_queries * n_docs, alpha_seed)
                alphas = torch.tensor(
                    alphas, dtype=labels.dtype, device=labels.device
                ).reshape(n_queries, n_docs)

            item_pb = compute_item_position_bias(
                alphas.cpu().numpy().flatten(),
                ctx.K,
                sim_config.ipm_alpha_bound,
                ctx.position_bias.numpy(),
            )
            item_pb = torch.tensor(
                item_pb, dtype=labels.dtype, device=labels.device
            ).reshape(n_queries, n_docs, ctx.K)

            subset_alphas[subset] = alphas
            subset_item_pb[subset] = item_pb

            # Save alphas once at base level
            torch.save(alphas.cpu(), os.path.join(ctx.base_dir, f"{subset}_alphas.pt"))

    for repeat in tqdm(range(sim_config.n_repeats), desc="Repeats"):
        repeat_dir = os.path.join(ctx.base_dir, f"repeat_{repeat}")
        os.makedirs(repeat_dir, exist_ok=True)

        for subset in sim_config.subsets:
            scores, labels, mask = subset_data[subset]

            with torch.no_grad():
                rankings = sample_rankings_batched(
                    scores, sim_config.n_samples, mask, ctx.K, sim_config.batch_size
                )

                if use_ipm:
                    clicks = simulate_clicks_ipm(
                        rankings, labels, subset_item_pb[subset]
                    )
                else:
                    clicks = simulate_clicks_pbm(rankings, labels, ctx.position_bias)

            torch.save(
                rankings.cpu(), os.path.join(repeat_dir, f"{subset}_rankings.pt")
            )
            torch.save(clicks.cpu(), os.path.join(repeat_dir, f"{subset}_clicks.pt"))
            print(
                f"Repeat {repeat}, {subset}: rankings {rankings.shape}, clicks {clicks.shape}"
            )


def run_simulation(sim_config: SimulationConfig):
    """Run full simulation pipeline: compute propensities and sample clicks."""
    ctx = load_model_and_config(sim_config)
    subset_data, features_data = compute_and_save_propensities(sim_config, ctx)
    if not sim_config.propensities_only:
        sample_and_save_clicks(sim_config, subset_data, ctx, features_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_config", help="Path to simulation config file")
    args = parser.parse_args()

    config = load_config_from_path(args.sim_config)
    run_simulation(config)
