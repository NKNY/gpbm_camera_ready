"""Dataset loading and preprocessing for learning-to-rank."""

import os
import hashlib

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals(
    [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
)

import tensorflow_datasets as tfds
from scipy.stats import spearmanr
from tqdm.auto import tqdm


DATASET_FEATURES = {"mslr_web": 136, "yahoo_ltrc": 699}
DATASET_MAX_DOCS = {"mslr_web": 250, "yahoo_ltrc": 139}


def get_dataset_tfds_name(dataset: str) -> str:
    """Convert short dataset names to tensorflow_datasets names.

    Examples: "yahoo" -> "yahoo_ltrc", "mslr_1" -> "mslr_web/30k_fold1"
    """
    if dataset == "yahoo":
        return "yahoo_ltrc"
    if dataset.startswith("mslr_web/"):
        return dataset
    if dataset == "mslr" or dataset.startswith("mslr_"):
        fold = dataset.split("_")[1] if "_" in dataset else "1"
        return f"mslr_web/30k_fold{fold}"
    return dataset


def get_n_feat(dataset: str) -> int:
    """Get number of features for dataset (136 for MSLR, 699 for Yahoo)."""
    dataset = get_dataset_tfds_name(dataset)
    for key, val in DATASET_FEATURES.items():
        if key in dataset:
            return val
    raise ValueError(f"Unknown dataset: {dataset}")


def get_max_docs(dataset: str) -> int:
    """Get max documents per query for dataset (250 for MSLR, 139 for Yahoo)."""
    dataset = get_dataset_tfds_name(dataset)
    for key, val in DATASET_MAX_DOCS.items():
        if key in dataset:
            return val
    raise ValueError(f"Unknown dataset: {dataset}")


def normalize_query_features(feat):
    """Min-max normalize features across documents within a query."""
    if isinstance(feat, torch.Tensor):
        min_f = feat.min(dim=0)[0]
        max_f = feat.max(dim=0)[0]
        denom = (max_f - min_f).clamp(min=1e-6)
    else:
        min_f, max_f = feat.min(axis=0), feat.max(axis=0)
        denom = np.maximum(max_f - min_f, 1e-6)
    return (feat - min_f) / denom


def collate_fn(
    batch,
    n_docs_max,
    n_feat,
    relevance_transform,
    normalize_features=False,
    feature_indices=None,
):
    """Collate batch of queries into padded tensors.

    :param batch: List of query dicts with "float_features" and "label" keys.
    :param n_docs_max: Max documents per query (truncates longer queries).
    :param n_feat: Number of features per document.
    :param relevance_transform: Function to transform labels (e.g., identity or binarize).
    :param normalize_features: If True, min-max normalize features per query.
    :param feature_indices: If set, select only these feature columns.
    :return: (features, labels) tensors of shape (batch, n_docs_max, n_feat) and (batch, n_docs_max).
        Padding positions have label=-1.
    """
    features = torch.zeros(
        len(batch),
        n_docs_max,
        len(feature_indices) if feature_indices is not None else n_feat,
    )
    labels = torch.full((len(batch), n_docs_max), -1.0)
    for i, query in enumerate(batch):
        n = min(len(query["label"]), n_docs_max)
        feat = torch.tensor(query["float_features"][:n])
        if feature_indices is not None:
            feat = feat[:, feature_indices]
        if normalize_features:
            feat = normalize_query_features(feat)
        features[i, :n] = feat
        labels[i, :n] = relevance_transform(
            torch.tensor(query["label"][:n], dtype=torch.float32)
        )
    return features, labels


def filter_dataset(source, n_docs_min, n_docs_max, desc=None):
    """Filter dataset to queries with n_docs_min <= n_docs <= n_docs_max."""
    return [
        x
        for x in tqdm(source, desc=desc, leave=False)
        if n_docs_min <= len(x["label"]) <= n_docs_max
    ]


def get_cache_path(dataset, subset, n_docs_min, n_docs_max):
    """Get cache file path for filtered dataset."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ranking")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{dataset}_{subset}_{n_docs_min}_{n_docs_max}"
    return os.path.join(cache_dir, f"{hashlib.sha256(key.encode()).hexdigest()}.pt")


def load_or_create_dataset(dataset, subset, n_docs_min, n_docs_max, use_cache=True):
    """Load dataset from cache or create by filtering from tfds source."""
    cache_path = get_cache_path(dataset, subset, n_docs_min, n_docs_max)

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached {subset} dataset.")
        return torch.load(cache_path, weights_only=True)

    source = tfds.data_source(dataset)[subset]
    data = filter_dataset(source, n_docs_min, n_docs_max, desc=f"Loading {subset}")

    if use_cache:
        print(f"Saving cached {subset} dataset.")
        torch.save(data, cache_path)

    return data


class CachedDataset(torch.utils.data.Dataset):
    """Simple wrapper to make a list compatible with DataLoader."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _worker_init_fn(worker_id):
    """Initialize numpy random seed for DataLoader workers."""
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def compute_feature_correlations(dataset, n_docs_min, n_docs_max, use_cache=True):
    """Compute Spearman correlation between each feature and relevance labels.

    Returns array of shape (n_features,) with correlation values.
    """
    n_docs_max = n_docs_max or get_max_docs(dataset)

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ranking")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"corr_{dataset}_{n_docs_min}_{n_docs_max}"
    cache_path = os.path.join(
        cache_dir, f"{hashlib.sha256(cache_key.encode()).hexdigest()}.npy"
    )

    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    data = load_or_create_dataset(dataset, "train", n_docs_min, n_docs_max, use_cache)

    all_features, all_labels = [], []
    for item in data:
        feat = np.array(item["float_features"])
        labels = np.array(item["label"])
        feat = normalize_query_features(feat)
        all_features.append(feat)
        all_labels.append(labels)

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # Vectorized: correlate all features with labels at once
    corr_matrix = spearmanr(features, labels).statistic
    correlations = corr_matrix[:-1, -1]  # last column, all but last row

    correlations = np.nan_to_num(correlations)
    if use_cache:
        np.save(cache_path, correlations)
    return correlations


def select_top_features(
    dataset,
    n_docs_min,
    n_docs_max,
    top_k=None,
    top_pct=None,
    threshold=None,
    mode="best",
    use_cache=True,
):
    """Select feature indices by correlation with relevance labels.

    :param top_k: Number of features to select.
    :param top_pct: Percentage of features to select (0-100).
    :param threshold: Minimum |correlation| to include.
    :param mode: "best" (highest |corr|), "worst" (lowest), "middle" (around median).
    :return: (indices, correlations) - selected feature indices and their correlation values.
    """
    corr = compute_feature_correlations(dataset, n_docs_min, n_docs_max, use_cache)
    abs_corr = np.abs(corr)
    sorted_idx = np.argsort(abs_corr)
    n_feat = len(corr)

    if top_pct is not None:
        top_k = int(n_feat * top_pct / 100)

    if threshold is not None:
        indices = np.where(abs_corr >= threshold)[0]
    elif top_k is not None:
        if mode == "best":
            indices = sorted_idx[::-1][:top_k]
        elif mode == "worst":
            indices = sorted_idx[:top_k]
        elif mode == "middle":
            mid = n_feat // 2
            start = mid - top_k // 2
            indices = sorted_idx[start : start + top_k]
    else:
        indices = np.arange(n_feat)

    indices = np.sort(indices)
    return indices, corr[indices]


def create_dataloaders(config, n_feat=None, use_cache=True, feature_indices=None):
    """Create train/vali/test DataLoaders from config.

    :return: (loaders_dict, feat_dim) - dict mapping subset names to DataLoaders, and feature dimension.
    """
    dataset = get_dataset_tfds_name(config.dataset)
    subsets = ["train", "vali", "test"]

    n_docs_max = config.n_docs_max or get_max_docs(dataset)

    datasets = {
        subset: CachedDataset(
            load_or_create_dataset(
                dataset, subset, config.n_docs_min, n_docs_max, use_cache
            )
        )
        for subset in subsets
    }

    raw_feat_dim = n_feat or config.n_feat or get_n_feat(dataset)
    feat_dim = len(feature_indices) if feature_indices is not None else raw_feat_dim
    collate = lambda x: collate_fn(
        x,
        n_docs_max,
        raw_feat_dim,
        config.relevance_transform,
        config.normalize_features,
        feature_indices,
    )

    generator = torch.Generator()
    if config.seed is not None:
        generator.manual_seed(config.seed)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        worker_init_fn=_worker_init_fn if config.num_workers > 0 else None,
    )
    return {
        subset: torch.utils.data.DataLoader(
            datasets[subset],
            shuffle=(subset == "train"),
            generator=generator if subset == "train" else None,
            **loader_kwargs,
        )
        for subset in subsets
    }, feat_dim
