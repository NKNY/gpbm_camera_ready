#!/usr/bin/env python3
"""Generate experiment configs from a metaconfig YAML file."""

import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("metaconfig", type=str, help="Path to metaconfig YAML file")
args = parser.parse_args()

with open(args.metaconfig) as f:
    cfg = yaml.safe_load(f)


def interpolate(val, ctx):
    if isinstance(val, str):
        return val.format(**ctx)
    if isinstance(val, list):
        return [interpolate(v, ctx) for v in val]
    return val


def temp_to_name(t):
    if t < 0:
        return f"temp_neg{abs(t)}".replace(".", "_")
    return f"temp{t}".replace(".", "_")


def get_pb_exprs(pb_type, bias_name, val, bias_type, K):
    zigzag = f"[1, -1] * {K // 2}" + (f" + [1]" if K % 2 else "")
    zigzag_scaled = (
        f"[z * math.sin(math.pi * (k+1) / ({K}+1)) for k, z in enumerate({zigzag})]"
    )
    if pb_type == "pbdcg":
        true_pb_expr = f"[math.log2(i+1)**-1 for i in range(1, {K + 1})]"
        if bias_type == "pow":
            pb_expr = f"[(math.log2(i+1)**-1)**{val} for i in range(1, {K + 1})]"
            base_eps_expr = f"[abs((math.log2(i+1)**-1) - (math.log2(i+1)**-1)**{val}) for i in range(1, {K + 1})]"
        elif bias_type == "plus":
            pb_expr = f"[min(math.log2(i+1)**-1 + {val}, 1) for i in range(1, {K + 1})]"
            base_eps_expr = f"[{val}]*{K}"
        elif bias_type == "zigzag":
            pb_expr = f"[min(1, math.log2(i+1)**-1 + s*{val}) for i, s in zip(range(1, {K + 1}), {zigzag_scaled})]"
            base_eps_expr = f"[abs(s*{val}) for s in {zigzag_scaled}]"
        else:
            pb_expr = f"[(math.log2(i+1)**-1) - {val} for i in range(1, {K + 1})]"
            base_eps_expr = f"[{val}]*{K}"
    else:
        print("Unsupported bias type!")
    return pb_expr, base_eps_expr, true_pb_expr


folds = cfg.get("folds", [1])
base_dataset = cfg["dataset"]

for fold in folds:
    # For MSLR, fold changes the dataset; for Yahoo, fold is just a repeat with different seed
    if "mslr" in base_dataset:
        dataset = f"mslr_{fold}"
        fold_seed = 0
    else:
        dataset = base_dataset
        fold_seed = fold - 1

    ctx = {"exp_name": cfg["exp_name"], "dataset": dataset, "fold": fold}
    c = {k: interpolate(v, ctx) for k, v in cfg.items()}

    base_config_path = os.path.expanduser(c["base_config_path"])
    base_exp_path = c["base_exp_path"]
    base_training_path = c["base_training_path"]
    mlflow_tracking_uri = c.get("mlflow_tracking_uri")
    mlflow_experiment = c.get("mlflow_experiment")
    mlflow_tracking = mlflow_tracking_uri is not None and mlflow_experiment is not None
    logging_model = c["logging_model"]
    target_model = c["target_model"]
    nsampled_values = c["nsampled_values"]
    pb_types = c["pb_types"]
    ndocs_values = c["ndocs_values"]
    K = c.get("K", 10)  # ranking size, defaults to 10
    logging_temps = c["logging_temps"]
    target_temps = c["target_temps"]
    bias_configs = [tuple(x) for x in c["bias_configs"]]
    eps_multipliers = [tuple(x) for x in c["eps_multipliers"]]
    single_repeat = c["single_repeat"]
    n_F_repeats = c.get(
        "n_F_repeats", None
    )  # n_repeats for F optimization and eval (separate from simulation)
    max_outer = c["max_outer"]
    patience = c["patience"]
    chunk_size = c.get("chunk_size", None)
    bootstrap_chunk = c.get("bootstrap_chunk", None)
    subsample_ratio = c.get("subsample_ratio", None)
    bias_subsample_ratio = c.get("bias_subsample_ratio", None)
    use_slope = c["use_slope"]
    use_blue = c["use_blue"]
    use_opera = c.get("use_opera", False)
    blue_configs = c.get("blue_configs")
    opera_configs = c.get("opera_configs")
    max_clip = c.get("max_clip", None)
    prop_clamp_min = c.get("prop_clamp_min", None)
    relevance_transform = c.get("relevance_transform", None)
    n_samples = c["n_samples"]
    n_repeats = c["n_repeats"]
    training_configs = [tuple(x) for x in c["training_configs"]]
    subsets = c.get("subsets", ["test"])

    # IPM settings
    click_model = c.get("click_model", "pbm")  # "pbm" or "ipm"
    ipm_alpha_bound = c.get("ipm_alpha_bound", 1.4)
    ipm_alpha_seed = c.get("ipm_alpha_seed", None)
    ipm_alpha_from_features = c.get("ipm_alpha_from_features", False)
    ipm_feature_frac = c.get("ipm_feature_frac", 0.1)
    ipm_alpha_from_model = c.get("ipm_alpha_from_model", False)
    ipm_model = c.get(
        "ipm_model", None
    )  # training config name for separate IPM alpha model

    print(f"Generating configs for fold {fold} (dataset={dataset}, seed={fold_seed})")

    # Path component for fold
    fold_part = f"fold{fold}"

    # Generate training configs
    training_dir = os.path.join(base_config_path, "training", dataset, fold_part)
    os.makedirs(training_dir, exist_ok=True)

    for tc in training_configs:
        name, ds = tc[0], tc[1]
        if len(tc) == 3 and isinstance(tc[2], list):
            # feature_indices mode: [name, dataset, [indices]]
            feature_line = f"    feature_indices={tc[2]},"
        else:
            # feature_select mode: [name, dataset, pct, mode]
            feature_line = (
                f'    feature_select_pct={tc[2]},\n    feature_select_mode="{tc[3]}",'
            )
        training_config = f'''from ope.training.config import Config

config = Config(
    dataset="{ds}",
{feature_line}
    output_scale=10.,
    n_samples_grad=1000,
    batch_size=256,
    lr=1e-2,
    max_epochs=20,
    seed={fold_seed},
    compile_model=True,
    patience=5,
    checkpoint_dir="{base_training_path}/{fold_part}/{name}",
    keep_checkpoint=True,
    mlflow_tracking={mlflow_tracking},
    mlflow_tracking_uri={f'"{mlflow_tracking_uri}"' if mlflow_tracking_uri else None},
    mlflow_experiment={f'"{mlflow_experiment}"' if mlflow_experiment else None},
    mlflow_note="dataset={dataset}, fold={fold}, model={name}",
)
'''
        with open(os.path.join(training_dir, f"{name}.py"), "w") as f:
            f.write(training_config)

    # Generate logging configs
    for model, model_temps, propensities_only in [
        (logging_model, logging_temps, False),
        (target_model, target_temps, True),
    ]:
        training_config_relpath = f"training/{dataset}/{fold_part}/{model}.py"
        for pb_type in pb_types:
            true_pb_expr = (
                f"[math.log2(i+1)**-1 for i in range(1, {K + 1})]"
                if pb_type == "pbdcg"
                else f"[math.log2(i+1)**-1 for i in range(1, {K + 1})]"
            )

            for ndocs in ndocs_values:
                for t in model_temps:
                    temp_name = temp_to_name(t)

                    log_dir = os.path.join(
                        base_config_path,
                        "logging",
                        dataset,
                        fold_part,
                        model,
                        pb_type,
                        f"ndocs{ndocs}",
                        temp_name,
                    )
                    os.makedirs(log_dir, exist_ok=True)

                    rel_override = (
                        f', "relevance_transform": {relevance_transform}'
                        if relevance_transform
                        else ""
                    )

                    # IPM config lines
                    click_model_line = (
                        f'click_model="{click_model}",' if click_model != "pbm" else ""
                    )
                    ipm_alpha_line = (
                        f"ipm_alpha_bound={ipm_alpha_bound},"
                        if click_model == "ipm"
                        else ""
                    )
                    ipm_seed_line = (
                        f"ipm_alpha_seed={ipm_alpha_seed},"
                        if click_model == "ipm" and ipm_alpha_seed is not None
                        else ""
                    )
                    ipm_from_feat_line = (
                        f"ipm_alpha_from_features={ipm_alpha_from_features},"
                        if click_model == "ipm"
                        else ""
                    )
                    ipm_feat_frac_line = (
                        f"ipm_feature_frac={ipm_feature_frac},"
                        if click_model == "ipm" and ipm_alpha_from_features
                        else ""
                    )
                    ipm_from_model_line = (
                        f"ipm_alpha_from_model={ipm_alpha_from_model},"
                        if click_model == "ipm" and ipm_alpha_from_model
                        else ""
                    )
                    ipm_model_config_line = ""
                    if click_model == "ipm" and ipm_alpha_from_model and ipm_model:
                        ipm_training_relpath = (
                            f"training/{dataset}/{fold_part}/{ipm_model}.py"
                        )
                        ipm_model_config_line = f'ipm_model_config=load_training_config("{ipm_training_relpath}"),'

                    log_config = f'''import math
from ope.get_clicks_propensities import SimulationConfig
model_config = load_training_config("{training_config_relpath}")

config = SimulationConfig(
    model_config=model_config,
    output_dir="{base_exp_path}/{model}/{pb_type}/ndocs{ndocs}/{temp_name}",
    config_name=None,
    n_samples={n_samples},
    n_repeats={n_repeats},
    n_docs_max_truncate={ndocs},
    position_bias={true_pb_expr},
    subsets={subsets},
    seed=0,
    propensity_n_quadrature=1000,
    propensity_percentile=1e-8,
    propensity_dtype="float64",
    propensities_only={propensities_only},
    config_overrides={{"temperature": {t}{rel_override}}},
    {click_model_line}
    {ipm_alpha_line}
    {ipm_seed_line}
    {ipm_from_feat_line}
    {ipm_feat_frac_line}
    {ipm_from_model_line}
    {ipm_model_config_line}
)
'''
                    with open(os.path.join(log_dir, "config.py"), "w") as f:
                        f.write(log_config)

    # Generate F and eval configs
    for nsampled in nsampled_values:
        for pb_type in pb_types:
            for ndocs in ndocs_values:
                ndocs_part = f"ndocs{ndocs}"
                for t1 in logging_temps:
                    temp1 = temp_to_name(t1)
                    for t2 in target_temps:
                        temp2 = temp_to_name(t2)

                        for bias_name, val, bias_type in bias_configs:
                            pb_expr, base_eps_expr, true_pb_expr = get_pb_exprs(
                                pb_type, bias_name, val, bias_type, K
                            )

                            for eps_entry in eps_multipliers:
                                eps_name = eps_entry[0]
                                if len(eps_entry) == 2:
                                    eps_mult_expr = str(eps_entry[1])
                                else:
                                    a, b = eps_entry[1], eps_entry[2]
                                    eps_mult_expr = f"[{a} + ({b} - {a}) * i / ({K} - 1) for i in range({K})][j]"
                                base_or_floor = (
                                    f"{base_eps_expr}"
                                    if any(
                                        eval(
                                            base_eps_expr,
                                            {
                                                "__builtins__": {},
                                                "math": __import__("math"),
                                                "range": range,
                                                "abs": abs,
                                                "enumerate": enumerate,
                                            },
                                        )
                                    )
                                    else f"[0.005]*{K}"
                                )
                                if len(eps_entry) == 2:
                                    eps_expr = f"[x * {eps_mult_expr} for x in {base_or_floor}]"
                                else:
                                    eps_expr = f"[x * ({eps_mult_expr}) for j, x in enumerate({base_or_floor})]"

                                logging_dir = f"{base_exp_path}/{logging_model}/{pb_type}/{ndocs_part}/{temp1}"
                                target_prop_path = f"{base_exp_path}/{target_model}/{pb_type}/{ndocs_part}/{temp2}"
                                labels_path = f"{base_exp_path}/{logging_model}/{pb_type}/{ndocs_part}/{temp1}"

                                f_dir = os.path.join(
                                    base_config_path,
                                    "F",
                                    dataset,
                                    fold_part,
                                    logging_model,
                                    target_model,
                                    f"nsampled{nsampled}",
                                    pb_type,
                                    ndocs_part,
                                    temp1,
                                    temp2,
                                    bias_name,
                                    eps_name,
                                )
                                os.makedirs(f_dir, exist_ok=True)

                                f_output_dir = f"{base_exp_path}/{logging_model}/{target_model}/nsampled{nsampled}/{pb_type}/{ndocs_part}/{temp1}/{temp2}/{bias_name}/{eps_name}"

                                f_position_bias = pb_expr
                                f_n_repeats_line = (
                                    f"n_repeats={n_F_repeats}," if n_F_repeats else ""
                                )
                                f_chunk_size_line = (
                                    f"chunk_size={chunk_size}," if chunk_size else ""
                                )
                                f_bootstrap_chunk_line = (
                                    f"bootstrap_chunk={bootstrap_chunk},"
                                    if bootstrap_chunk
                                    else ""
                                )
                                f_subsample_ratio_line = (
                                    f"subsample_ratio={subsample_ratio},"
                                    if subsample_ratio
                                    else ""
                                )
                                f_bias_subsample_ratio_line = (
                                    f"bias_subsample_ratio={bias_subsample_ratio},"
                                    if bias_subsample_ratio
                                    else ""
                                )
                                n_bootstrap_var = c.get("n_bootstrap_var")
                                f_n_bootstrap_var_line = (
                                    f"n_bootstrap_var={n_bootstrap_var},"
                                    if n_bootstrap_var is not None
                                    else ""
                                )
                                n_bootstrap_bias = c.get("n_bootstrap_bias")
                                f_n_bootstrap_bias_line = (
                                    f"n_bootstrap_bias={n_bootstrap_bias},"
                                    if n_bootstrap_bias is not None
                                    else ""
                                )

                                f_config = f'''import math
from ope.optimize_F import FOptimizationConfig

config = FOptimizationConfig(
    logging_dir="{logging_dir}",
    target_propensities_path="{target_prop_path}",
    output_dir="{f_output_dir}",
    position_bias={f_position_bias},
    eps_k={eps_expr},
    subsets={subsets},
    n_samples_per_query={nsampled},
    seed=0,
    max_outer={max_outer},
    patience={patience},
    single_repeat={single_repeat},
    {f_n_repeats_line}
    {f_chunk_size_line}
    {f_bootstrap_chunk_line}
    {f_subsample_ratio_line}
    {f_bias_subsample_ratio_line}
    {f_n_bootstrap_var_line}
    {f_n_bootstrap_bias_line}
)
'''
                                with open(os.path.join(f_dir, "config.py"), "w") as f:
                                    f.write(f_config)

                                eval_dir = os.path.join(
                                    base_config_path,
                                    "eval",
                                    dataset,
                                    fold_part,
                                    logging_model,
                                    target_model,
                                    f"nsampled{nsampled}",
                                    pb_type,
                                    ndocs_part,
                                    temp1,
                                    temp2,
                                    bias_name,
                                    eps_name,
                                )
                                os.makedirs(eval_dir, exist_ok=True)

                                eval_position_bias = pb_expr
                                eval_n_repeats_line = (
                                    f"n_repeats={n_F_repeats}," if n_F_repeats else ""
                                )
                                eval_max_clip_line = (
                                    f"max_clip={max_clip},"
                                    if max_clip is not None
                                    else ""
                                )
                                eval_prop_clamp_line = (
                                    f"prop_clamp_min={prop_clamp_min},"
                                    if prop_clamp_min is not None
                                    else ""
                                )
                                eval_ipm_alpha_bound_line = (
                                    f"ipm_alpha_bound={ipm_alpha_bound},"
                                    if click_model == "ipm"
                                    else ""
                                )

                                eval_config = f'''import math
from ope.evaluate import EvalConfig

config = EvalConfig(
    logging_dir="{logging_dir}",
    target_propensities_path="{target_prop_path}",
    labels_path="{labels_path}",
    F_source="{f_output_dir}",
    output_path="{f_output_dir}/eval.json",
    position_bias={eval_position_bias},
    true_position_bias={true_pb_expr},
    subsets={subsets},
    n_samples_per_query={nsampled},
    seed=0,
    window_sizes=list(range({K})),
    use_single_F={single_repeat},
    use_slope={use_slope},
    use_blue={use_blue},
    use_opera={use_opera},
    blue_configs={blue_configs},
    opera_configs={opera_configs},
    {eval_n_repeats_line}
    {eval_max_clip_line}
    {eval_prop_clamp_line}
    {eval_ipm_alpha_bound_line}
)
'''
                                with open(
                                    os.path.join(eval_dir, "config.py"), "w"
                                ) as f:
                                    f.write(eval_config)

print("Done generating configs")
