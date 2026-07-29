#!/bin/bash

# Usage: ./run_pipeline.sh --exp_name <name> --dataset <dataset> [options]
# Example: ./run_pipeline.sh --exp_name 260115_test --dataset yahoo --folds "1 2 3 4 5" --steps 1234

set -e

# Defaults
EXP_NAME="test"
DATASET="yahoo"
FOLDS="1"
LOGGING_MODEL="2_3_worst_features"
TARGET_MODEL="all_features"
PB_LIST="pbdcg pbdcg"
NDOCS_LIST="10 50"
LOGGING_TEMPS="-0.2 0.2 0.6 1"
TARGET_TEMPS="-0.2 0.2 0.6 1"
NSAMPLED_LIST="100 50 10"
BIAS_PATTERNS="bias_minus0_05 bias_minus0_1 bias_pow1_4 bias_pow0_6"
EPS_PATTERNS="eps_times0_5 eps_times1 eps_times2"
CONFIG_BASE=""
STEPS="1234"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --folds) FOLDS="$2"; shift 2 ;;
        --logging_model) LOGGING_MODEL="$2"; shift 2 ;;
        --target_model) TARGET_MODEL="$2"; shift 2 ;;
        --pb) PB_LIST="$2"; shift 2 ;;
        --ndocs) NDOCS_LIST="$2"; shift 2 ;;
        --logging_temp) LOGGING_TEMPS="$2"; shift 2 ;;
        --target_temp) TARGET_TEMPS="$2"; shift 2 ;;
        --nsampled) NSAMPLED_LIST="$2"; shift 2 ;;
        --bias_pattern) BIAS_PATTERNS="$2"; shift 2 ;;
        --eps_pattern) EPS_PATTERNS="$2"; shift 2 ;;
        --config_base) CONFIG_BASE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory and cd to parent
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"

temp_to_name() {
    local t=$1
    if [[ "$t" == -* ]]; then
        echo "temp_neg${t#-}" | tr '.' '_'
    else
        echo "temp${t}" | tr '.' '_'
    fi
}

for FOLD in $FOLDS; do
    # For MSLR, fold changes dataset name; for Yahoo, dataset stays the same
    if [[ "$DATASET" == mslr* ]]; then
        FOLD_DATASET="mslr_${FOLD}"
    else
        FOLD_DATASET="$DATASET"
    fi
    
    FOLD_CONFIG_BASE=${CONFIG_BASE:-/tmp/configs/runs}/${EXP_NAME}
    FOLD_PART="fold${FOLD}"
    
    echo "========================================"
    echo "=== Processing FOLD $FOLD (dataset=$FOLD_DATASET) ==="
    echo "========================================"

    if [[ "$STEPS" == *"0"* ]]; then
        echo ""
        echo "=== Step 0: Train models ==="
        for config_path in "${FOLD_CONFIG_BASE}/training/${FOLD_DATASET}/${FOLD_PART}"/*.py; do
            if [ -f "$config_path" ]; then
                echo "--- Training: $(basename $config_path .py) ---"
                echo "python -m ope.training.train $config_path"
                python -m ope.training.train "$config_path"
            fi
        done
    fi

for PB in $PB_LIST; do
for NDOCS in $NDOCS_LIST; do
    echo ""
    echo "=== pb=$PB, ndocs=$NDOCS ==="
    
    if [[ "$STEPS" == *"1"* ]]; then
        echo ""
        echo "=== Step 1: Generate clicks and propensities for LOGGING model ($LOGGING_MODEL) ==="
        for t in $LOGGING_TEMPS; do
            temp_name=$(temp_to_name $t)
            config_path="${FOLD_CONFIG_BASE}/logging/${FOLD_DATASET}/${FOLD_PART}/${LOGGING_MODEL}/${PB}/ndocs${NDOCS}/${temp_name}/config.py"
            if [ -f "$config_path" ]; then
                echo "--- Generating logging: temp=$t ---"
                echo "python -m ope.get_clicks_propensities $config_path"
                python -m ope.get_clicks_propensities "$config_path"
            else
                echo "Config not found: $config_path"
            fi
        done
    fi
    
    if [[ "$STEPS" == *"2"* ]]; then
        echo ""
        echo "=== Step 2: Generate propensities for TARGET model ($TARGET_MODEL) ==="
        for t in $TARGET_TEMPS; do
            temp_name=$(temp_to_name $t)
            config_path="${FOLD_CONFIG_BASE}/logging/${FOLD_DATASET}/${FOLD_PART}/${TARGET_MODEL}/${PB}/ndocs${NDOCS}/${temp_name}/config.py"
            if [ -f "$config_path" ]; then
                echo "--- Generating target: temp=$t ---"
                echo "python -m ope.get_clicks_propensities $config_path"
                python -m ope.get_clicks_propensities "$config_path"
            else
                echo "Config not found: $config_path"
            fi
        done
    fi
    
    if [[ "$STEPS" == *"3"* ]] || [[ "$STEPS" == *"4"* ]]; then
        echo ""
        echo "=== Step 3/4: Optimize F and Evaluate ==="
        for nsampled in $NSAMPLED_LIST; do
            for logging_t in $LOGGING_TEMPS; do
                logging_name=$(temp_to_name $logging_t)
                for target_t in $TARGET_TEMPS; do
                    target_name=$(temp_to_name $target_t)
                    for bias in $BIAS_PATTERNS; do
                        for eps in $EPS_PATTERNS; do
                            f_config="${FOLD_CONFIG_BASE}/F/${FOLD_DATASET}/${FOLD_PART}/${LOGGING_MODEL}/${TARGET_MODEL}/nsampled${nsampled}/${PB}/ndocs${NDOCS}/${logging_name}/${target_name}/${bias}/${eps}/config.py"
                            eval_config="${FOLD_CONFIG_BASE}/eval/${FOLD_DATASET}/${FOLD_PART}/${LOGGING_MODEL}/${TARGET_MODEL}/nsampled${nsampled}/${PB}/ndocs${NDOCS}/${logging_name}/${target_name}/${bias}/${eps}/config.py"
                            
                            if [[ "$STEPS" == *"3"* ]] && [ -f "$f_config" ]; then
                                echo "--- Optimizing F: nsampled=$nsampled, logging=$logging_t -> target=$target_t, bias=$bias, eps=$eps ---"
                                echo "python -m ope.optimize_F $f_config"
                                python -m ope.optimize_F "$f_config"
                            fi
                            
                            if [[ "$STEPS" == *"4"* ]] && [ -f "$eval_config" ]; then
                                echo "--- Evaluating ---"
                                echo "python -m ope.evaluate $eval_config"
                                python -m ope.evaluate "$eval_config"
                            fi
                        done
                    done
                done
            done
        done
    fi
done
done
done

echo ""
echo "=== Pipeline complete ==="
