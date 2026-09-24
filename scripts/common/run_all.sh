#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Submit every training and decoding job of one figure or table, chained with SLURM dependencies.
#
# For every domain in $DOMAINS it trains and decodes the base model, then every run listed in the
# figure/table config (EXPERIMENTS, as METHOD or METHOD:CTX). Runs whose checkpoint or decoding
# output already exists are skipped, so runs shared between figures and tables are done once.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -z "$TABLE_DIR" ]; then
    echo "ERROR: TABLE_DIR is not set. Launch this through scripts/<figure|table>/run_all.sh." >&2
    exit 1
fi

# Value of a variable resolved by config_defaults.sh for one run, without exporting anything here
# (every stage script resolves its own configuration).
run_var() {  # run_var DOMAIN METHOD CTX VARIABLE
    ( export DOMAIN="$1" METHOD="$2" CTX="$3" RETURN_JOB_ID=true
      source "$SCRIPT_DIR/config_defaults.sh" > /dev/null && eval "echo \"\${$4}\"" )
}

# Value of DOMAINS or EXPERIMENTS for one domain, from the figure/table config.
table_var() {  # table_var DOMAIN VARIABLE
    ( export DOMAIN="$1"
      source "$TABLE_DIR/config.sh" && eval "echo \"\${$2:-\${DEFAULT_$2}}\"" )
}

submit() {  # submit STAGE_SCRIPT DOMAIN METHOD CTX DEPENDENCY_JOB_ID
    DOMAIN="$2" METHOD="$3" CTX="$4" DEPENDENCY_JOB_ID="$5" RETURN_JOB_ID=true bash "$SCRIPT_DIR/$1"
}

ALL_JOBS=""

# Train (unless the checkpoint exists) and decode (unless the output exists) one run.
# Sets TRAIN_JOB to the training job id, or to an empty string when training was skipped.
run_experiment() {  # run_experiment DOMAIN METHOD CTX DEPENDENCY_JOB_ID
    local domain=$1 method=$2 ctx=$3 dep=$4
    local train_script decode_script ckpt_var
    case "$method" in
        base|raw_kw|raw_ctx) train_script=1.finetune_base.sh; decode_script=2.decode_base.sh; ckpt_var=BASE_CKPT_FOLDER ;;
        cp|kw_cp)            train_script=3.finetune_cp.sh;   decode_script=4.decode_cp.sh;   ckpt_var=CP_CKPT_FOLDER ;;
        *) echo "ERROR: unknown METHOD '$method' in EXPERIMENTS" >&2; exit 1 ;;
    esac
    local run_dir run_name ckpt_dir
    run_dir=$(run_var "$domain" "$method" "$ctx" RUN_EXP_DIR) || exit 1
    run_name=$(run_var "$domain" "$method" "$ctx" RUN_NAME)
    ckpt_dir="$run_dir/$(run_var "$domain" "$method" "$ctx" $ckpt_var)"

    TRAIN_JOB=""
    if [ -f "$ckpt_dir/model.pt" ]; then
        echo "  [$run_name] ✓ Skipping training, checkpoint exists: $ckpt_dir/model.pt"
    else
        TRAIN_JOB=$(submit $train_script "$domain" "$method" "$ctx" "$dep")
        if [ -z "$TRAIN_JOB" ]; then
            echo "ERROR: failed to submit the training job of $domain/$run_name" >&2
            exit 1
        fi
        echo "  [$run_name] Training job $TRAIN_JOB${dep:+ (waits for $dep)}"
        ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$TRAIN_JOB"
    fi

    if [ -f "$ckpt_dir/decode_output_pred" ]; then
        echo "  [$run_name] ✓ Skipping decoding, output exists: $ckpt_dir/decode_output_pred"
    else
        local decode_job
        decode_job=$(submit $decode_script "$domain" "$method" "$ctx" "$TRAIN_JOB")
        if [ -z "$decode_job" ]; then
            echo "ERROR: failed to submit the decoding job of $domain/$run_name" >&2
            exit 1
        fi
        echo "  [$run_name] Decoding job $decode_job${TRAIN_JOB:+ (waits for $TRAIN_JOB)}"
        ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$decode_job"
    fi
}

DOMAINS=${DOMAINS:-$(table_var banking DOMAINS)}

echo "==============================================="
echo "Submitting $(basename "$TABLE_DIR")"
echo "  DOMAINS: $DOMAINS"
echo "==============================================="

for domain in $DOMAINS; do
    experiments=$(table_var "$domain" EXPERIMENTS)
    echo ""
    echo "[$domain] base $experiments"

    # The base model: CTX_00 in Figure 2, and the model every context projector is trained on
    run_experiment "$domain" base 10 ""
    base_job=$TRAIN_JOB

    for experiment in $experiments; do
        method=${experiment%%:*}
        ctx=10
        [ "$experiment" != "$method" ] && ctx=${experiment#*:}
        case "$method" in
            cp|kw_cp) run_experiment "$domain" "$method" "$ctx" "$base_job" ;;
            *)        run_experiment "$domain" "$method" "$ctx" "" ;;
        esac
    done
done

echo ""
echo "==============================================="
if [ -n "$ALL_JOBS" ]; then
    echo "Monitor jobs with:"
    echo "  squeue -j $ALL_JOBS"
    echo "Cancel all jobs with:"
    echo "  scancel ${ALL_JOBS//,/ }"
else
    echo "Nothing to submit: every run is already trained and decoded."
fi
echo "When the jobs finish, print the results with:"
echo "  bash scripts/$(basename "$TABLE_DIR")/results.sh"
echo "==============================================="
