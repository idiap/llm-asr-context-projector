#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Shared defaults for every experiment in this repository.
#
# You normally do not run this file directly. Each scripts/<figure|table>/ folder holds a small
# config.sh with the settings specific to one figure or table, and this file sources it first.
#
# Resolution order, from highest to lowest priority:
#   1. Variables you export before launching a script   (e.g. DOMAINS=banking bash scripts/table2/run_all.sh)
#   2. DEFAULT_* values set by scripts/<figure|table>/config.sh
#   3. The generic DEFAULT_* values below
#
# The first thing to edit for your own environment is the "Model and data paths" block.
#
# Every stage script works on one run, identified by three variables:
#   DOMAIN  banking | healthcare | insurance | retail | teleco
#   METHOD  base    : speech projector only, no context (the base model)
#           raw_kw  : speech projector trained with the keywords of previous turns in the prompt (+kw)
#           raw_ctx : speech projector trained with the last CTX previous transcripts in the prompt
#           cp      : context projector over the last CTX turns, on top of the frozen base model (+cp)
#           kw_cp   : like cp, plus the keywords of previous turns in the prompt (+kw+cp)
#   CTX     1 | 5 | 10 | ALL   (number of previous turns; ignored by base and raw_kw)

# ============================================
# FIGURE / TABLE-SPECIFIC CONFIG
# ============================================
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$CONFIG_DIR")")"
export REPO_ROOT

# TABLE_DIR is exported by scripts/<figure|table>/run_all.sh.
if [ -z "$TABLE_DIR" ]; then
    echo "ERROR: TABLE_DIR is not set. Launch experiments through scripts/<figure|table>/run_all.sh," >&2
    echo "       or export TABLE_DIR=scripts/table2 (for example) to call a stage in scripts/common/ directly." >&2
    # This file is always sourced from a script, so exit stops that script too.
    exit 1
fi
source "$TABLE_DIR/config.sh"

# ============================================
# DEFAULT VALUES
# ============================================

# --- Model and data paths (EDIT THESE FOR YOUR ENVIRONMENT) ---
DEFAULT_SPEECH_ENCODER_PATH=${DEFAULT_SPEECH_ENCODER_PATH:-/path/to/wavlm/WavLM-Large.pt}
DEFAULT_SPEECH_ENCODER_DIM=${DEFAULT_SPEECH_ENCODER_DIM:-1024}

DEFAULT_LLM_NAME=${DEFAULT_LLM_NAME:-llama}
DEFAULT_LLM_PATH=${DEFAULT_LLM_PATH:-/path/to/meta-llama/Llama-3.2-3B-Instruct}
DEFAULT_LLM_DIM=${DEFAULT_LLM_DIM:-3072}

# DefinedAI data, as written by data_prep/prepare_definedai.sh:
#   $DATA_ROOT/<domain>/definedai_<domain>_{train,dev,test}.jsonl   utterances with context and keywords
#   $DATA_ROOT/<domain>/definedai_<domain>_test_entities            reference entities for bias WER
#   $DATA_ROOT/<domain>/d2f_embeddings/                              Dialog2Flow embeddings per conversation
DEFAULT_DATA_ROOT=${DEFAULT_DATA_ROOT:-/path/to/definedai}

# --- Run selection ---
DOMAIN=${DOMAIN:-banking}
METHOD=${METHOD:-base}
CTX=${CTX:-10}

# --- Training (Section 2.5 of the paper) ---
DEFAULT_NUM_EPOCHS_BASE=${DEFAULT_NUM_EPOCHS_BASE:-5}   # speech projector (base, raw_kw, raw_ctx)
DEFAULT_NUM_EPOCHS_CP=${DEFAULT_NUM_EPOCHS_CP:-2}       # context projector (cp, kw_cp)
DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END=${DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END:-true}

# Per-domain batch size and validation interval (in steps) of the published runs.
# The prompts with raw dialogue history are longer, so those runs use a batch size of 4.
case "$DOMAIN" in
    banking)    PAPER_BS_BASE=10; PAPER_VAL_BASE=1000; PAPER_BS_CP=10; PAPER_VAL_CP=1000 ;;
    healthcare) PAPER_BS_BASE=10; PAPER_VAL_BASE=200;  PAPER_BS_CP=10; PAPER_VAL_CP=200 ;;
    insurance)  PAPER_BS_BASE=8;  PAPER_VAL_BASE=1000; PAPER_BS_CP=6;  PAPER_VAL_CP=1000 ;;
    retail)     PAPER_BS_BASE=10; PAPER_VAL_BASE=1000; PAPER_BS_CP=10; PAPER_VAL_CP=500 ;;
    teleco)     PAPER_BS_BASE=10; PAPER_VAL_BASE=1000; PAPER_BS_CP=10; PAPER_VAL_CP=800 ;;
    *)
        echo "ERROR: unknown DOMAIN '$DOMAIN' (expected banking, healthcare, insurance, retail or teleco)" >&2
        exit 1 ;;
esac
case "$METHOD" in
    base)         DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-$PAPER_BS_BASE}; DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-$PAPER_VAL_BASE} ;;
    raw_kw|raw_ctx) DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-4};          DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-1000} ;;
    cp|kw_cp)     DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-$PAPER_BS_CP};   DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-$PAPER_VAL_CP} ;;
    *)
        echo "ERROR: unknown METHOD '$METHOD' (expected base, raw_kw, raw_ctx, cp or kw_cp)" >&2
        exit 1 ;;
esac

# --- Prompt (conf/*.yaml), raw-text context and run name, derived from METHOD and CTX ---
case "$CTX" in
    1) CTX_TAG=01 ;;
    5) CTX_TAG=05 ;;
    10) CTX_TAG=10 ;;
    ALL) CTX_TAG=ALL ;;
    *)
        echo "ERROR: unknown CTX '$CTX' (expected 1, 5, 10 or ALL)" >&2
        exit 1 ;;
esac
case "$METHOD" in
    base)    RUN_NAME=base;              PROMPT_DEFAULT=prompt_10_CTX_Empty;          RAW_TEXT_CONTEXT=false; CONTEXT_FORMAT=keywords_history ;;
    raw_kw)  RUN_NAME=raw_kw;            PROMPT_DEFAULT=prompt_10_CTX_ALL;            RAW_TEXT_CONTEXT=true;  CONTEXT_FORMAT=keywords_history ;;
    raw_ctx) RUN_NAME=raw_ctx_${CTX};    PROMPT_DEFAULT=prompt_10_CTX_${CTX_TAG};     RAW_TEXT_CONTEXT=true;  CONTEXT_FORMAT=context ;;
    cp)      RUN_NAME=cp_ctx_${CTX};     PROMPT_DEFAULT=prompt_10_CTX_${CTX_TAG};     RAW_TEXT_CONTEXT=false; CONTEXT_FORMAT=keywords_history ;;
    kw_cp)   RUN_NAME=kw_cp_ctx_${CTX};  PROMPT_DEFAULT=prompt_10_CTX_${CTX_TAG}_HYB; RAW_TEXT_CONTEXT=true;  CONTEXT_FORMAT=keywords_history ;;
esac
DEFAULT_PROMPT=${DEFAULT_PROMPT:-$PROMPT_DEFAULT}

# --- Experiment naming (outputs go to exp/$EXPERIMENT_NAME/<domain>/<run>/) ---
DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-definedai}

# --- Checkpoints decoded (the last epoch of each stage) ---
DEFAULT_BASE_CKPT_FOLDER=${DEFAULT_BASE_CKPT_FOLDER:-"epoch_${DEFAULT_NUM_EPOCHS_BASE}"}
DEFAULT_CP_CKPT_FOLDER=${DEFAULT_CP_CKPT_FOLDER:-"epoch_${DEFAULT_NUM_EPOCHS_CP}"}

# --- System ---
DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-0}
DEFAULT_TOKENIZERS_PARALLELISM=${DEFAULT_TOKENIZERS_PARALLELISM:-false}
DEFAULT_OMP_NUM_THREADS=${DEFAULT_OMP_NUM_THREADS:-1}

# --- SLURM ---
DEFAULT_SLURM_ACCOUNT=${DEFAULT_SLURM_ACCOUNT:-""}
DEFAULT_SLURM_PARTITION=${DEFAULT_SLURM_PARTITION:-gpu}
DEFAULT_SLURM_QOS=${DEFAULT_SLURM_QOS:-""}
DEFAULT_SLURM_GPU_TYPE=${DEFAULT_SLURM_GPU_TYPE:-h100}
DEFAULT_SLURM_GPU_TYPE_DECODE=${DEFAULT_SLURM_GPU_TYPE_DECODE:-"rtx3090|v100"}
DEFAULT_SLURM_NUM_GPUS=${DEFAULT_SLURM_NUM_GPUS:-1}
DEFAULT_SLURM_TIME_TRAIN=${DEFAULT_SLURM_TIME_TRAIN:-07:00:00}
DEFAULT_SLURM_TIME_DECODE=${DEFAULT_SLURM_TIME_DECODE:-05:00:00}

# ============================================
# ENVIRONMENT VARIABLE ASSIGNMENT
# (only set if not already defined by the user)
# ============================================

export DOMAIN METHOD CTX RUN_NAME RAW_TEXT_CONTEXT CONTEXT_FORMAT

export SPEECH_ENCODER_PATH=${SPEECH_ENCODER_PATH:-$DEFAULT_SPEECH_ENCODER_PATH}
export SPEECH_ENCODER_DIM=${SPEECH_ENCODER_DIM:-$DEFAULT_SPEECH_ENCODER_DIM}
export LLM_NAME=${LLM_NAME:-$DEFAULT_LLM_NAME}
export LLM_PATH=${LLM_PATH:-$DEFAULT_LLM_PATH}
export LLM_DIM=${LLM_DIM:-$DEFAULT_LLM_DIM}

export DATA_ROOT=${DATA_ROOT:-$DEFAULT_DATA_ROOT}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$DATA_ROOT/$DOMAIN/definedai_${DOMAIN}_train.jsonl}
export VAL_DATA_PATH=${VAL_DATA_PATH:-$DATA_ROOT/$DOMAIN/definedai_${DOMAIN}_dev.jsonl}
export TEST_DATA_PATH=${TEST_DATA_PATH:-$DATA_ROOT/$DOMAIN/definedai_${DOMAIN}_test.jsonl}
export ENTITIES_PATH=${ENTITIES_PATH:-$DATA_ROOT/$DOMAIN/definedai_${DOMAIN}_test_entities}
export CONVERSATION_EMBEDDINGS_PATH=${CONVERSATION_EMBEDDINGS_PATH:-$DATA_ROOT/$DOMAIN/d2f_embeddings}

export NUM_EPOCHS_BASE=${NUM_EPOCHS_BASE:-$DEFAULT_NUM_EPOCHS_BASE}
export NUM_EPOCHS_CP=${NUM_EPOCHS_CP:-$DEFAULT_NUM_EPOCHS_CP}
export BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
export VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-$DEFAULT_VALIDATION_INTERVAL}
export SAVE_CKPT_ONLY_AT_EPOCH_END=${SAVE_CKPT_ONLY_AT_EPOCH_END:-$DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END}

export PROMPT=${PROMPT:-$DEFAULT_PROMPT}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}

export BASE_CKPT_FOLDER=${BASE_CKPT_FOLDER:-$DEFAULT_BASE_CKPT_FOLDER}
export CP_CKPT_FOLDER=${CP_CKPT_FOLDER:-$DEFAULT_CP_CKPT_FOLDER}

# Output folders: exp/$EXPERIMENT_NAME/<domain>/<run>/
export DOMAIN_EXP_DIR=$REPO_ROOT/exp/$EXPERIMENT_NAME/$DOMAIN
export RUN_EXP_DIR=$DOMAIN_EXP_DIR/$RUN_NAME
# The context projector is trained on top of this base model
export BASE_PROJECTOR=${BASE_PROJECTOR:-$DOMAIN_EXP_DIR/base/$BASE_CKPT_FOLDER/model.pt}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-$DEFAULT_TOKENIZERS_PARALLELISM}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$DEFAULT_OMP_NUM_THREADS}
# Required by torch.use_deterministic_algorithms(True), which the training pipeline enables.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export SLURM_ACCOUNT=${SLURM_ACCOUNT:-$DEFAULT_SLURM_ACCOUNT}
export SLURM_PARTITION=${SLURM_PARTITION:-$DEFAULT_SLURM_PARTITION}
export SLURM_QOS=${SLURM_QOS:-$DEFAULT_SLURM_QOS}
export SLURM_GPU_TYPE=${SLURM_GPU_TYPE:-$DEFAULT_SLURM_GPU_TYPE}
export SLURM_GPU_TYPE_DECODE=${SLURM_GPU_TYPE_DECODE:-$DEFAULT_SLURM_GPU_TYPE_DECODE}
export SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-$DEFAULT_SLURM_NUM_GPUS}
export SLURM_TIME_TRAIN=${SLURM_TIME_TRAIN:-$DEFAULT_SLURM_TIME_TRAIN}
export SLURM_TIME_DECODE=${SLURM_TIME_DECODE:-$DEFAULT_SLURM_TIME_DECODE}

# Optional sbatch flags, empty unless the corresponding variable is set.
SLURM_ACCOUNT_ARG=${SLURM_ACCOUNT:+--account $SLURM_ACCOUNT}
SLURM_QOS_ARG=${SLURM_QOS:+--qos=$SLURM_QOS}
export SLURM_ACCOUNT_ARG SLURM_QOS_ARG

get_free_port() {
    python - <<'PYEOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PYEOF
}

# ============================================
# SUMMARY
# ============================================
if [ "$RETURN_JOB_ID" != "true" ]; then
    echo "Configuration defaults loaded successfully!"
    echo "DOMAIN: $DOMAIN  METHOD: $METHOD  CTX: $CTX  RUN: $RUN_NAME"
    echo "SPEECH_ENCODER_PATH: $SPEECH_ENCODER_PATH"
    echo "LLM_PATH: $LLM_PATH"
    echo "TRAIN_DATA_PATH: $TRAIN_DATA_PATH"
    echo "PROMPT: $PROMPT"
    echo "RUN_EXP_DIR: $RUN_EXP_DIR"
    echo "BATCH_SIZE: $BATCH_SIZE  VALIDATION_INTERVAL: $VALIDATION_INTERVAL"
fi
