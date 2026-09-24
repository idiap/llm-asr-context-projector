#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Train the context projector on top of the frozen base model (encoder, speech projector and LLM frozen).
#   METHOD=cp    : Dialog2Flow embeddings of the last CTX turns, projected into context tokens (+cp)
#   METHOD=kw_cp : the same, plus the keywords of the previous turns as raw text (+kw+cp)
# Needs the base model of the same domain (exp/.../<domain>/base/epoch_5/model.pt).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source environment variables (TABLE_DIR selects the figure/table-specific config)
source "$SCRIPT_DIR/config_defaults.sh"

case "$METHOD" in
    cp|kw_cp) ;;
    *) echo "ERROR: 3.finetune_cp.sh trains METHOD=cp or kw_cp, not '$METHOD'" >&2; exit 1 ;;
esac

cd $RUN_DIR

output_dir=$RUN_EXP_DIR

[ ! -d $output_dir ] && mkdir -p $output_dir

context_args="++dataset_config.enable_context_projector=true \
++dataset_config.context_embeddings_path=$CONVERSATION_EMBEDDINGS_PATH"
if [ "$RAW_TEXT_CONTEXT" = "true" ]; then
    context_args="$context_args \
++dataset_config.enable_raw_text_context=true \
++dataset_config.context_format=$CONTEXT_FORMAT"
fi

hydra_args="hydra.run.dir=$output_dir \
++model_config.llm_name=$LLM_NAME \
++model_config.llm_path=$LLM_PATH \
++model_config.llm_dim=$LLM_DIM \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_path=$SPEECH_ENCODER_PATH \
++model_config.encoder_dim=$SPEECH_ENCODER_DIM \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$TRAIN_DATA_PATH \
++dataset_config.val_data_path=$VAL_DATA_PATH \
++dataset_config.input_type=raw \
$context_args \
++train_config.model_name=asr \
++train_config.num_epochs=$NUM_EPOCHS_CP \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.freeze_projector=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=$VALIDATION_INTERVAL \
++train_config.batch_size_training=$BATCH_SIZE \
++train_config.val_batch_size=$BATCH_SIZE \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.save_checkpoint_only_at_epoch_end=$SAVE_CKPT_ONLY_AT_EPOCH_END \
++log_config.log_file=$output_dir/train.log \
++metric=acc \
++ckpt_path=$BASE_PROJECTOR \
"

cmd="sbatch $SLURM_ACCOUNT_ARG --job-name ft-$DOMAIN-$RUN_NAME --partition=$SLURM_PARTITION --gpus=${SLURM_GPU_TYPE}:${SLURM_NUM_GPUS} --ntasks=1 --nodes=1 $SLURM_QOS_ARG"
cmd="${cmd} --cpus-per-task=20 --mem=60G --time=$SLURM_TIME_TRAIN --output=$output_dir/train.%j.out --error=$output_dir/train.%j.err"
# Add dependency if specified
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    cmd="${cmd} --dependency=afterok:$DEPENDENCY_JOB_ID"
fi

job_output=$($cmd --wrap="torchrun \
   --nnodes 1 \
   --nproc_per_node 1 \
   --master_port=$(get_free_port) \
   $RUN_DIR/finetune_asr.py \
   --config-path "conf" \
   --config-name "${PROMPT}.yaml" \
   ++train_config.enable_fsdp=false \
   ++train_config.enable_ddp=true \
   ++train_config.use_bf16=true \
   ${hydra_args}")

# Extract and return job ID if requested
if [ "$RETURN_JOB_ID" = "true" ]; then
    echo "$job_output" | grep -oP 'Submitted batch job \K\d+'
else
    echo "$job_output"
fi
