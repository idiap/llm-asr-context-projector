<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# 📂 Experiment Scripts

One folder per result in the paper.
Each folder holds the runs of that figure or table plus a `run_all.sh` that submits them and a `results.sh` that prints their scores, and the scripts they all share live in [`common/`](common/).
If you only want one number from the paper, find its figure or table below and run that folder.

> [!NOTE]
> These scripts submit jobs to a SLURM cluster with `sbatch`.
> See [SLURM configuration](#-slurm-configuration) for how to adapt them to your cluster.

## 🗺️ Which folder reproduces which result

| Result | Folder | Experiment |
|--------|--------|------------|
| Figure 2 | [`figure2/`](figure2/) | Raw context appended to the prompt: keywords, previous turn, last 10 turns |
| Figure 3 | [`figure3/`](figure3/) | Context projection (+cp) and keywords with context projection (+kw+cp), context sizes 1, 5, 10 and all |
| Table 2 | [`table2/`](table2/) | WER, bias WER and entity F1 of base, +kw, +cp and +kw+cp in the five domains |

Table 1 describes the data rather than an experiment, so there is nothing to run for it; see [`data_prep/README.md`](../data_prep/README.md#-splits).

## 🚀 Quick start

1. **Point the scripts at your models and data.** Open [`common/config_defaults.sh`](common/config_defaults.sh) and edit the block marked *"Model and data paths"*:

   ```bash
   DEFAULT_SPEECH_ENCODER_PATH=/path/to/wavlm/WavLM-Large.pt
   DEFAULT_LLM_PATH=/path/to/meta-llama/Llama-3.2-3B-Instruct
   DEFAULT_DATA_ROOT=/path/to/definedai
   ```

   `DEFAULT_DATA_ROOT` is the folder written by [`data_prep/prepare_definedai.sh`](../data_prep/README.md).

2. **Tell the scripts about your cluster** (only if the defaults do not fit):

   ```bash
   export SLURM_ACCOUNT=your_account
   export SLURM_PARTITION=your_partition
   export SLURM_GPU_TYPE=a100
   ```

3. **Run a figure or table.** Every folder launches the same way:

   ```bash
   bash scripts/table2/run_all.sh                    # all five domains
   DOMAINS="banking retail" bash scripts/table2/run_all.sh
   ```

4. **Read the results** once the jobs finish:

   ```bash
   bash scripts/table2/results.sh
   ```

   See each folder's `README.md` for what it runs and the numbers to expect.

## 🔗 Pipeline

A run is identified by a domain, a method and a context size.
There are five methods, trained by two pairs of stages:

| Method | Paper name | Trains | Stages |
|--------|-----------|--------|--------|
| `base` | base, CTX_00 | speech projector, empty context | `1.finetune_base.sh` → `2.decode_base.sh` |
| `raw_kw` | +kw, Keywords | speech projector, keywords of previous turns in the prompt | `1.finetune_base.sh` → `2.decode_base.sh` |
| `raw_ctx` | CTX_01, CTX_10 | speech projector, transcripts of the last CTX turns in the prompt | `1.finetune_base.sh` → `2.decode_base.sh` |
| `cp` | +cp(·) | context projector over the last CTX turns, on the frozen base model | `3.finetune_cp.sh` → `4.decode_cp.sh` |
| `kw_cp` | +kw+cp(·) | the same, plus keywords of previous turns in the prompt | `3.finetune_cp.sh` → `4.decode_cp.sh` |

`run_all.sh` submits every run of a folder with SLURM dependencies, so a whole figure can be queued in one go:

```
1. finetune base       5 epochs, speech projector            exp/definedai/<domain>/base/epoch_5/
   ↓
2. decode base         WER.txt next to the checkpoint
   ↓
3. finetune cp         2 epochs, context projector           exp/definedai/<domain>/cp_ctx_<CTX>/epoch_2/
   ↓                   (starts from base/epoch_5/model.pt)
4. decode cp           WER.txt next to the checkpoint
```

The `raw_kw` and `raw_ctx` runs go through stages 1 and 2 on their own, into `exp/definedai/<domain>/raw_kw/` and `exp/definedai/<domain>/raw_ctx_<CTX>/`.

> [!TIP]
> Training is skipped when the run's checkpoint (`model.pt`) already exists, and decoding when its output does.
> Runs are shared across folders: the base model of a domain is trained once, and Table 2 reuses the runs of Figures 2 and 3.
> Delete a run folder to force it to run again.

## 📊 Reading the results

`results.sh` prints one line per run of the folder:

```
domain      method   ctx      WER    BWER     F1
----------- -------- ---- ------- ------- ------
banking     base     -      11.56   19.66   0.82
banking     raw_kw   -      11.51   18.71   0.84
...
```

WER comes from the `WER.txt` that every decoding writes next to the checkpoint it evaluated.
Bias WER and entity F1 are computed by [`bias_unbias_wer.py`](../bias_unbias_wer.py) the first time a run is listed (a few seconds on CPU) and saved as `BWER.txt` in the same folder, with the full breakdown (bias and unbiased WER, substitutions, insertions, deletions, entity precision and recall).
They need the reference entities written by `data_prep/prepare_definedai.sh`.

## 🧩 How the configuration is layered

```
scripts/table2/run_all.sh          sets TABLE_DIR, calls common/run_all.sh
  └── common/run_all.sh            one training and one decoding job per run
        └── common/<stage>.sh      DOMAIN, METHOD and CTX select the run
              └── common/config_defaults.sh   generic defaults + exports
                    └── table2/config.sh      only what is specific to this table
```

Values resolve highest-priority first:

1. variables you export before launching a script;
2. `DEFAULT_*` set by `<figure|table>/config.sh`;
3. the generic `DEFAULT_*` in `common/config_defaults.sh`.

So anything can be overridden without editing a file:

```bash
DOMAINS=insurance BATCH_SIZE=4 bash scripts/figure3/run_all.sh
```

A single stage can also be submitted by hand, which is handy to try a setting that no folder runs:

```bash
TABLE_DIR=scripts/figure3 DOMAIN=retail METHOD=kw_cp CTX=5 bash scripts/common/3.finetune_cp.sh
```

## 📋 Environment variables

### Runs

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAINS` | all five | Domains submitted by `run_all.sh` and listed by `results.sh` |
| `EXPERIMENTS` | per folder | Runs per domain, as `METHOD` or `METHOD:CTX` (the base model is always included) |
| `DOMAIN` | `banking` | Domain of a single stage: `banking`, `healthcare`, `insurance`, `retail` or `teleco` |
| `METHOD` | `base` | Method of a single stage: `base`, `raw_kw`, `raw_ctx`, `cp` or `kw_cp` |
| `CTX` | `10` | Previous turns used as context: `1`, `5`, `10` or `ALL` (ignored by `base` and `raw_kw`) |

### Models and data

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_ENCODER_PATH` | *(edit)* | WavLM-Large checkpoint |
| `SPEECH_ENCODER_DIM` | `1024` | Speech encoder output dimension |
| `LLM_PATH` | *(edit)* | LLM directory |
| `LLM_DIM` | `3072` | LLM embedding dimension (3072 for Llama 3.2 3B) |
| `DATA_ROOT` | *(edit)* | Output folder of `data_prep/prepare_definedai.sh` |
| `TRAIN_DATA_PATH` / `VAL_DATA_PATH` / `TEST_DATA_PATH` | under `$DATA_ROOT/<domain>/` | Utterance JSONL splits |
| `CONVERSATION_EMBEDDINGS_PATH` | `$DATA_ROOT/<domain>/d2f_embeddings` | Dialog2Flow embeddings, one file per conversation |
| `ENTITIES_PATH` | `$DATA_ROOT/<domain>/definedai_<domain>_test_entities` | Reference entities of the test set |

### Training

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_EPOCHS_BASE` | `5` | Epochs of the speech projector (`base`, `raw_kw`, `raw_ctx`) |
| `NUM_EPOCHS_CP` | `2` | Epochs of the context projector (`cp`, `kw_cp`) |
| `BATCH_SIZE` | per domain and method | 10, except Insurance (8 for `base`, 6 for `cp` and `kw_cp`); 4 for `raw_kw` and `raw_ctx` |
| `VALIDATION_INTERVAL` | per domain and method | Steps between validations: 1000, except Healthcare (200 for `base`, `cp` and `kw_cp`), Retail (500 for `cp` and `kw_cp`) and Teleco (800 for `cp` and `kw_cp`) |
| `SAVE_CKPT_ONLY_AT_EPOCH_END` | `true` | `false` checkpoints at every validation (`epoch_<E>_step_<S>/`) instead of at the end of each epoch; set `BASE_CKPT_FOLDER` or `CP_CKPT_FOLDER` to one of those folders to decode it |
| `PROMPT` | per method | Prompt config in [`../conf/`](../conf/) |

The per-domain batch sizes and validation intervals are the ones of the runs behind the paper.

### Experiments and checkpoints

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENT_NAME` | `definedai` | Top folder under `exp/` |
| `BASE_CKPT_FOLDER` | `epoch_5` | Speech projector checkpoint decoded, and loaded by the context projector stages |
| `CP_CKPT_FOLDER` | `epoch_2` | Context projector checkpoint decoded |
| `BASE_PROJECTOR` | `exp/definedai/<domain>/base/$BASE_CKPT_FOLDER/model.pt` | Base model the context projector is trained on |

### System

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) |
| `TOKENIZERS_PARALLELISM` | `false` | Silences tokenizer warnings |
| `OMP_NUM_THREADS` | `1` | OpenMP threads |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Set automatically; required for deterministic training |

### 🖥️ SLURM configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_ACCOUNT` | *(unset)* | Account name; the flag is omitted when empty |
| `SLURM_PARTITION` | `gpu` | Partition |
| `SLURM_QOS` | *(unset)* | QOS; the flag is omitted when empty |
| `SLURM_GPU_TYPE` | `h100` | GPU type for training |
| `SLURM_GPU_TYPE_DECODE` | `rtx3090\|v100` | GPU constraint for decoding |
| `SLURM_NUM_GPUS` | `1` | GPUs per job |
| `SLURM_TIME_TRAIN` | `07:00:00` | Training walltime |
| `SLURM_TIME_DECODE` | `05:00:00` | Decoding walltime |

## 🩺 Troubleshooting

### Monitoring jobs

`run_all.sh` prints the commands to track the jobs it just submitted:

```bash
squeue -u $USER                                                   # all your jobs
squeue -j JOB1,JOB2,JOB3                                          # these jobs
sacct -j JOB1,JOB2,JOB3 --format=JobID,JobName,State,Elapsed      # detail
scancel JOB1 JOB2 JOB3                                            # cancel
```

Job names are `ft-<domain>-<run>` for training and `dec-<domain>-<run>` for decoding.
If a job fails, the jobs that depend on it stay queued as `DependencyNeverSatisfied`.

### A run is skipped when it should run

A training job is skipped when its `model.pt` exists, and a decoding job when its `decode_output_pred` exists.
Remove the file or run folder that `run_all.sh` reports as existing, then submit again.

### The context projector cannot find its embeddings

A missing embedding file only raises a warning (`Context file ... does not exist`) and the turn gets no context tokens, so check the first lines of `train.log` after launching.
`CONVERSATION_EMBEDDINGS_PATH` must contain one `<conversation_id>/<conversation_id>_embedding.npy` per conversation, for every split.

### `results.sh` shows no BWER

Bias WER needs the reference entities at `ENTITIES_PATH`; without them only WER is printed.
If `BWER.txt` is missing next to a decoded run, `BWER.log` in the same folder holds the error.

### `RuntimeError` about deterministic algorithms

Training enables `torch.use_deterministic_algorithms(True)`, which requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
The scripts export it; if you call `finetune_asr.py` directly, export it yourself.

## 📝 Notes

- Speech projector training (`base`, `raw_kw`, `raw_ctx`) freezes the speech encoder and the LLM, and trains only the speech projector.
- Context projector training (`cp`, `kw_cp`) also freezes the speech projector of the base model, and trains only the context projector.
- The checkpoint of the last epoch is decoded, with beam search (beam size 4) and a batch size of 1.
