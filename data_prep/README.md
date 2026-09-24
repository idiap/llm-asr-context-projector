<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# 📁 Data preparation

The experiments use **DefinedAI**, a commercial corpus of scripted contact-center conversations between agents and customers, in five domains: Banking, Healthcare, Insurance, Retail and Telecommunications (Teleco).

## ⚠️ What is distributed, and what is not

**No data is distributed with this repository**: no audio, no transcripts, no keywords, no embeddings and no entity annotations.
DefinedAI is licensed commercially, so you need your own copy of the corpus from [defined.ai](https://defined.ai), under its terms of use.

What this folder provides is the code that turns that copy into everything the experiments read, so that the data behind the paper can be rebuilt exactly.

## 📦 Requirements

On top of the main environment ([`environment.yml`](../environment.yml)):

```bash
pip install lhotse sdialog==0.4.4
```

Keyword extraction runs Gemma 3 27B locally through [Ollama](https://ollama.com):

```bash
ollama serve &
ollama pull gemma3:27b
```

The Dialog2Flow sentence encoder is downloaded from Hugging Face the first time it is used.
Both the keyword extraction and the embeddings run much faster on a GPU.

## 🔄 The pipeline

One command builds a whole domain:

```bash
bash data_prep/prepare_definedai.sh banking /path/to/manifests /path/to/definedai
```

It expects one lhotse cut manifest per split, named `<domain>_{train,dev,test}_cuts*.jsonl.gz`, and runs these steps (each is also a standalone script with `--help`):

| Step | Script | Adds |
|------|--------|------|
| 1 | [`dump_jsonl.py`](dump_jsonl.py) | one line per utterance (`key`, `source`, `target`), cutting each utterance out of its recording as a 16 kHz wav |
| 2 | [`add_dialogue_context.py`](add_dialogue_context.py) | turns ordered by start time, with `context`, `context_audio`, `utt_idx` and `conversation_id` |
| 3 | [`extract_keywords.py`](extract_keywords.py) | `keywords` of every turn, extracted by Gemma 3 27B |
| 4 | [`add_keywords_history.py`](add_keywords_history.py) | `keywords_history`, the keywords of all previous turns |
| 5 | [`dump_d2f_embeddings.py`](dump_d2f_embeddings.py) | one Dialog2Flow embedding file per conversation |
| 6 | [`extract_entities_from_cuts.py`](extract_entities_from_cuts.py) | reference entities of the test set, for bias WER |

Each step is skipped when its output exists, so the command can be re-run after a failure (for instance if the Ollama server stops).
Set `DATA_ROOT` in [`scripts/common/config_defaults.sh`](../scripts/common/config_defaults.sh) to the output folder and the experiments find everything.

### Inputs

Each cut of the manifests must carry the transcript and the annotated entities in its supervision:

```json
{"id": "0e6c0103-..._banking_agent_00000-00004", "start": 0.0, "duration": 4.25, "channel": 0,
 "supervisions": [{"text": "HELLO THANKS FOR CALLING BANK OF AMERICA HOW MAY I HELP YOU",
                   "custom": {"role": "agent", "domain": "banking", "entities": "bank of america"}}],
 "recording": {"id": "0e6c0103-...", "sampling_rate": 16000, "sources": [{"type": "file", "channels": [0, 1], "source": "/path/to/0e6c0103-....wav"}]}}
```

Cut ids follow `<conversation_id>_<domain>_<role>_<start>-<end>`: the conversation id groups the turns, and the start time orders them.

## 📄 Output formats

Everything lands in `<data_root>/<domain>/`:

```
definedai_<domain>_{train,dev,test}.jsonl   utterances
d2f_embeddings/<conversation_id>/<conversation_id>_embedding.npy
definedai_<domain>_test_entities            reference entities
audio/{train,dev,test}/<key>.wav            segmented audio
intermediate/                               outputs of steps 1 to 3 and 6
```

### Utterances

One JSON object per line, grouped by conversation and ordered by turn:

| Field | Content |
|-------|---------|
| `key` | utterance id |
| `source` | path of the utterance audio |
| `target` | reference transcript |
| `context` | transcripts of all previous turns of the conversation, oldest first; `raw_ctx` runs use the last CTX of them |
| `context_audio` | audio paths of the same turns (not used by the experiments) |
| `utt_idx` | position of the turn in the conversation, from 0; row index in the embedding file |
| `conversation_id` | conversation id |
| `keywords` | keywords of this turn, upper-cased, in order of appearance |
| `keywords_history` | de-duplicated keywords of all previous turns; `raw_kw` and `kw_cp` runs insert them, comma-separated |

Both speakers count as turns: the context of a customer turn includes the agent turns before it.

### Embeddings

`<conversation_id>_embedding.npy` is a `float32` array of shape `(number of turns, 768)`.
Row `i` is the mean-pooled [dialog2flow-joint-bert-base](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base) embedding of the transcript of turn `utt_idx = i`.
For a turn with index `i`, the context projector reads rows `i - CTX` to `i - 1`, or rows `0` to `i - 1` for `CTX=ALL`.
Conversation ids are unique across splits, so the three splits share one folder.

### Entities

One line per test utterance, with its entities upper-cased and comma-separated (the line has only the id when there is none):

```
0e6c0103-..._banking_agent_00000-00004 BANK OF AMERICA
```

[`extract_entities_from_cuts.py`](extract_entities_from_cuts.py) only upper-cases the entities and strips their dots.
The bias WER scorer normalises them with Whisper's English text normalizer when it runs, as it does with the transcripts.

## 📊 Splits

The data split of Table 1 of the paper:

| Split | Domain | Utterances | Entities | Hours |
|-------|--------|-----------:|---------:|------:|
| Train | Banking | 26,496 | 9,852 | 54.5 |
| | Healthcare | 4,708 | 1,480 | 11.5 |
| | Insurance | 30,758 | 12,253 | 65.2 |
| | Retail | 10,981 | 2,938 | 27.8 |
| | Teleco | 18,889 | 6,392 | 50.9 |
| Dev | Banking | 1,447 | 527 | 2.9 |
| | Healthcare | 236 | 62 | 0.5 |
| | Insurance | 1,794 | 681 | 3.9 |
| | Retail | 623 | 111 | 1.5 |
| | Teleco | 1,130 | 382 | 3.1 |
| Test | Banking | 3,166 | 1,215 | 6.5 |
| | Healthcare | 488 | 154 | 1.1 |
| | Insurance | 3,670 | 1,432 | 7.7 |
| | Retail | 1,312 | 324 | 3.3 |
| | Teleco | 2,243 | 711 | 6.0 |
| **Total** | | **107,941** | **38,514** | **246.4** |

These are the counts of the lhotse manifests.
The pipeline keeps 107,928 of those utterances, the ones the runs behind the paper used: [`dump_jsonl.py`](dump_jsonl.py) keeps a single cut per duplicated id (7 cuts in Insurance and Teleco), and [`add_dialogue_context.py`](add_dialogue_context.py) skips the cuts whose id carries no start and end time (6 in the Healthcare training set).

## 🔁 Reproducibility notes

- **Keywords.** Step 3 queries an LLM, so a new extraction can differ slightly from ours even with the same model; everything downstream of it is deterministic.
- **Turn order.** Turns are ordered by start time; the few turns of a conversation that start at the same second keep the order of the manifest.
- **Embeddings.** Recomputing them on a different device changes the values only at the level of floating-point noise (around 1e-5).

## 📚 Citations

If you use the Dialog2Flow embeddings, please cite:

```bibtex
@inproceedings{burdisso-etal-2024-dialog2flow,
  title     = {{D}ialog2{F}low: Pre-training Soft-Contrastive Action-Driven Sentence Embeddings for Automatic Dialog Flow Extraction},
  author    = {Burdisso, Sergio and Madikeri, Srikanth and Motlicek, Petr},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages     = {5421--5440},
  year      = {2024},
  address   = {Miami, Florida, USA},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.emnlp-main.310/}
}
```

The keywords are extracted with Gemma 3 ([Gemma Team, 2025](https://goo.gle/Gemma3Report)) through [sdialog](https://github.com/idiap/sdialog).
