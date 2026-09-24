<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Table 2: WER, BWER and entity F1 across the five domains

Four systems per domain:

| Row | Run | Description |
|-----|-----|-------------|
| base | `base` | speech projector only, no context |
| +kw | `raw_kw` | keywords of all previous turns in the prompt |
| +cp(·) | `cp:<CTX>` | context projector over the last CTX turns, on the frozen base model |
| +kw+cp(·) | `kw_cp:<CTX>` | the same, plus the keywords of all previous turns |

CTX is set automatically per domain to the value reported in the paper (10 everywhere, except Teleco with 1 and Insurance +kw+cp with 5), so the commands below need no extra flags.
Every run in this table is also part of [Figure 3](../figure3/), and the base and +kw rows of [Figure 2](../figure2/); runs already done there are skipped.

## ▶️ Running it

```bash
bash scripts/table2/run_all.sh                     # all five domains
DOMAINS=insurance bash scripts/table2/run_all.sh   # one domain
```

## 📊 Expected results

Once the jobs finish:

```bash
bash scripts/table2/results.sh
```

`results.sh` reads `WER.txt` and computes `BWER.txt` in every decoded run folder:

```
exp/definedai/<domain>/base/epoch_5/
exp/definedai/<domain>/raw_kw/epoch_5/
exp/definedai/<domain>/cp_ctx_<CTX>/epoch_2/
exp/definedai/<domain>/kw_cp_ctx_<CTX>/epoch_2/
```

WER (%), bias WER (BWER, %) and entity F1 of the runs behind the table, scored with the code in this repository (see [why they differ slightly from the paper](../../README.md#-expected-results)):

| Domain | System | CTX | WER | BWER | F1 |
|--------|--------|----:|----:|-----:|---:|
| Banking | base | - | 11.56 | 19.66 | 0.82 |
| | +kw | - | 11.51 | 18.71 | 0.84 |
| | +cp | 10 | 11.19 | 19.28 | 0.82 |
| | +kw+cp | 10 | 11.15 | 17.95 | 0.83 |
| Healthcare | base | - | 15.24 | 38.71 | 0.68 |
| | +kw | - | 16.84 | 33.06 | 0.76 |
| | +cp | 10 | 14.64 | 39.52 | 0.66 |
| | +kw+cp | 10 | 15.06 | 37.10 | 0.71 |
| Insurance | base | - | 9.98 | 21.58 | 0.81 |
| | +kw | - | 10.35 | 20.05 | 0.84 |
| | +cp | 10 | 9.87 | 21.41 | 0.82 |
| | +kw+cp | 5 | 9.90 | 20.62 | 0.83 |
| Retail | base | - | 16.79 | 45.26 | 0.67 |
| | +kw | - | 17.75 | 34.32 | 0.76 |
| | +cp | 10 | 15.56 | 44.63 | 0.68 |
| | +kw+cp | 10 | 15.77 | 42.11 | 0.71 |
| Teleco | base | - | 13.09 | 40.83 | 0.68 |
| | +kw | - | 13.70 | 34.06 | 0.74 |
| | +cp | 1 | 12.84 | 37.94 | 0.70 |
| | +kw+cp | 1 | 13.13 | 37.13 | 0.71 |

The last rows of the paper's table average the relative change over the base model across the five domains.
Computed from the numbers above, it is:

| System | WER | BWER | F1 |
|--------|----:|-----:|---:|
| +kw | +4.8% | -13.5% | +7.9% |
| +cp | -3.5% | -1.8% | +0.4% |
| +kw+cp | -2.3% | -6.7% | +3.6% |

A negative change is a gain for WER and BWER, and a positive one for F1; the paper prints gains as positive numbers instead.

## 📐 How the metrics are computed

- **WER** is computed at the end of decoding by [`wer_result.py`](../../wer_result.py), after normalising both the reference and the hypothesis with Whisper's English text normalizer.
- **BWER** counts the substitutions and deletions of reference words that belong to an annotated entity, divided by the number of entity words; insertions are counted in the unbiased part.
- **F1** is the entity-level F1: an entity counts as recognised when it appears exactly in the aligned hypothesis.

BWER and F1 come from [`bias_unbias_wer.py`](../../bias_unbias_wer.py), run by `results.sh` as:

```bash
python bias_unbias_wer.py --ref decode_output_gt --hyp decode_output_pred \
    --utt2ner <domain test entities> --entity-match-mode aligned --entity-metrics --normalize
```

`--normalize` applies Whisper's English text normalizer ([`whisper-normalizer`](https://pypi.org/project/whisper-normalizer/)) to the references, the hypotheses and each entity before scoring, as `wer_result.py` does for the WER.
It lower-cases the text, removes fillers and bracketed tags, expands contractions, writes numbers as digits and maps British spellings to American ones.
