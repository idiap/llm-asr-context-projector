#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Build the data of one DefinedAI domain from its lhotse cut manifests.
#
# usage: bash data_prep/prepare_definedai.sh <domain> <manifest_dir> <data_root>
#
#   <domain>        banking | healthcare | insurance | retail | teleco
#   <manifest_dir>  folder with <domain>_{train,dev,test}_cuts*.jsonl.gz
#   <data_root>     output root; point DATA_ROOT (scripts/common/config_defaults.sh) at it
#
# Writes, for the experiment scripts:
#   <data_root>/<domain>/definedai_<domain>_{train,dev,test}.jsonl   utterances with context and keywords
#   <data_root>/<domain>/definedai_<domain>_test_entities            reference entities for bias WER
#   <data_root>/<domain>/d2f_embeddings/                              Dialog2Flow embeddings per conversation
# plus the segmented audio (<data_root>/<domain>/audio/) and intermediate files.
#
# Every step is skipped when its output already exists, so the script can be re-run after a failure.
# Keyword extraction (step 3) needs a running Ollama server with gemma3:27b; see data_prep/README.md.
set -euo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

DOMAIN=$1
MANIFEST_DIR=$2
DATA_ROOT=$3
PREP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=$DATA_ROOT/$DOMAIN
WORK=$OUT/intermediate
mkdir -p "$WORK"

manifest() {  # manifest SPLIT
    local matches=("$MANIFEST_DIR/${DOMAIN}_$1_cuts"*.jsonl.gz)
    if [ ${#matches[@]} -ne 1 ] || [ ! -f "${matches[0]}" ]; then
        echo "ERROR: expected exactly one $MANIFEST_DIR/${DOMAIN}_$1_cuts*.jsonl.gz" >&2
        exit 1
    fi
    echo "${matches[0]}"
}

for split in train dev test; do
    final=$OUT/definedai_${DOMAIN}_${split}.jsonl
    if [ -f "$final" ]; then
        echo "[$split] ✓ $final exists"
        continue
    fi

    # 1. lhotse cuts -> utterance JSONL, segmenting the audio
    if [ ! -f "$WORK/1.utterances_$split.jsonl" ]; then
        split_manifest=$(manifest $split)
        python "$PREP_DIR/dump_jsonl.py" --inp "$split_manifest" --out "$WORK/1.utterances_$split.jsonl" \
            --save-audio "$OUT/audio/$split"
    fi

    # 2. dialogue history, utt_idx and conversation_id
    [ -f "$WORK/2.context_$split.jsonl" ] || \
        python "$PREP_DIR/add_dialogue_context.py" --inp "$WORK/1.utterances_$split.jsonl" --out "$WORK/2.context_$split.jsonl"

    # 3. keywords of every turn, extracted with Gemma 3 27B
    if [ ! -f "$WORK/3.keywords_$split.jsonl" ]; then
        python "$PREP_DIR/extract_keywords.py" --inp "$WORK/2.context_$split.jsonl" --out "$WORK/3.keywords_$split.jsonl.partial"
        mv "$WORK/3.keywords_$split.jsonl.partial" "$WORK/3.keywords_$split.jsonl"
    fi

    # 4. keyword memory of the previous turns
    python "$PREP_DIR/add_keywords_history.py" --inp "$WORK/3.keywords_$split.jsonl" --out "$final"
    echo "[$split] wrote $final"
done

# 5. Dialog2Flow embeddings of every turn, one file per conversation
if [ -d "$OUT/d2f_embeddings" ]; then
    echo "[embeddings] ✓ $OUT/d2f_embeddings exists"
else
    python "$PREP_DIR/dump_d2f_embeddings.py" --output_root "$OUT/d2f_embeddings.partial" \
        --inputs "$OUT/definedai_${DOMAIN}_train.jsonl" "$OUT/definedai_${DOMAIN}_dev.jsonl" "$OUT/definedai_${DOMAIN}_test.jsonl"
    mv "$OUT/d2f_embeddings.partial" "$OUT/d2f_embeddings"
fi

# 6. reference entities of the test set, for bias WER
entities=$OUT/definedai_${DOMAIN}_test_entities
if [ -f "$entities" ]; then
    echo "[entities] ✓ $entities exists"
else
    test_manifest=$(manifest test)
    python "$PREP_DIR/extract_entities_from_cuts.py" --output-dir "$WORK" "$test_manifest"
    stem=$(basename "$test_manifest"); stem=${stem%%.*}
    cp "$WORK/${stem}_entities" "$entities"
    echo "[entities] wrote $entities"
fi

echo "Done: $OUT"
