#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
"""
Step 6: extract the reference entities used by the bias-WER scorer.

Reads lhotse cut manifests whose supervisions carry the annotated entities in
supervisions[0].custom["entities"] (comma-separated) and writes one utt2ner file per manifest.

Output format (one line per cut):
    <cut_id> <ENTITY 1>, <ENTITY 2>, ...

The entities are only upper-cased and stripped of dots here: bias_unbias_wer.py --normalize applies
Whisper's English text normalizer to them, as it does to the references and the hypotheses.

Usage:
    python data_prep/extract_entities_from_cuts.py \
        --output-dir /path/to/entities \
        banking_test_cuts.jsonl.gz insurance_test_cuts.jsonl.gz ...

The output file for each input is named after the input stem
(everything before the first '.' in the filename) with '_entities' appended, e.g.:
    banking_test_cuts_fbank.jsonl.gz  ->  <output_dir>/banking_test_cuts_fbank_entities
"""

import argparse
import gzip
import json
import os
import sys


def get_output_name(input_path: str) -> str:
    """Strip all extensions (e.g. '.fbank.jsonl.gz') and append '_entities'."""
    basename = os.path.basename(input_path)
    # Remove everything from the first '.' onwards
    stem = basename.split(".")[0]
    return stem + "_entities"


def print_stats(total_cuts: int, cuts_with_entities: int, all_entities: list) -> None:
    """Print statistics about the extracted entities."""
    total_entities = len(all_entities)
    avg_per_cut = total_entities / total_cuts if total_cuts else 0.0
    char_lengths = [len(e) for e in all_entities if e]
    word_lengths = [len(e.split()) for e in all_entities if e]
    avg_char = sum(char_lengths) / len(char_lengths) if char_lengths else 0.0
    avg_words = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0

    print(f"  --- Entity Stats ---")
    print(f"  Total cuts processed       : {total_cuts}")
    print(f"  Cuts with entities         : {cuts_with_entities}")
    print(f"  Total individual entities  : {total_entities}")
    print(f"  Avg entities per cut       : {avg_per_cut:.2f}")
    print(f"  Avg char length per entity : {avg_char:.2f}")
    print(f"  Avg word length per entity : {avg_words:.2f}")


def process_file(input_path: str, output_path: str) -> tuple:
    """Convert one fbank.jsonl.gz file to a utt2ner_norm file.

    Returns (total_cuts, cuts_with_entities, all_individual_entities).
    """
    total_cuts = 0
    cuts_with_entities = 0
    all_entities = []  # one entry per individual comma-separated entity
    with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            cut = json.loads(line)
            cut_id = cut["id"]
            supervisions = cut.get("supervisions", [])
            entities = ""
            if supervisions:
                custom = supervisions[0].get("custom", {})
                entities = custom.get("entities", "") or ""

            # Normalize: strip whitespace, remove dots, uppercase
            entities_norm = entities.strip().replace(".", "").upper()

            # Collect individual entities (comma-separated) for stats
            cut_items = [item.strip() for item in entities_norm.split(",") if item.strip()]
            if cut_items:
                cuts_with_entities += 1
            all_entities.extend(cut_items)

            fout.write(f"{cut_id} {entities_norm}\n")
            total_cuts += 1
    return total_cuts, cuts_with_entities, all_entities

def main():
    parser = argparse.ArgumentParser(
        description="Extract reference entities from lhotse cut manifests."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="FILE",
        help="One or more lhotse cut manifests (*.jsonl.gz).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where output files will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for input_path in args.input_files:
        if not os.path.isfile(input_path):
            print(f"[WARNING] File not found, skipping: {input_path}", file=sys.stderr)
            continue

        stem = get_output_name(input_path)
        output_path = os.path.join(args.output_dir, stem)

        print(f"Processing: {input_path}  ->  {output_path}")
        total_cuts, cuts_with_entities, all_entities = process_file(input_path, output_path)
        print(f"  Wrote {total_cuts} lines.")
        print_stats(total_cuts, cuts_with_entities, all_entities)


if __name__ == "__main__":
    main()
