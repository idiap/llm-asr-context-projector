# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Step 4: build the keyword memory of every turn.
#
# "keywords_history" is the ordered, de-duplicated list of the keywords of all *previous* turns
# of the same conversation, so the first turn of every conversation has an empty history.
# The input must be ordered by conversation and turn, as written by add_dialogue_context.py.

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add keywords_history (keywords of previous turns) to every utterance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inp", required=True, help="input JSONL with 'keywords' (output of extract_keywords.py)")
    parser.add_argument("--out", required=True, help="output JSONL")
    args = parser.parse_args()

    conv_keywords = {}  # accumulated keywords per conversation
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            conv_id = entry["key"].split("_")[0]

            past_keywords = conv_keywords.get(conv_id, [])
            entry["keywords_history"] = list(past_keywords)

            seen = set(past_keywords)
            for kw in entry.get("keywords", []):
                if kw not in seen:
                    past_keywords.append(kw)
                    seen.add(kw)
            conv_keywords[conv_id] = past_keywords

            fout.write(json.dumps(entry) + "\n")
