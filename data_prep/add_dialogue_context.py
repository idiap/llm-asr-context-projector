# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Step 2: order the utterances of every conversation by start time and add the dialogue history.
#
# Utterance keys follow the DefinedAI convention <conversation_id>_<domain>_<role>_<start>-<end>.
# Every output entry gets:
#   context          transcripts of all previous turns of the conversation (both speakers)
#   context_audio    audio paths of those turns
#   utt_idx          0-based position of the turn in the conversation; it indexes the rows of the
#                    per-conversation embedding file written by dump_d2f_embeddings.py
#   conversation_id  the conversation UUID

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path


def utt_id_to_conv_id(utt_id):
    return utt_id.split("_")[0]


def get_st_end_time(identifier_str):
    # For identifiers like: <uuid>_banking_agent_00000-00005
    time_range = identifier_str.split("_")[-1]
    st_str, end_str = time_range.split("-", 1)
    return int(st_str), int(end_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add dialogue history (context), utt_idx and conversation_id to an utterance-level JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inp", required=True, type=Path, help="utterance-level JSONL (output of dump_jsonl.py)")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL")
    args = parser.parse_args()

    assert args.inp.exists(), f"inp doesn't exist: {args.inp}"

    with args.inp.open() as f:
        all_data = [json.loads(line) for line in f]
    print(f"loaded {len(all_data)} utterances")

    grouped_by_conv = defaultdict(list)
    for data in all_data:
        try:
            st_time, end_time = get_st_end_time(data["key"])
        except Exception as e:
            print(f"failed to get st/end time for {data['key']}, error: {e}")
            continue
        grouped_by_conv[utt_id_to_conv_id(data["key"])].append([st_time, end_time, data])

    print(f"found {len(grouped_by_conv)} conversations")
    for key in grouped_by_conv.keys():
        grouped_by_conv[key].sort(key=lambda x: x[0])

    n_written = 0
    with args.out.open(mode="w") as f_:
        for conv_id, cur_conv in grouped_by_conv.items():
            cur_context = []
            cur_context_audio = []
            for utt_idx, (_, _, data) in enumerate(cur_conv):
                data["context"] = copy.deepcopy(cur_context)
                data["context_audio"] = copy.deepcopy(cur_context_audio)
                data["utt_idx"] = utt_idx
                data["conversation_id"] = conv_id
                f_.write(json.dumps(data) + "\n")
                n_written += 1
                cur_context.append(data["target"])
                cur_context_audio.append(data["source"])

    print(f"wrote {n_written} utterances with context to {args.out}")
