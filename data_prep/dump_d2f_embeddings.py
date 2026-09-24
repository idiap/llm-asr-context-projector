# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Step 5: pre-compute the Dialog2Flow sentence embeddings of every turn (Section 2.3 of the paper).
#
# For every conversation found in the input JSONL files, writes
#   <output_root>/<conversation_id>/<conversation_id>_embedding.npy
# a float32 array of shape (number of turns, 768) whose row i is the mean-pooled embedding of the
# transcript of turn utt_idx=i. Point dataset_config.context_embeddings_path at <output_root>.
#
# Conversation ids are unique across splits, so train, dev and test can share one output folder.

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_conversation_id(key):
    return key.split("_")[0]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def build_embedder(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def get_embedding(text):
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        encoded_input.to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        return sentence_embeddings.cpu().numpy()

    return get_embedding


def process_file(input_file, output_root, get_embedding):
    conversations = {}
    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            conversations.setdefault(get_conversation_id(entry["key"]), []).append(entry)

    for conv_id, entries in tqdm(conversations.items(), desc=os.path.basename(input_file)):
        # Rows must follow utt_idx, the turn order used by the dataset
        entries.sort(key=lambda e: e["utt_idx"])
        assert [e["utt_idx"] for e in entries] == list(range(len(entries))), \
            f"utt_idx of conversation {conv_id} is not 0..N-1"

        conv_folder = os.path.join(output_root, conv_id)
        os.makedirs(conv_folder, exist_ok=True)
        embeddings = get_embedding([entry["target"] for entry in entries])
        with open(os.path.join(conv_folder, conv_id + "_embedding.npy"), "wb") as f:
            np.save(f, embeddings)

    return len(conversations)


def main():
    parser = argparse.ArgumentParser(
        description="Dump Dialog2Flow mean-pooled embeddings, one file per conversation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="JSONL files with utt_idx (e.g. the train, dev and test splits of one domain)")
    parser.add_argument("--output_root", required=True,
                        help="folder where <conversation_id>/<conversation_id>_embedding.npy files are written")
    parser.add_argument("--model_name", default="sergioburdisso/dialog2flow-joint-bert-base",
                        help="Hugging Face sentence encoder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_embedding = build_embedder(args.model_name, device)

    total = sum(process_file(f, args.output_root, get_embedding) for f in args.inputs)
    print(f"Embeddings generated for {total} conversations in {args.output_root}")


if __name__ == "__main__":
    main()
