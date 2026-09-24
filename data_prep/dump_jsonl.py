# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Step 1: write an utterance-level JSONL ({"key", "source", "target"}) from a lhotse cut manifest.
#
# With --save-audio, every cut is cut out of its (long, two-channel) recording and saved as a
# 16 kHz wav file, and "source" points to that file.

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from lhotse import load_manifest_lazy
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump an utterance-level JSONL from a lhotse cut manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inp", required=True, type=Path, help="lhotse cut manifest (e.g. banking_test_cuts.jsonl.gz)")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL file")
    parser.add_argument("--sr", type=int, default=16000, help="expected sampling rate")
    parser.add_argument(
        "--save-audio",
        type=Path,
        help="if given, save every cut as a wav file in this folder and point 'source' to it",
    )
    args = parser.parse_args()

    assert args.inp.exists(), f"inp doesn't exist: {args.inp}"
    if args.save_audio is not None:
        args.save_audio.mkdir(parents=True, exist_ok=True)
        args.save_audio = args.save_audio.resolve()
        print(f"saving segmented audio to: {args.save_audio}")

    wav = {}
    text = {}
    for cut_ in tqdm(load_manifest_lazy(args.inp).to_eager(), desc="manifest"):
        text[cut_.id] = cut_.supervisions[0].text
        assert (
            len(cut_.recording.sources[0].channels) == 1
            or args.save_audio is not None
        ), f"multichannel audio, use --save-audio: {cut_.id}"
        assert cut_.recording.sampling_rate == args.sr, f"wav sr: {cut_.id}"

        wav[cut_.id] = cut_.recording.sources[0].source

        if args.save_audio is not None:
            aud_ = cut_.load_audio()
            aud_path = args.save_audio / f"{cut_.id}.wav"
            torchaudio.save(
                f"{aud_path}",
                torch.from_numpy(aud_),
                sample_rate=cut_.recording.sampling_rate,
            )
            wav[cut_.id] = f"{aud_path}"

    print(f"total wav: {len(wav)}, text: {len(text)}")

    total_dumped = 0
    with args.out.open(mode="w") as f_:
        for wav_id, wav_path in wav.items():
            if text.get(wav_id) is None:
                continue
            total_dumped += 1
            json.dump({"key": wav_id, "source": wav_path, "target": text[wav_id]}, f_)
            f_.write("\n")
    print(f"total written: {total_dumped}")
