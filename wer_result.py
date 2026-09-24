# SPDX-FileCopyrightText: Copyright © 2025-2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Computes WER and sentence error rate (SER) of a decoding, after normalizing the reference
# and the hypothesis with Whisper's English text normalizer.
# Hypotheses stuck in a loop (more than 5 words, each distinct word repeated more than 7 times
# on average) are replaced by UNK before scoring.
#
# Usage:
#   python wer_result.py <decode_output> --save
# reads <decode_output>_gt and <decode_output>_pred, and --save writes WER.txt next to them.

import os
import sys
import argparse

try:
    from compute_wer import Calculator
except ImportError:
    print("Error: `compute_wer` package not found. Please ensure it is installed: pip install compute-wer>=0.2.1")
    sys.exit(1)

try:
    from whisper_normalizer.english import EnglishTextNormalizer
except ImportError:
    print("Error: `whisper_normalizer` package not found. Please ensure it is installed: pip install whisper-normalizer")
    sys.exit(1)

HALLUCINATION_NEW_HYP = "UNK"
WORD_RATIO_THRESHOLD = 7.0

def read_and_normalize_kaldi_ark(file_path, normalizer):
    """Read Kaldi ark,t format file, normalize in memory, and return dict of utterance_id -> text."""
    utterances = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                utt_id, text = line.split('\t', 1)
                # Apply Whisper's English text normalization
                utterances[utt_id] = normalizer(text)
            elif line:  # Handle space-separated format
                parts = line.split(None, 1)
                if len(parts) == 2:
                    utterances[parts[0]] = normalizer(parts[1])
                elif len(parts) == 1:
                    utterances[parts[0]] = ""
    return utterances


def fix_hallucination(text, utt_id):
    """Fix hallucination in prediction text."""
    words = text.split()
    if len(words) > 5 and len(words) / len(set(words)) > WORD_RATIO_THRESHOLD:
        return HALLUCINATION_NEW_HYP
    return text


def compute_wer_from_utterances(gt_utterances, pred_utterances, remove_hallucinations=True):
    """Compute WER using Calculator class from utterance dictionaries."""
    calculator = Calculator()

    # Calculate WER for each utterance
    for utt_id in gt_utterances:
        reference = gt_utterances.get(utt_id, "")
        hypothesis = pred_utterances.get(utt_id, "")

        # Remove hallucinations if requested
        if remove_hallucinations:
            hypothesis = fix_hallucination(hypothesis, utt_id)

        calculator.calculate(reference, hypothesis)

    # Get overall results
    overall_wer, _ = calculator.overall()

    return f"Overall WER: {overall_wer}\nOverall SER: {calculator.ser}\n"


def compute_wer(path, save_to_file=False):
    """
    Compute WER from ground truth and prediction files.
    All processing is done in memory without creating intermediate files.

    Args:
        path: Path to the files (with or without _gt/_pred suffix)
        save_to_file: If True, save WER results to WER.txt file
    """
    # Strip _gt or _pred suffix if present
    if path.endswith('_gt'):
        path = path[:-3]
    elif path.endswith('_pred'):
        path = path[:-5]

    # Define file paths
    gt_file = f"{path}_gt"
    pred_file = f"{path}_pred"

    # Check if files exist
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file {gt_file} not found")
        sys.exit(1)
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file {pred_file} not found")
        sys.exit(1)

    # Whisper's English text normalizer is applied to both the reference and the hypothesis
    normalizer = EnglishTextNormalizer()

    # Read and normalize files in memory
    print(f"Reading and normalizing {gt_file}...")
    gt_utterances = read_and_normalize_kaldi_ark(gt_file, normalizer)
    print(f"Loaded {len(gt_utterances)} ground truth utterances")

    print(f"Reading and normalizing {pred_file}...")
    pred_utterances = read_and_normalize_kaldi_ark(pred_file, normalizer)
    print(f"Loaded {len(pred_utterances)} prediction utterances")

    # Compute WER with hallucination removal
    print(f"\n=== Computing WER ===")
    wer_output_final = compute_wer_from_utterances(gt_utterances, pred_utterances)
    print(wer_output_final)
    
    # Save to file if requested
    if save_to_file:
        # Get the directory where the transcript files are located
        output_dir = os.path.dirname(gt_file)
        wer_file_path = os.path.join(output_dir, "WER.txt")
        
        with open(wer_file_path, 'w') as f:
            f.write(wer_output_final)
        
        print(f"WER results saved to: {wer_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute WER (Word Error Rate) from ground truth and prediction files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python wer.py /path/to/results
  python wer.py /path/to/results_gt
  python wer.py /path/to/results --save
        '''
    )
    
    parser.add_argument(
        'path',
        help='Path to the files (with or without _gt/_pred suffix)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save WER results to WER.txt file in the same directory as the transcripts'
    )
    
    args = parser.parse_args()
    compute_wer(args.path, save_to_file=args.save)
