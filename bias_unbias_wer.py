# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Computes WER, bias WER (errors on reference tokens covered by a named entity),
# unbiased WER (all other tokens), and entity-level precision, recall and F1.
#
# Usage:
#   python bias_unbias_wer.py --ref decode_output_gt --hyp decode_output_pred \
#       --utt2ner <entities file> --entity-match-mode aligned --entity-metrics --normalize

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import kaldialign
from whisper_normalizer.english import EnglishTextNormalizer


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_per_line(file_path: Path) -> Dict[str, List[str]]:
    """Loads a Kaldi-style text file: `utt_id token1 token2 ...`.

    Returns dict of utt_id -> list of tokens (possibly empty).
    """
    cur_dict: Dict[str, List[str]] = {}
    for raw_line in file_path.open().readlines():
        line = _normalize_spaces(raw_line)
        if not line:
            continue
        parts = line.split(" ")
        utt_id = parts[0]
        if utt_id in cur_dict:
            raise ValueError(f"duplicate utt_id in {file_path}: {utt_id}")
        cur_dict[utt_id] = [t.upper() for t in parts[1:]]
    return cur_dict


def load_ners_from_file(ner_file_path: Path) -> Dict[str, List[List[str]]]:
    """Loads utt2ner with format: `utt_id NE1, NE2`.

    Each NE is split on whitespace into tokens. NEs are split on commas.
    Lines with only utt_id are allowed.
    """
    all_ne: Dict[str, List[List[str]]] = defaultdict(list)
    for raw_line in ner_file_path.open().readlines():
        line = _normalize_spaces(raw_line)
        if not line:
            continue
        parts = line.split(" ")
        utt_id = parts[0]
        if len(parts) == 1:
            continue

        nes_text = " ".join(parts[1:]).strip()
        if not nes_text:
            continue

        for ne_text in nes_text.split(","):
            ne_text = _normalize_spaces(ne_text)
            if not ne_text:
                continue
            all_ne[utt_id].append([t.upper() for t in ne_text.split(" ")])
    return all_ne


def _upper_dict_words(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {utt_id: [t.upper() for t in words] for utt_id, words in data.items()}


def _upper_dict_ners(data: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
    return {
        utt_id: [[t.upper() for t in ne] for ne in ner_list]
        for utt_id, ner_list in data.items()
    }


def find_sublist_indices(a: Sequence[str], b: Sequence[str]) -> List[int]:
    n, m = len(a), len(b)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if all(a[i + j] == b[j] for j in range(m))]


def count_phrase_occurrences_aligned(
    aligned_tokens: Sequence[str], phrase: Sequence[str], eps_symbol: str
) -> int:
    """Count non-overlapping occurrences of `phrase` in an aligned token stream.

    `aligned_tokens` can include `eps_symbol`. Eps tokens are ignored for the
    purpose of matching, allowing matches to span over eps positions.

    Matching is greedy left-to-right and non-overlapping.
    """
    if not phrase:
        return 0

    count = 0
    i = 0
    n = len(aligned_tokens)
    m = len(phrase)
    while i < n:
        k = i
        j = 0
        while k < n and j < m:
            tok = aligned_tokens[k]
            if tok == eps_symbol:
                k += 1
                continue
            if tok == phrase[j]:
                k += 1
                j += 1
                continue
            break

        if j == m:
            count += 1
            i = k
        else:
            i += 1
    return count


def build_ne_mask(
    ref_words: Sequence[str], ner_list: Sequence[Sequence[str]]
) -> List[bool]:
    """Returns a boolean mask over ref_words for tokens covered by any NE phrase."""
    mask = [False] * len(ref_words)
    for ne in ner_list:
        if not ne:
            continue
        for start in find_sublist_indices(ref_words, ne):
            for j in range(len(ne)):
                mask[start + j] = True
    return mask


def validate_ners_in_ref(
    ref_words: Sequence[str], ner_list: Sequence[Sequence[str]]
) -> None:
    """Raise if any NE phrase is not found as a contiguous span in ref_words."""
    for ne in ner_list:
        if not ne:
            continue
        if not find_sublist_indices(ref_words, ne):
            raise ValueError(
                "NE not found in reference: "
                f"NE={' '.join(ne)!r} ref={' '.join(ref_words)!r}"
            )


def ners_not_found_in_ref(
    ref_words: Sequence[str], ner_list: Sequence[Sequence[str]]
) -> List[List[str]]:
    missing: List[List[str]] = []
    for ne in ner_list:
        if not ne:
            continue
        if not find_sublist_indices(ref_words, ne):
            missing.append(list(ne))
    return missing


def build_phrase_mask(
    words: Sequence[str], phrases: Sequence[Sequence[str]]
) -> List[bool]:
    """Mark tokens in `words` that belong to any matched phrase in `phrases`."""
    mask = [False] * len(words)
    for phrase in phrases:
        if not phrase:
            continue
        for start in find_sublist_indices(words, phrase):
            for j in range(len(phrase)):
                mask[start + j] = True
    return mask


@dataclass
class EntityTokenCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        if d == 0:
            return None
        return self.tp / d

    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        if d == 0:
            return None
        return self.tp / d

    def f1(self) -> Optional[float]:
        p = self.precision()
        r = self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def accuracy(self) -> Optional[float]:
        d = self.tp + self.fp + self.fn + self.tn
        if d == 0:
            return None
        return (self.tp + self.tn) / d


@dataclass
class EntityMentionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        if d == 0:
            return None
        return self.tp / d

    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        if d == 0:
            return None
        return self.tp / d

    def f1(self) -> Optional[float]:
        p = self.precision()
        r = self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def accuracy(self) -> Optional[float]:
        # Interpreted as "fraction of gold mentions found".
        d = self.tp + self.fn
        if d == 0:
            return None
        return self.tp / d


@dataclass
class EntityMetrics:
    total_entities_in_ref: int = 0
    entities_ignored_not_in_ref: int = 0
    entities_found_in_hyp: int = 0
    entities_predicted_in_hyp: int = 0
    mention_counts: EntityMentionCounts = field(default_factory=EntityMentionCounts)
    token_counts: EntityTokenCounts = field(default_factory=EntityTokenCounts)


@dataclass
class WerCounts:
    # Numerators
    sub: int = 0
    ins: int = 0
    dele: int = 0
    cor: int = 0
    # Denominator (number of reference words)
    ref_words: int = 0

    def wer(self) -> Optional[float]:
        if self.ref_words == 0:
            return None
        return 100.0 * (self.sub + self.ins + self.dele) / self.ref_words


@dataclass
class WerMetrics:
    overall: WerCounts
    bias: WerCounts
    unbiased: WerCounts
    # bookkeeping
    num_utts_scored: int
    num_utts_missing_ref: int
    num_utts_missing_hyp: int
    ne_not_found: Optional[List[Dict[str, object]]] = None
    entity_metrics: Optional[EntityMetrics] = None


def _accumulate_alignment(
    ali: Sequence[Tuple[str, str]],
    ne_mask: Sequence[bool],
    eps: str,
    overall: WerCounts,
    bias: WerCounts,
    unbiased: WerCounts,
) -> None:
    ref_index = -1
    for ref_tok, hyp_tok in ali:
        if ref_tok != eps:
            ref_index += 1
            is_ne = bool(ne_mask[ref_index]) if ref_index < len(ne_mask) else False
            overall.ref_words += 1
            (bias if is_ne else unbiased).ref_words += 1
        else:
            is_ne = False

        if ref_tok == eps and hyp_tok != eps:
            overall.ins += 1
            # Insertions have no reference position; by default we attribute them to
            # the unbiased bucket (non-NE).
            unbiased.ins += 1
        elif ref_tok != eps and hyp_tok == eps:
            overall.dele += 1
            (bias if is_ne else unbiased).dele += 1
        elif ref_tok != eps and hyp_tok != eps and ref_tok != hyp_tok:
            overall.sub += 1
            (bias if is_ne else unbiased).sub += 1
        else:
            # correct includes the (eps, eps) case which kaldialign shouldn't emit,
            # but harmless.
            overall.cor += 1
            if ref_tok != eps:
                (bias if is_ne else unbiased).cor += 1


def whisper_normalize(
    words: Sequence[str], normalizer: EnglishTextNormalizer
) -> List[str]:
    """Apply Whisper's English text normalizer to a token list.

    The output is upper-cased again, like every other token in this script.
    """
    return normalizer(" ".join(words)).upper().split()


def compute_wer_metrics(
    ref: Union[str, Path, Dict[str, List[str]]],
    hyp: Union[str, Path, Dict[str, List[str]]],
    utt2ner: Optional[Union[str, Path, Dict[str, List[List[str]]]]] = None,
    *,
    eps_symbol: str = "*",
    strict: bool = False,
    ne_strict: bool = False,
    collect_ne_not_found: bool = False,
    compute_entity_metrics: bool = False,
    entity_match_mode: str = "raw",
    normalize: bool = False,
) -> WerMetrics:
    """Compute overall WER, bias-WER (NE ref tokens), and unbiased-WER.

    `ref`/`hyp` can be a Path (or str) to a file, or already-loaded dict mapping
    utt_id -> list of tokens.
    `utt2ner` is optional. If not provided, bias bucket will be empty (NA).
    With `normalize`, Whisper's English text normalizer is applied to the
    references, the hypotheses and every entity.

    Notes:
    - We score only utterances present in both ref and hyp.
    - Bias/unbiased buckets are defined by reference tokens that are covered by
      any NE phrase for that utterance.
    - Insertions are attributed to the unbiased bucket.
    """
    if isinstance(ref, (str, Path)):
        ref_dict = load_per_line(Path(ref))
    else:
        ref_dict = _upper_dict_words(ref)
    if isinstance(hyp, (str, Path)):
        hyp_dict = load_per_line(Path(hyp))
    else:
        hyp_dict = _upper_dict_words(hyp)

    if utt2ner is None:
        utt2ner_dict: Dict[str, List[List[str]]] = {}
    elif isinstance(utt2ner, (str, Path)):
        utt2ner_dict = load_ners_from_file(Path(utt2ner))
    else:
        utt2ner_dict = _upper_dict_ners(utt2ner)

    if normalize:
        print(
            "Running Whisper's English text normalization on references, hypotheses and entities",
            file=sys.stderr,
        )
        normalizer = EnglishTextNormalizer()
        ref_dict = {u: whisper_normalize(w, normalizer) for u, w in ref_dict.items()}
        hyp_dict = {u: whisper_normalize(w, normalizer) for u, w in hyp_dict.items()}
        # Each entity is normalized on its own; entities that become empty are dropped
        utt2ner_dict = {
            utt_id: [ne for ne in (whisper_normalize(ne, normalizer) for ne in ner_list) if ne]
            for utt_id, ner_list in utt2ner_dict.items()
        }

    if ne_strict and utt2ner_dict:
        # Ensure every utt_id in utt2ner exists in ref, and every NE is discoverable.
        for utt_id, ner_list in utt2ner_dict.items():
            if utt_id not in ref_dict:
                raise ValueError(
                    f"utt_id {utt_id!r} present in utt2ner but missing in ref"
                )
            try:
                validate_ners_in_ref(ref_dict[utt_id], ner_list)
            except ValueError as e:
                raise ValueError(f"utt_id {utt_id!r}: {e}") from e

    ref_utts = set(ref_dict.keys())
    hyp_utts = set(hyp_dict.keys())
    common_utts = sorted(ref_utts & hyp_utts)
    missing_ref = sorted(hyp_utts - ref_utts)
    missing_hyp = sorted(ref_utts - hyp_utts)

    if strict:
        if missing_ref:
            raise ValueError(f"{len(missing_ref)} utts missing in ref (present in hyp)")
        if missing_hyp:
            raise ValueError(f"{len(missing_hyp)} utts missing in hyp (present in ref)")

    overall = WerCounts()
    bias = WerCounts()
    unbiased = WerCounts()

    entity_metrics: Optional[EntityMetrics] = (
        EntityMetrics() if compute_entity_metrics else None
    )

    ne_not_found_records: Optional[List[Dict[str, object]]] = (
        [] if collect_ne_not_found else None
    )

    for utt_id in common_utts:
        ref_words = ref_dict.get(utt_id, [])
        hyp_words = hyp_dict.get(utt_id, [])
        ner_list = utt2ner_dict.get(utt_id, [])
        if ne_strict and ner_list:
            # Redundant with the global check above, but also supports cases
            # where utt2ner_dict is provided only for a subset.
            validate_ners_in_ref(ref_words, ner_list)

        if (not ne_strict) and ner_list and collect_ne_not_found:
            missing = ners_not_found_in_ref(ref_words, ner_list)
            if missing and ne_not_found_records is not None:
                ne_not_found_records.append(
                    {
                        "utt_id": utt_id,
                        "ref": " ".join(ref_words),
                        "hyp": " ".join(hyp_words),
                        "ne_list": [" ".join(ne) for ne in ner_list],
                        "ne_not_found": [" ".join(ne) for ne in missing],
                    }
                )

        if entity_metrics is not None and ner_list:
            # Mention-level scoring (counts duplicates exactly as in utt2ner).
            # Any mention not found in reference is ignored.
            if entity_match_mode not in {"raw", "aligned"}:
                raise ValueError(
                    f"entity_match_mode must be 'raw' or 'aligned', got {entity_match_mode!r}"
                )

            ali_for_entities = kaldialign.align(ref_words, hyp_words, eps_symbol)
            ref_aligned = [r for r, _ in ali_for_entities]
            hyp_aligned = [h for _, h in ali_for_entities]

            def _ref_occ(phrase_tokens: List[str]) -> int:
                if entity_match_mode == "aligned":
                    return count_phrase_occurrences_aligned(
                        ref_aligned, phrase_tokens, eps_symbol
                    )
                return len(find_sublist_indices(ref_words, phrase_tokens))

            def _hyp_occ(phrase_tokens: List[str]) -> int:
                if entity_match_mode == "aligned":
                    return count_phrase_occurrences_aligned(
                        hyp_aligned, phrase_tokens, eps_symbol
                    )
                return len(find_sublist_indices(hyp_words, phrase_tokens))

            utt2ner_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
            for ne in ner_list:
                if not ne:
                    continue
                utt2ner_counts[tuple(ne)] += 1

            gold_phrase_counts: Dict[Tuple[str, ...], int] = {}
            ignored_mentions = 0
            for phrase, requested_count in utt2ner_counts.items():
                phrase_list = list(phrase)
                ref_occ = _ref_occ(phrase_list)
                effective_gold = min(requested_count, ref_occ)
                if effective_gold == 0:
                    ignored_mentions += requested_count
                    continue
                if ref_occ < requested_count:
                    ignored_mentions += requested_count - ref_occ
                gold_phrase_counts[phrase] = effective_gold

            tp = fp = fn = 0
            predicted_mentions = 0
            found_mentions = 0
            for phrase, gold_count in gold_phrase_counts.items():
                phrase_list = list(phrase)
                hyp_occ = _hyp_occ(phrase_list)
                predicted_mentions += hyp_occ
                tp_phrase = min(gold_count, hyp_occ)
                fp_phrase = max(0, hyp_occ - gold_count)
                fn_phrase = max(0, gold_count - hyp_occ)
                tp += tp_phrase
                fp += fp_phrase
                fn += fn_phrase
                found_mentions += tp_phrase

            entity_metrics.total_entities_in_ref += sum(gold_phrase_counts.values())
            entity_metrics.entities_ignored_not_in_ref += ignored_mentions
            entity_metrics.entities_found_in_hyp += found_mentions
            entity_metrics.entities_predicted_in_hyp += predicted_mentions
            entity_metrics.mention_counts.tp += tp
            entity_metrics.mention_counts.fp += fp
            entity_metrics.mention_counts.fn += fn

            # Token-level scoring using the set of gold phrases (duplicates don't
            # change token masks).
            gold_phrases_for_mask = [list(p) for p in gold_phrase_counts.keys()]
            ref_entity_mask = build_phrase_mask(ref_words, gold_phrases_for_mask)
            hyp_entity_mask = build_phrase_mask(hyp_words, gold_phrases_for_mask)

            # Token-level classification via alignment indices.
            ali = ali_for_entities
            ref_idx = -1
            hyp_idx = -1
            for ref_tok, hyp_tok in ali:
                if ref_tok != eps_symbol:
                    ref_idx += 1
                    gold = bool(ref_entity_mask[ref_idx])
                else:
                    gold = False
                if hyp_tok != eps_symbol:
                    hyp_idx += 1
                    pred = bool(hyp_entity_mask[hyp_idx])
                else:
                    pred = False

                if gold and pred:
                    entity_metrics.token_counts.tp += 1
                elif (not gold) and pred:
                    entity_metrics.token_counts.fp += 1
                elif gold and (not pred):
                    entity_metrics.token_counts.fn += 1
                else:
                    entity_metrics.token_counts.tn += 1
        ne_mask = build_ne_mask(ref_words, ner_list)

        ali = kaldialign.align(ref_words, hyp_words, eps_symbol)
        _accumulate_alignment(ali, ne_mask, eps_symbol, overall, bias, unbiased)

    return WerMetrics(
        overall=overall,
        bias=bias,
        unbiased=unbiased,
        num_utts_scored=len(common_utts),
        num_utts_missing_ref=len(missing_ref),
        num_utts_missing_hyp=len(missing_hyp),
        ne_not_found=ne_not_found_records,
        entity_metrics=entity_metrics,
    )


def _pct(n: int, d: int) -> Optional[float]:
    if d == 0:
        return None
    return 100.0 * n / d


def counts_to_json(counts: WerCounts) -> Dict[str, object]:
    wer_value = counts.wer()
    return {
        "wer": wer_value,
        "ref_words": counts.ref_words,
        "sub": counts.sub,
        "ins": counts.ins,
        "del": counts.dele,
        "cor": counts.cor,
        "sub_pct": _pct(counts.sub, counts.ref_words),
        "ins_pct": _pct(counts.ins, counts.ref_words),
        "del_pct": _pct(counts.dele, counts.ref_words),
    }


def entity_metrics_to_json(
    metrics: Optional[EntityMetrics],
) -> Optional[Dict[str, object]]:
    if metrics is None:
        return None
    mc = metrics.mention_counts
    tc = metrics.token_counts
    return {
        "total_entities_in_ref": metrics.total_entities_in_ref,
        "entities_ignored_not_in_ref": metrics.entities_ignored_not_in_ref,
        "entities_found_in_hyp": metrics.entities_found_in_hyp,
        "entities_predicted_in_hyp": metrics.entities_predicted_in_hyp,
        "counts": {"tp": mc.tp, "fp": mc.fp, "fn": mc.fn},
        "accuracy": mc.accuracy(),
        "precision": mc.precision(),
        "recall": mc.recall(),
        "f1": mc.f1(),
        "token_counts": {"tp": tc.tp, "fp": tc.fp, "fn": tc.fn, "tn": tc.tn},
        "token_accuracy": tc.accuracy(),
        "token_precision": tc.precision(),
        "token_recall": tc.recall(),
        "token_f1": tc.f1(),
    }


def main(argv: Optional[Sequence[str]] = None) -> WerMetrics:
    parser = argparse.ArgumentParser(
        description="Compute WER, bias-WER (NE-only) and unbiased-WER (non-NE-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ref", type=Path, required=True, help="Reference file")
    parser.add_argument("--hyp", type=Path, required=True, help="Hypothesis file")
    parser.add_argument(
        "--utt2ner",
        type=Path,
        default=None,
        help="Optional utt2ner file: utt_id NE1, NE2",
    )
    parser.add_argument(
        "--eps-symbol",
        type=str,
        default="*",
        help="Epsilon symbol used for insertions/deletions in alignment",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error if ref and hyp utt_id sets differ",
    )
    parser.add_argument(
        "--ne-strict",
        action="store_true",
        help="Error if any NE in utt2ner is not found in the corresponding ref text",
    )
    parser.add_argument(
        "--no-print-missing-ne",
        action="store_true",
        help="Disable printing missing NEs to stderr when --ne-strict is not set",
    )
    parser.add_argument(
        "--entity-metrics",
        action="store_true",
        help="Compute entity accuracy/precision/recall/F1 based on utt2ner entities",
    )
    parser.add_argument(
        "--entity-match-mode",
        choices=["raw", "aligned"],
        default="raw",
        help="How to count entity occurrences for mention metrics",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply Whisper's English text normalizer to references, hypotheses and entities",
    )

    args = parser.parse_args(argv)
    if not args.ref.exists():
        raise FileNotFoundError(f"ref file {args.ref} does not exist")
    if not args.hyp.exists():
        raise FileNotFoundError(f"hyp file {args.hyp} does not exist")
    if args.utt2ner is not None and not args.utt2ner.exists():
        raise FileNotFoundError(f"utt2ner file {args.utt2ner} does not exist")

    metrics = compute_wer_metrics(
        args.ref,
        args.hyp,
        args.utt2ner,
        eps_symbol=args.eps_symbol,
        strict=args.strict,
        ne_strict=args.ne_strict,
        collect_ne_not_found=(
            (args.utt2ner is not None)
            and (not args.ne_strict)
            and (not args.no_print_missing_ne)
        ),
        compute_entity_metrics=args.entity_metrics,
        entity_match_mode=args.entity_match_mode,
        normalize=args.normalize,
    )

    if (not args.ne_strict) and (not args.no_print_missing_ne) and metrics.ne_not_found:
        for rec in metrics.ne_not_found:
            print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)

    out = {
        "WER": metrics.overall.wer(),
        "BiasWER": metrics.bias.wer(),
        "UnbiasedWER": metrics.unbiased.wer(),
        "overall": counts_to_json(metrics.overall),
        "bias": counts_to_json(metrics.bias),
        "unbiased": counts_to_json(metrics.unbiased),
        "entities": (
            None
            if metrics.entity_metrics is None
            else {
                "match_mode": args.entity_match_mode,
                **entity_metrics_to_json(metrics.entity_metrics),
            }
        ),
        "num_utts_scored": metrics.num_utts_scored,
        "num_utts_missing_ref": metrics.num_utts_missing_ref,
        "num_utts_missing_hyp": metrics.num_utts_missing_hyp,
        "num_utts_with_missing_ne": (
            0 if not metrics.ne_not_found else len(metrics.ne_not_found)
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
