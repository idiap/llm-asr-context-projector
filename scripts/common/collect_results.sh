#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Print WER, bias WER (BWER) and entity F1 for every decoded run of one figure or table.
#
# WER comes from WER.txt, written at the end of decoding. BWER and F1 are computed here on CPU
# with bias_unbias_wer.py the first time a run is listed, and saved as BWER.txt next to WER.txt.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -z "$TABLE_DIR" ]; then
    echo "ERROR: TABLE_DIR is not set. Launch this through scripts/<figure|table>/results.sh." >&2
    exit 1
fi

run_var() {  # run_var DOMAIN METHOD CTX VARIABLE
    ( export DOMAIN="$1" METHOD="$2" CTX="$3" RETURN_JOB_ID=true
      source "$SCRIPT_DIR/config_defaults.sh" > /dev/null && eval "echo \"\${$4}\"" )
}

table_var() {  # table_var DOMAIN VARIABLE
    ( export DOMAIN="$1"
      source "$TABLE_DIR/config.sh" && eval "echo \"\${$2:-\${DEFAULT_$2}}\"" )
}

DOMAINS=${DOMAINS:-$(table_var banking DOMAINS)}

printf "%-11s %-8s %-4s %7s %7s %6s\n" domain method ctx WER BWER F1
printf "%-11s %-8s %-4s %7s %7s %6s\n" ----------- -------- ---- ------- ------- ------
for domain in $DOMAINS; do
    for experiment in base $(table_var "$domain" EXPERIMENTS); do
        method=${experiment%%:*}
        ctx=10
        [ "$experiment" != "$method" ] && ctx=${experiment#*:}
        case "$method" in
            base|raw_kw|raw_ctx) ckpt_var=BASE_CKPT_FOLDER ;;
            *)                   ckpt_var=CP_CKPT_FOLDER ;;
        esac
        decode_dir="$(run_var "$domain" "$method" "$ctx" RUN_EXP_DIR)/$(run_var "$domain" "$method" "$ctx" $ckpt_var)"
        shown_ctx=$ctx
        case "$method" in base|raw_kw) shown_ctx=- ;; esac

        if [ ! -f "$decode_dir/WER.txt" ]; then
            printf "%-11s %-8s %-4s %7s\n" "$domain" "$method" "$shown_ctx" "(not decoded yet)"
            continue
        fi

        entities=$(run_var "$domain" "$method" "$ctx" ENTITIES_PATH)
        if [ ! -f "$decode_dir/BWER.txt" ] && [ -f "$entities" ]; then
            python "$RUN_DIR/bias_unbias_wer.py" \
                --ref "$decode_dir/decode_output_gt" \
                --hyp "$decode_dir/decode_output_pred" \
                --utt2ner "$entities" \
                --entity-match-mode aligned \
                --entity-metrics \
                --normalize \
                > "$decode_dir/BWER.txt" 2> "$decode_dir/BWER.log" || rm -f "$decode_dir/BWER.txt"
        fi

        python - "$decode_dir" "$domain" "$method" "$shown_ctx" <<'PYEOF'
import json, os, re, sys
decode_dir, domain, method, ctx = sys.argv[1:]
wer = re.search(r"Overall WER:\s*([\d.]+)", open(os.path.join(decode_dir, "WER.txt")).read())
bwer = f1 = "-"
path = os.path.join(decode_dir, "BWER.txt")
if os.path.exists(path):
    text = open(path).read()
    scores = json.loads(text[text.index("{"):])
    bwer = f"{scores['BiasWER']:.2f}"
    f1 = f"{scores['entities']['f1']:.2f}" if scores.get("entities") else "-"
print(f"{domain:11} {method:8} {ctx:4} {wer.group(1) if wer else '-':>7} {bwer:>7} {f1:>6}")
PYEOF
    done
done
