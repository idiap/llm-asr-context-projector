#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Table 2: WER, BWER and entity F1 across the five domains (Section 3.2 of the paper)
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here.

DEFAULT_DOMAINS=${DEFAULT_DOMAINS:-"banking healthcare insurance retail teleco"}

# Context size (CTX column) reported for each domain.
case "${DOMAIN:-banking}" in
    banking)    PAPER_CTX_CP=10; PAPER_CTX_KW_CP=10 ;;
    healthcare) PAPER_CTX_CP=10; PAPER_CTX_KW_CP=10 ;;
    insurance)  PAPER_CTX_CP=10; PAPER_CTX_KW_CP=5 ;;
    retail)     PAPER_CTX_CP=10; PAPER_CTX_KW_CP=10 ;;
    teleco)     PAPER_CTX_CP=1;  PAPER_CTX_KW_CP=1 ;;
esac

# Rows per domain, as METHOD or METHOD:CTX. The base row is always run.
DEFAULT_EXPERIMENTS=${DEFAULT_EXPERIMENTS:-"raw_kw cp:$PAPER_CTX_CP kw_cp:$PAPER_CTX_KW_CP"}
