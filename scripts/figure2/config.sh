#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Figure 2: raw context appended to the prompt (Section 3.1 of the paper)
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here.

DEFAULT_DOMAINS=${DEFAULT_DOMAINS:-"banking healthcare insurance retail teleco"}

# Runs per domain, as METHOD or METHOD:CTX. The base model (CTX_00 in the figure) is always run.
#   raw_kw     "Keywords": keywords of the previous turns in the prompt
#   raw_ctx:1  "CTX_01":   transcript of the previous turn in the prompt
#   raw_ctx:10 "CTX_10":   transcripts of the last 10 turns in the prompt
DEFAULT_EXPERIMENTS=${DEFAULT_EXPERIMENTS:-"raw_kw raw_ctx:1 raw_ctx:10"}
