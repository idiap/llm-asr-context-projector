#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Figure 3: context projection (+cp) and keywords with context projection (+kw+cp)
# as a function of the context size (Section 3.2 of the paper)
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here.

DEFAULT_DOMAINS=${DEFAULT_DOMAINS:-"banking healthcare insurance retail teleco"}

# Runs per domain, as METHOD or METHOD:CTX. The base model is always run.
# raw_kw gives the "+kw" reference line of the figure.
DEFAULT_EXPERIMENTS=${DEFAULT_EXPERIMENTS:-"raw_kw cp:1 cp:5 cp:10 cp:ALL kw_cp:1 kw_cp:5 kw_cp:10 kw_cp:ALL"}
