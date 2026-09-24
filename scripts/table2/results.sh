#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Prints WER, bias WER and entity F1 for every decoded run of table2.
export TABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TABLE_DIR/../common/collect_results.sh" "$@"
