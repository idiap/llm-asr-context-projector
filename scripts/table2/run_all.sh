#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Entry point for table2. Submits the training and decoding jobs of every run.
export TABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TABLE_DIR/../common/run_all.sh" "$@"
