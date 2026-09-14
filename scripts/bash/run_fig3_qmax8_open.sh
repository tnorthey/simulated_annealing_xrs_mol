#!/usr/bin/env bash
# Figure 3: qmax=8, qlen=81, C1–C6 open (bond_ignore_array = [[0, 5]]).
# Results: results_fig3_qmax8_open
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fig3_common.sh" --qmax 8 --qlen 81 --ring open "$@"
