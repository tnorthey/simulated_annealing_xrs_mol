#!/usr/bin/env bash
# Figure 3: qmax=4, qlen=41, C1–C6 closed (bond_ignore_array = []).
# Results: results_fig3_qmax4_closed
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fig3_common.sh" --qmax 4 --qlen 41 --ring closed "$@"
