#!/usr/bin/env bash
# Figure 3: qmax=4, qlen=41, C1–C6 open (bond_ignore_array = [[0, 5]]).
# Results: results_fig3_qmax4_open[_COMMENT]
#
# Optional detail appended to the results directory.
# Leave empty, or set e.g. COMMENT="rerun2" -> results_fig3_qmax4_open_rerun2
# Also: ./scripts/bash/run_fig3_qmax4_open.sh --comment rerun2
COMMENT="${COMMENT:-}"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fig3_common.sh" --qmax 4 --qlen 41 --ring open ${COMMENT:+--comment "$COMMENT"} "$@"
