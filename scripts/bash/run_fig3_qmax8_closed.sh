#!/usr/bin/env bash
# Figure 3: qmax=8, qlen=81, C1–C6 closed (bond_ignore_array = []).
# Results: results_fig3_qmax8_closed[_COMMENT]
#
# Optional detail appended to the results directory.
# Leave empty, or set e.g. COMMENT="rerun2" -> results_fig3_qmax8_closed_rerun2
# Also: ./scripts/bash/run_fig3_qmax8_closed.sh --comment rerun2
COMMENT="${COMMENT:-}"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_fig3_common.sh" --qmax 8 --qlen 81 --ring closed ${COMMENT:+--comment "$COMMENT"} "$@"
