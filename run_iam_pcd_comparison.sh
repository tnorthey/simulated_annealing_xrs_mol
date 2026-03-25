#!/usr/bin/env bash
# Run two calculate_iam.py PCD cases (correction+DAT ref vs ion+DAT ref) and compare outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
INPUT_XYZ="${INPUT_XYZ:-chd+_data/CCSD-neut.xyz}"
OUT_A="${OUT_A:-results/iam_pcd_corr_plus44.dat}"
OUT_B="${OUT_B:-results/iam_pcd_ion_neut_iam.dat}"
# Match input.toml [scattering_params.q]
QFLAGS=(--qmin 0.001 --qmax 4.0 --qlen 100)

mkdir -p "$(dirname "$OUT_A")"
mkdir -p "$(dirname "$OUT_B")"

echo "=== Run A: PCD + correction factor, reference CCSD_neut_scat_plus44.dat ==="
"$PYTHON" calculate_iam.py "$INPUT_XYZ" "$OUT_A" --inelastic --pcd \
  --reference-dat chd+_data/CCSD_neut_scat_plus44.dat \
  --correction-factor chd+_data/IAM_corr.dat \
  "${QFLAGS[@]}"

echo "=== Run B: ion + PCD, reference CCSD-neut_iam_scat.dat ==="
"$PYTHON" calculate_iam.py "$INPUT_XYZ" "$OUT_B" --inelastic --ion --pcd \
  --reference-dat chd+_data/CCSD-neut_iam_scat.dat \
  "${QFLAGS[@]}"

echo "=== Compare column-2 (PCD) on common q grid ==="
"$PYTHON" -c "
import numpy as np
import sys

def load(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        raise SystemExit(f'{path}: expected 2 columns (q, PCD)')
    return a[:, 0], a[:, 1]

q1, y1 = load('$OUT_A')
q2, y2 = load('$OUT_B')
if not np.allclose(q1, q2):
    print('Warning: q columns differ; interpolating B onto A grid', file=sys.stderr)
    y2 = np.interp(q1, q2, y2, left=y2[0], right=y2[-1])
    q2 = q1
d = y1 - y2
print(f'max abs diff: {np.max(np.abs(d)):.6e}')
print(f'RMS diff:     {np.sqrt(np.mean(d*d)):.6e}')
"

echo "Done. Outputs: $OUT_A  $OUT_B"
