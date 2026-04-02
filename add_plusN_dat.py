#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _safe_n_for_filename(n_str: str) -> str:
    s = n_str.strip()
    if not s:
        return "0"
    # Keep it readable but filesystem-friendly.
    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Load a 2-column .dat file, add N to column 2, write *_plusN.dat"
    )
    ap.add_argument("input_dat", type=Path, help="Input .dat file (2 columns)")
    ap.add_argument("N", type=str, help="Offset to add to 2nd column (e.g. 0.1, -2)")
    args = ap.parse_args()

    in_path: Path = args.input_dat
    n_str: str = args.N
    try:
        n = float(n_str)
    except ValueError as e:
        raise SystemExit(f"Error: N must be a number, got {n_str!r}") from e

    data = np.loadtxt(in_path, dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise SystemExit(
            f"Error: expected at least 2 columns in {str(in_path)!r}, got shape {arr.shape}"
        )

    out = arr.copy()
    out[:, 1] = out[:, 1] + n

    n_tag = _safe_n_for_filename(n_str)
    out_path = in_path.with_name(f"{in_path.stem}_plus{n_tag}.dat")

    np.savetxt(out_path, out[:, :2], fmt="%.8f %.8f")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

