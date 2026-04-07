#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _parse_indices(expr: str, natom: int, one_based: bool) -> List[int]:
    """
    Parse indices like:
      - "0,1,2"
      - "0-5"
      - "0-5,8,10-12"
    Empty expr => all atoms.
    """
    expr = expr.strip()
    if not expr:
        return list(range(natom))

    out: List[int] = []
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if one_based:
                a -= 1
                b -= 1
            if a > b:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            idx = int(p)
            if one_based:
                idx -= 1
            out.append(idx)

    uniq = sorted(set(out))
    for i in uniq:
        if i < 0 or i >= natom:
            raise ValueError(f"Index {i} out of range for natom={natom}")
    return uniq


@dataclass(frozen=True)
class XyzSingle:
    natom: int
    comment: str
    atoms: np.ndarray  # shape (natom,), dtype=str
    xyz: np.ndarray  # shape (natom, 3), float


def _read_single_xyz(path: str) -> XyzSingle:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from modules.mol import Xyz  # noqa: E402

    xyz_util = Xyz()
    natom, comment, atoms, xyz = xyz_util.read_xyz(path)
    atoms = np.array(atoms, dtype=str, copy=True)
    xyz = np.array(xyz, dtype=float, copy=True)
    if xyz.shape != (natom, 3):
        raise ValueError(f"Bad xyz shape in {path!r}: got {xyz.shape}, expected {(natom, 3)}")
    if atoms.shape != (natom,):
        raise ValueError(f"Bad atom labels shape in {path!r}: got {atoms.shape}, expected {(natom,)}")
    return XyzSingle(natom=natom, comment=str(comment).rstrip("\n"), atoms=atoms, xyz=xyz)


def _centroid(xyz: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("centroid indices is empty")
    return xyz[np.array(indices, dtype=int), :].mean(axis=0)


def _max_carbon_displacement(
    ref: XyzSingle,
    mov: XyzSingle,
    centroid_indices: Sequence[int],
) -> Tuple[float, Optional[int]]:
    """
    Returns (max_disp, atom_index_of_max_or_None_if_no_carbons)
    max_disp computed over atoms where label == 'C' (case-insensitive).
    Displacement is computed after subtracting per-structure centroid (same indices for both).
    """
    if ref.natom != mov.natom:
        raise ValueError(f"natom mismatch: ref={ref.natom} mov={mov.natom}")
    if not np.array_equal(ref.atoms, mov.atoms):
        raise ValueError("Atom labels/order mismatch between reference and moving structure")

    c_mask = np.char.upper(ref.atoms.astype(str)) == "C"
    c_idx = np.nonzero(c_mask)[0]
    if c_idx.size == 0:
        return 0.0, None

    ref_c = ref.xyz - _centroid(ref.xyz, centroid_indices)
    mov_c = mov.xyz - _centroid(mov.xyz, centroid_indices)
    diff = mov_c[c_idx, :] - ref_c[c_idx, :]
    d = np.linalg.norm(diff, axis=1)
    imax = int(np.argmax(d))
    return float(d[imax]), int(c_idx[imax])


def _iter_inputs(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        # bash will expand globs; keep as-is here
        out.append(p)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Detect XYZ outliers by carbon translation relative to a reference, "
            "using centroid-subtracted coordinates (no Kabsch)."
        )
    )
    ap.add_argument("xyz_files", nargs="+", help="Input XYZ files (single-frame).")
    ap.add_argument("--ref", default="", help="Reference XYZ file (default: first input after sorting).")
    ap.add_argument("--cutoff", type=float, default=2.0, help="Outlier cutoff in Angstrom (default: 2.0).")
    ap.add_argument(
        "--centroid-indices",
        default="",
        help='Atom indices to use for centroid subtraction (default: all). Example: "0-13,20".',
    )
    ap.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --centroid-indices as 1-based (e.g. '1-3' means atoms 0..2).",
    )
    ap.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print TSV header line.",
    )
    args = ap.parse_args(argv)

    if args.cutoff < 0:
        raise SystemExit("--cutoff must be >= 0")

    inputs = sorted(_iter_inputs(args.xyz_files))
    if len(inputs) == 0:
        raise SystemExit("No input files")

    ref_path = args.ref.strip() or inputs[0]
    if ref_path not in inputs:
        # allow reference not included in list, but still readable
        if not os.path.exists(ref_path):
            raise SystemExit(f"Reference file {ref_path!r} not found")

    ref = _read_single_xyz(ref_path)
    centroid_indices = _parse_indices(args.centroid_indices, natom=ref.natom, one_based=args.one_based)

    if not args.no_header:
        print("file\tmax_C_disp_A\tis_outlier\tatom_index_of_max")

    warned_no_c = False
    for p in inputs:
        mov = _read_single_xyz(p)
        if mov.natom != ref.natom:
            raise SystemExit(f"natom mismatch: {p} has {mov.natom}, ref has {ref.natom}")
        if not np.array_equal(mov.atoms, ref.atoms):
            raise SystemExit(f"Atom labels/order mismatch in {p} vs reference {ref_path}")

        max_disp, idx = _max_carbon_displacement(ref, mov, centroid_indices=centroid_indices)
        if idx is None and not warned_no_c:
            print("warning\tno carbon atoms found; treating as non-outlier", file=sys.stderr)
            warned_no_c = True

        is_outlier = 1 if max_disp > args.cutoff else 0
        idx_str = "" if idx is None else str(idx)
        print(f"{p}\t{max_disp:.6f}\t{is_outlier}\t{idx_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

