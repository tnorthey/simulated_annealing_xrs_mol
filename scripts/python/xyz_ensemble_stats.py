#!/usr/bin/env python3
"""
Summarize geometry and Kabsch RMSD over a list of XYZ structures.

Reads XYZ files from a bash-expanded list (or an unexpanded glob), computes
chosen bond lengths, dihedrals, and Kabsch RMSD versus a target XYZ, then
writes stats.dat and prints a table with mean, median, and range.

Dihedral summary statistics are circular (angles wrap at ±180°).

Examples:
    python3 scripts/python/xyz_ensemble_stats.py \\
        results_fig3_qmax8_open_phi0p5/fig3_qmax8_open_phi0p5_000.*.xyz \\
        --target results_fig3_qmax8_open_phi0p5/fig3_qmax8_open_phi0p5_target.xyz \\
        --bond 0 5 --dihedral 0 1 4 5 --rmsd-indices 0,1,2,3,4,5

    python3 scripts/python/xyz_ensemble_stats.py \\
        'results_fig3_qmax8_open_phi0p5/*.xyz' \\
        --bond 0 5 --dihedral 0 1 4 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Sequence

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import modules.analysis as analysis  # noqa: E402
from modules.mol import Xyz  # noqa: E402


def expand_xyz_paths(paths: Sequence[str]) -> List[str]:
    """Expand globs and return unique existing XYZ paths, sorted."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in paths:
        matches = sorted(glob.glob(raw)) if any(c in raw for c in "*?[") else [raw]
        if not matches:
            raise SystemExit(f"No files matched {raw!r}")
        for path in matches:
            if not os.path.isfile(path):
                raise SystemExit(f"Not a file: {path}")
            key = os.path.realpath(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    out.sort()
    return out


def infer_target_path(xyz_files: Sequence[str]) -> str | None:
    """Return a unique *_target.xyz beside the inputs, if present."""
    candidates: List[str] = []
    seen: set[str] = set()
    dirs = {os.path.dirname(os.path.abspath(p)) or "." for p in xyz_files}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*_target.xyz")):
            key = os.path.realpath(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    return None


def exclude_target(xyz_files: Sequence[str], target_path: str) -> List[str]:
    target_key = os.path.realpath(target_path)
    return [p for p in xyz_files if os.path.realpath(p) != target_key]


def parse_index_list(expr: str) -> List[int]:
    parts = [p.strip() for p in expr.replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("index list is empty")
    return [int(p) for p in parts]


def read_xyz(path: str) -> tuple[int, np.ndarray, np.ndarray]:
    xyz_util = Xyz()
    natom, _comment, atoms, coords = xyz_util.read_xyz(path)
    atoms = np.asarray(atoms, dtype=str)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Bad xyz shape in {path!r}: {coords.shape}")
    if atoms.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Atom-label / coordinate count mismatch in {path!r}: "
            f"{atoms.shape[0]} vs {coords.shape[0]}"
        )
    if int(natom) != coords.shape[0]:
        raise ValueError(
            f"Header natom {natom} does not match coordinate rows {coords.shape[0]} in {path!r}"
        )
    return int(natom), atoms, coords


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def wrap_delta_deg(delta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def circular_median_deg(angles_deg: np.ndarray) -> float:
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size == 0:
        return float("nan")
    mean = circular_mean_deg(angles)
    delta = wrap_delta_deg(angles - mean)
    med = mean + float(np.median(delta))
    return float(wrap_delta_deg(med))


def circular_min_max_range_deg(angles_deg: np.ndarray) -> tuple[float, float, float]:
    """Smallest-arc min, max (in [-180, 180]), and circular range."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size == 0:
        return float("nan"), float("nan"), float("nan")
    if angles.size == 1:
        a = float(wrap_delta_deg(angles[0]))
        return a, a, 0.0
    wrapped = np.mod(angles, 360.0)
    sorted_a = np.sort(wrapped)
    gaps = np.diff(sorted_a)
    wrap_gap = (sorted_a[0] + 360.0) - sorted_a[-1]
    i_max = int(np.argmax(np.append(gaps, wrap_gap)))
    n = sorted_a.size
    # Endpoints of the smallest arc that covers every angle.
    arc_start = float(sorted_a[(i_max + 1) % n])
    arc_end = float(sorted_a[i_max])
    circ_range = float((arc_end - arc_start) % 360.0)
    amin = float(wrap_delta_deg(arc_start))
    amax = float(wrap_delta_deg(arc_end))
    return amin, amax, circ_range


def linear_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    v = np.asarray(values, dtype=float)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return float(np.mean(v)), float(np.median(v)), vmin, vmax, vmax - vmin


def dihedral_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    v = np.asarray(values, dtype=float)
    mean = circular_mean_deg(v)
    median = circular_median_deg(v)
    vmin, vmax, crange = circular_min_max_range_deg(v)
    return mean, median, vmin, vmax, crange


def column_label_bond(bond: Sequence[int]) -> str:
    return f"bond_{bond[0]}-{bond[1]}"


def column_label_dihedral(dihedral: Sequence[int]) -> str:
    return f"dihedral_{dihedral[0]}-{dihedral[1]}-{dihedral[2]}-{dihedral[3]}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines = [header, rule]
    for row in rows:
        parts = []
        for i, cell in enumerate(row):
            if i == 0:
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        lines.append("  ".join(parts))
    return "\n".join(lines)


def fmt_num(x: float, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def compute_geometry(
    coords: np.ndarray,
    bonds: Sequence[Sequence[int]],
    dihedrals: Sequence[Sequence[int]],
) -> List[float]:
    values: List[float] = []
    for i, j in bonds:
        values.append(float(analysis.calculate_bond_length(coords, i, j)))
    for i, j, k, l in dihedrals:
        values.append(float(analysis.calculate_dihedral(coords, i, j, k, l)))
    return values


def kabsch_rmsd(moving: np.ndarray, target: np.ndarray, indices: Sequence[int]) -> float:
    rmsd, _R = Xyz().rmsd_kabsch(moving, target, list(indices))
    return float(rmsd)


def default_output_path(xyz_files: Sequence[str]) -> str:
    dirs = {os.path.dirname(os.path.abspath(p)) or "." for p in xyz_files}
    if len(dirs) == 1:
        return os.path.join(next(iter(dirs)), "stats.dat")
    return os.path.abspath("stats.dat")


def file_label(path: str, xyz_files: Sequence[str]) -> str:
    base = os.path.basename(path)
    if sum(1 for p in xyz_files if os.path.basename(p) == base) == 1:
        return base
    return path


def write_stats_dat(
    path: str,
    *,
    n_files: int,
    target_path: str,
    rmsd_indices: Sequence[int],
    headers: Sequence[str],
    file_rows: Sequence[Sequence[str]],
    summary_headers: Sequence[str],
    summary_rows: Sequence[Sequence[str]],
) -> None:
    with open(path, "w") as f:
        f.write("# xyz_ensemble_stats\n")
        f.write(f"# n_files = {n_files}\n")
        f.write(f"# target = {target_path}\n")
        f.write("# rmsd_indices = " + ",".join(str(i) for i in rmsd_indices) + "\n")
        f.write("# dihedral mean/median/range are circular\n")
        f.write("# " + "  ".join(headers) + "\n")
        for row in file_rows:
            f.write("  ".join(row) + "\n")
        f.write("\n")
        f.write("# SUMMARY\n")
        f.write("# " + "  ".join(summary_headers) + "\n")
        for row in summary_rows:
            f.write("# " + "  ".join(row) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bond lengths, dihedrals, and Kabsch RMSD for a list of XYZ files, "
            "then write stats.dat and print mean/median/range."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "xyz_files",
        nargs="+",
        help="XYZ files or globs, e.g. results_fig3_qmax8_open_phi0p5/*.xyz",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Target XYZ for Kabsch RMSD (default: unique *_target.xyz in the input directory)",
    )
    parser.add_argument(
        "--bond",
        type=int,
        nargs=2,
        metavar=("I", "J"),
        action="append",
        default=None,
        help="Bond atom indices I J (0-indexed). Repeatable.",
    )
    parser.add_argument(
        "--dihedral",
        type=int,
        nargs=4,
        metavar=("I", "J", "K", "L"),
        action="append",
        default=None,
        help="Dihedral atom indices I J K L (0-indexed). Repeatable.",
    )
    parser.add_argument(
        "--rmsd-indices",
        default="",
        help='Comma-separated atom indices for Kabsch RMSD (default: all). Example: "0,1,2,3,4,5"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output stats.dat path (default: <xyz-dir>/stats.dat)",
    )
    args = parser.parse_args(argv)

    bonds: List[List[int]] = [list(b) for b in (args.bond or [])]
    dihedrals: List[List[int]] = [list(d) for d in (args.dihedral or [])]

    xyz_files = expand_xyz_paths(args.xyz_files)
    target_path = args.target.strip() or infer_target_path(xyz_files)
    if not target_path:
        raise SystemExit(
            "No target XYZ found. Pass --target FILE, or put exactly one *_target.xyz "
            "in the results directory."
        )
    if not os.path.isfile(target_path):
        raise SystemExit(f"Target file not found: {target_path}")

    xyz_files = exclude_target(xyz_files, target_path)
    if not xyz_files:
        raise SystemExit("No XYZ files left after excluding the target")

    n_target, atoms_target, coords_target = read_xyz(target_path)
    if args.rmsd_indices.strip():
        try:
            rmsd_indices = parse_index_list(args.rmsd_indices)
        except ValueError as exc:
            raise SystemExit(f"--rmsd-indices: {exc}") from exc
    else:
        rmsd_indices = list(range(n_target))
    for idx in rmsd_indices:
        if idx < 0 or idx >= n_target:
            raise SystemExit(
                f"RMSD index {idx} out of range for target ({n_target} atoms)"
            )
    if not rmsd_indices:
        raise SystemExit("--rmsd-indices must contain at least one index")

    all_indices = [i for bond in bonds for i in bond] + [
        i for dih in dihedrals for i in dih
    ]
    for idx in all_indices:
        if idx < 0 or idx >= n_target:
            raise SystemExit(
                f"Geometry index {idx} out of range for target ({n_target} atoms)"
            )

    geom_labels = [column_label_bond(b) for b in bonds] + [
        column_label_dihedral(d) for d in dihedrals
    ]
    value_labels = geom_labels + ["rmsd_kabsch"]
    n_geom = len(geom_labels)
    n_dih = len(dihedrals)
    n_bond = len(bonds)

    per_file: List[List[float]] = []
    names: List[str] = []
    for path in xyz_files:
        n_atoms, atoms, coords = read_xyz(path)
        if n_atoms != n_target:
            raise SystemExit(
                f"Atom count mismatch: {path} has {n_atoms}, target has {n_target}"
            )
        if not np.array_equal(atoms, atoms_target):
            raise SystemExit(f"Atom labels/order mismatch in {path} vs {target_path}")
        geom = compute_geometry(coords, bonds, dihedrals)
        rmsd = kabsch_rmsd(coords, coords_target, rmsd_indices)
        per_file.append(geom + [rmsd])
        names.append(file_label(path, xyz_files))

    data = np.asarray(per_file, dtype=float)
    target_geom = compute_geometry(coords_target, bonds, dihedrals)

    file_headers = ["file"] + value_labels
    file_rows: List[List[str]] = []
    for name, row in zip(names, data):
        file_rows.append([name] + [fmt_num(x) for x in row])

    summary_headers = ["quantity", "mean", "median", "min", "max", "range", "target"]
    summary_rows: List[List[str]] = []
    for col, label in enumerate(value_labels):
        col_vals = data[:, col]
        is_dihedral = n_bond <= col < n_bond + n_dih
        if is_dihedral:
            mean, median, vmin, vmax, vrange = dihedral_stats(col_vals)
            target_val = target_geom[col] if col < n_geom else float("nan")
        elif col < n_geom:
            mean, median, vmin, vmax, vrange = linear_stats(col_vals)
            target_val = target_geom[col]
        else:
            mean, median, vmin, vmax, vrange = linear_stats(col_vals)
            target_val = 0.0
        summary_rows.append(
            [
                label,
                fmt_num(mean),
                fmt_num(median),
                fmt_num(vmin),
                fmt_num(vmax),
                fmt_num(vrange),
                fmt_num(target_val),
            ]
        )

    out_path = args.output.strip() or default_output_path(xyz_files)
    write_stats_dat(
        out_path,
        n_files=len(xyz_files),
        target_path=os.path.abspath(target_path),
        rmsd_indices=rmsd_indices,
        headers=file_headers,
        file_rows=file_rows,
        summary_headers=summary_headers,
        summary_rows=summary_rows,
    )

    print(f"N structures: {len(xyz_files)}")
    print(f"Target: {target_path}")
    print(f"RMSD indices: {','.join(str(i) for i in rmsd_indices)}")
    print(f"Wrote {out_path}")
    print()
    print(format_table(file_headers, file_rows))
    print()
    print(format_table(summary_headers, summary_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
