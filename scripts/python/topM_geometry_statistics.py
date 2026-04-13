#!/usr/bin/env python3
"""
topM_geometry_statistics.py

For each timestep, select the topM best-fitting xyz files, compute the
requested geometry (bond / angle / dihedral), and plot the per-timestep
mean and closest-to-mean frame.

The closest-to-mean frame can be selected either by:
  - RMSD to the per-timestep mean structure, or
  - distance in requested geometry space to the per-timestep geometry mean.

No optimal-path solving is performed; this script directly examines
the raw candidates at each timestep.

Dihedral convention: angles are kept as returned by arctan2 ([-180, 180]°), so
crossing through 0 appears as a smooth change into negative values (no +360 jump).
An optional --dihedral-offset is added (no modulo). Optional --dihedral-negate
multiplies the dihedral column by -1 last.
Optional --dihedral-wrap-360 adds 360 to negative dihedral values (after offset/negate),
so output values are non-negative (useful for 0..360 style plots).

Usage:
    python3 topM_geometry_statistics.py results/ --topM 50 --dihedral 2 3 4 5
    python3 topM_geometry_statistics.py results/ --topM 50 --dihedral 2 3 4 5 --dihedral 6 7 8 9
    python3 topM_geometry_statistics.py results/ --topM 100 --dihedral 2 3 4 5 --bond 0 1
    python3 topM_geometry_statistics.py results/ --dihedral 2 3 4 5 --show-individuals
    python3 topM_geometry_statistics.py results/ --dihedral 2 3 4 5 --dihedral-offset 180

Time axis (default matches former hard-coded mapping: 10 + i*20 in fs):
    python3 topM_geometry_statistics.py results/ ... --dt 0.05 --time-units ps --time-origin 0
    # N points from T0 to T1 (e.g. 99 rows → dt_eff = 5/98 ps, not 0.05):
    python3 topM_geometry_statistics.py results/ ... --time-units ps --time-origin -1 --time-end 4
    # Or one time per timestep (same order as sorted frames):
    python3 topM_geometry_statistics.py results/ ... --time-file path/to/time.dat

Uniform grid math: N equally spaced samples from T0 to T1 use
dt_eff = (T1 - T0) / max(N - 1, 1) (e.g. 99 points from -1 ps to 4 ps → dt_eff = 5/98 ps,
not 0.05 ps). Use --time-origin T0 --time-end T1 with --time-basis plot-index to match
that linspace. If physics time is t = -1 + (step-1)*0.05 for 1-based step id, step 99 is
at 3.9 ps unless the schedule uses a different rule. Use --time-basis filename-step so
the axis follows the integer in each filename (e.g. 18_..., 99_...), not the plot row index.

Target vs closest candidate scattering (optional):
    For each timestep the chosen structure is e.g. 18_000.12482393.xyz; the script overlays
    the sibling file 18_000.12482393.dat with TARGET_FUNCTION_18.dat (see --plot-target-comparison).

    python3 topM_geometry_statistics.py results/ --dihedral 2 3 4 5 --plot-target-comparison
    # Optional: --target-results-dir path/to/results
"""

import argparse
import glob
import heapq
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repo root (parent of `modules/`) so the script runs when invoked from any cwd.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import modules.analysis as analysis  # noqa: E402

NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path):
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    t = int(m.group("t"))
    fit = float(m.group("fit"))
    return t, fit


def sorted_timestep_ids_from_directory(directory: str) -> list[int]:
    """Sorted timestep integers from ``<t>_<fit>...`` xyz names (same keys as load_topM_candidates)."""
    xyzs = glob.glob(os.path.join(directory, "*.xyz"))
    ids: set[int] = set()
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            ids.add(parsed[0])
    if not ids:
        raise RuntimeError(
            f"No xyz files with expected naming pattern found in {directory}"
        )
    return sorted(ids)


def load_time_values_from_file(path: str) -> np.ndarray:
    """Load one time per row; first column used if the file has multiple columns. Lines starting with # skipped."""
    try:
        raw = np.loadtxt(path, dtype=np.float64, comments="#")
    except OSError as e:
        raise SystemExit(f"--time-file: cannot read {path!r}: {e}") from e
    except ValueError as e:
        raise SystemExit(
            f"--time-file: failed to parse numbers in {path!r}: {e}\n"
            "Expected one value per line (whitespace-separated), or a single column."
        ) from e
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64))
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def build_time_axis(
    *,
    time_basis: str,
    n: int,
    timestep_ids: Sequence[int],
    dt: float,
    time_origin: float,
    timestep_anchor: int | None,
    time_end: float | None = None,
) -> np.ndarray:
    """Time coordinate for each row: plot row index, linspace endpoints, or filename step ids."""
    if time_basis == "plot-index" and time_end is not None:
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return np.array([float(time_origin)], dtype=np.float64)
        return np.linspace(float(time_origin), float(time_end), n, dtype=np.float64)
    if time_basis == "filename-step":
        if len(timestep_ids) != n:
            raise ValueError("timestep_ids length must match n")
        anchor = float(timestep_anchor if timestep_anchor is not None else timestep_ids[0])
        t = np.asarray(timestep_ids, dtype=np.float64)
        return np.asarray(time_origin, dtype=np.float64) + (t - anchor) * np.asarray(
            dt, dtype=np.float64
        )
    return np.arange(n, dtype=np.float64) * float(dt) + float(time_origin)


def read_xyz_coords(path):
    with open(path, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            xyz[i, 0] = float(parts[1])
            xyz[i, 1] = float(parts[2])
            xyz[i, 2] = float(parts[3])
    return xyz


def read_xyz_frame(path):
    """Read a single-frame xyz file, returning (natoms, comment, atoms, coords)."""
    with open(path, "r") as f:
        n = int(f.readline().strip())
        comment = f.readline().rstrip("\n")
        atoms: list[str] = []
        coords = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            atoms.append(parts[0])
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
            coords[i, 2] = float(parts[3])
    return n, comment, atoms, coords


def write_xyz_trajectory(frames: Sequence[tuple], path: str):
    """Write a multi-frame xyz trajectory.

    frames: list of (natoms, comment, atoms, coords) tuples.
    """
    with open(path, "w") as f:
        for natoms, comment, atoms, coords in frames:
            f.write(f"{natoms}\n")
            f.write(f"{comment}\n")
            for i in range(natoms):
                f.write(
                    f"{atoms[i]:>2s} {coords[i, 0]:16.10f} {coords[i, 1]:16.10f} {coords[i, 2]:16.10f}\n"
                )


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray, indices: Sequence[int] | None = None) -> float:
    """Kabsch-aligned RMSD between two (N,3) coordinate arrays."""
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        P, Q = P[idx], Q[idx]
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    Pr = Pc @ R
    d = Pr - Qc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _plain_rmsd(P: np.ndarray, Q: np.ndarray, indices: Sequence[int] | None = None) -> float:
    """Centroid-aligned RMSD without Kabsch rotation (translation only)."""
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        P, Q = P[idx], Q[idx]
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    d = Pc - Qc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def select_closest_to_mean(
    layer: Sequence[dict],
    *,
    use_kabsch: bool = True,
    rmsd_indices: Sequence[int] | None = None,
) -> int:
    """Return index of frame with minimal RMSD to layer mean structure."""
    K = len(layer)
    if K <= 1:
        return 0

    rmsd_fn = _kabsch_rmsd if use_kabsch else _plain_rmsd
    coords = [read_xyz_coords(c["xyz"]) for c in layer]
    mean_coords = np.mean(np.stack(coords, axis=0), axis=0)
    dists = np.array(
        [rmsd_fn(c, mean_coords, indices=rmsd_indices) for c in coords], dtype=np.float64
    )
    return int(np.argmin(dists))


def select_closest_by_geometry(
    geom: np.ndarray, target: np.ndarray, *, dihedral_cols: Sequence[int] | None = None
) -> int:
    """Return index of row in geom closest to target in geometry space."""
    diff = geom - target[None, :]
    if dihedral_cols is not None:
        # Shortest signed angular differences in degrees (for each dihedral column).
        for dc in dihedral_cols:
            ang = diff[:, dc]
            diff[:, dc] = (ang + 180.0) % 360.0 - 180.0
    d2 = np.sum(diff * diff, axis=1)
    return int(np.argmin(d2))


def load_topM_candidates(directory, topM):
    """Load xyz files grouped by timestep, keeping only the topM lowest-fit per timestep."""
    xyzs = glob.glob(os.path.join(directory, "*.xyz"))

    by_t: dict[int, list[dict]] = {}
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            t, fit = parsed
            by_t.setdefault(t, []).append({"t": t, "fit": fit, "xyz": p})

    if not by_t:
        raise RuntimeError(
            f"No xyz files with expected naming pattern found in {directory}"
        )

    timesteps = sorted(by_t.keys())
    layers = []
    for t in timesteps:
        layer = by_t[t]
        if topM is not None and len(layer) > topM:
            layer = heapq.nsmallest(topM, layer, key=lambda c: c["fit"])
        else:
            layer = sorted(layer, key=lambda c: c["fit"])
        layers.append(layer)

    return timesteps, layers


def compute_geometry_for_layer(
    layer: Sequence[dict],
    bond=None,
    angle=None,
    dihedral=None,
) -> np.ndarray:
    """Compute requested geometry for all candidates in a layer.

    Returns (n_candidates, n_cols) array.
    """
    results = []
    for c in layer:
        xyz = read_xyz_coords(c["xyz"])
        row: list[float] = []
        if bond is not None:
            row.append(analysis.calculate_bond_length(xyz, bond[0], bond[1]))
        if angle is not None:
            row.append(analysis.calculate_angle(xyz, angle[0], angle[1], angle[2]))
        if dihedral is not None:
            # Backwards compatible: a single dihedral tuple/list is accepted,
            # but we also allow a list of dihedrals (each 4 indices).
            if isinstance(dihedral, (list, tuple)) and len(dihedral) == 4 and all(
                isinstance(x, (int, np.integer)) for x in dihedral
            ):
                dihedrals = [dihedral]
            else:
                dihedrals = list(dihedral)
            for d in dihedrals:
                row.append(
                    analysis.calculate_dihedral(xyz, d[0], d[1], d[2], d[3])
                )
        results.append(row)
    return np.array(results, dtype=np.float64)


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular (directional) mean of angles in degrees, principal value in [-180, 180]."""
    rad = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def circular_std_deg(angles_deg: np.ndarray) -> float:
    """Circular standard deviation of angles in degrees.

    Uses sqrt(-2 * ln(R)) where R is the mean resultant length.
    """
    rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.sin(rad)) ** 2 + np.mean(np.cos(rad)) ** 2)
    R = min(R, 1.0)
    if R < 1e-15:
        return 180.0
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def _dihedral_column_indices(
    bond=None, angle=None, dihedral=None
) -> Optional[List[int]]:
    if dihedral is None:
        return None
    base = 0
    if bond is not None:
        base += 1
    if angle is not None:
        base += 1
    n_dihedrals = 1
    if isinstance(dihedral, list) and dihedral and isinstance(dihedral[0], (list, tuple)):
        n_dihedrals = len(dihedral)
    return list(range(base, base + n_dihedrals))


def _column_labels(bond=None, angle=None, dihedral=None) -> List[str]:
    labels: List[str] = []
    if bond is not None:
        labels.append(f"Bond {bond[0] + 1}-{bond[1] + 1} (Å)")
    if angle is not None:
        labels.append(f"Angle {angle[0] + 1}-{angle[1] + 1}-{angle[2] + 1} (°)")
    if dihedral is not None:
        if isinstance(dihedral, list) and dihedral and isinstance(dihedral[0], (list, tuple)):
            dihedrals = dihedral
        else:
            dihedrals = [dihedral]
        for d in dihedrals:
            labels.append(
                f"Dihedral {d[0] + 1}-{d[1] + 1}-{d[2] + 1}-{d[3] + 1} (°)"
            )
    return labels


def _read_scattering_dat(path: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (q, intensity, has_explicit_q). Same semantics as modules.wrap._read_scattering_dat."""
    data = np.loadtxt(path)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        q = np.arange(arr.size, dtype=np.float64)
        intensity = arr.astype(np.float64)
        has_explicit_q = False
    else:
        if arr.shape[1] >= 2:
            q = arr[:, 0].astype(np.float64)
            intensity = arr[:, 1].astype(np.float64)
            has_explicit_q = True
        else:
            q = np.arange(arr.shape[0], dtype=np.float64)
            intensity = arr[:, 0].astype(np.float64)
            has_explicit_q = False
    return q, intensity, has_explicit_q


def _timestep_to_target_run_id(t: int, *, pad: int) -> str:
    """Pad timestep id to match run_id (e.g. 1 -> '01')."""
    return f"{t:0{pad}d}"


def _overlay_on_qgrid(
    q_ref: np.ndarray,
    q_other: np.ndarray,
    y_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q_ref, y_other interpolated onto q_ref)."""
    if q_other.size != q_ref.size or not np.allclose(q_other, q_ref, rtol=1e-5, atol=1e-8):
        y_interp = np.interp(q_ref, q_other, y_other, left=y_other[0], right=y_other[-1])
    else:
        y_interp = y_other
    return q_ref, y_interp


def plot_target_vs_closest_dat_files(
    *,
    args: argparse.Namespace,
    n_timesteps: int,
    closest_dat_paths: list[str],
    plot_stem: str,
    font_scale: float,
) -> None:
    """Per-timestep plots: TARGET_FUNCTION_<ts>.dat vs sibling .dat of the chosen xyz (same basename)."""
    if len(closest_dat_paths) != n_timesteps:
        raise SystemExit(
            f"closest_dat_paths has {len(closest_dat_paths)} entries but "
            f"expected {n_timesteps} (one per timestep)."
        )

    target_dir = args.target_results_dir or args.directory
    parent = os.path.dirname(plot_stem) or "."
    out_subdir = os.path.join(parent, os.path.basename(plot_stem) + "_target_comparison")
    os.makedirs(out_subdir, exist_ok=True)

    pad = max(1, int(args.target_run_id_pad))
    label_fs = 12.0 * font_scale
    tick_fs = 10.0 * font_scale
    legend_fs = 10.0 * font_scale
    title_fs = 14.0 * font_scale

    overlay_series: list[dict] = []

    for ti in range(n_timesteps):
        cand_path = closest_dat_paths[ti]
        parsed = parse_name(cand_path)
        if parsed is None:
            parsed = parse_name(os.path.splitext(cand_path)[0] + ".xyz")
        if parsed is None:
            raise SystemExit(
                f"Could not parse timestep from filename: {cand_path!r} "
                "(expected pattern like 18_000.12345.dat)"
            )
        t_id = parsed[0]
        run_id = _timestep_to_target_run_id(int(t_id), pad=pad) if isinstance(t_id, int) else str(t_id)
        tf_path = os.path.join(target_dir, f"TARGET_FUNCTION_{run_id}.dat")
        if not os.path.isfile(tf_path):
            print(f"Warning: skip timestep {run_id}: missing {tf_path}", file=sys.stderr)
            continue
        if not os.path.isfile(cand_path):
            print(f"Warning: skip timestep {run_id}: missing candidate dat {cand_path}", file=sys.stderr)
            continue

        q_tgt, y_tgt, _ = _read_scattering_dat(tf_path)
        q_c, y_c, _ = _read_scattering_dat(cand_path)

        q_plot, y_c_plot = _overlay_on_qgrid(q_tgt, q_c, y_c)

        overlay_series.append(
            {
                "run_id": run_id,
                "q": q_plot,
                "y_tgt": y_tgt,
                "y_c": y_c_plot,
            }
        )

        base = f"frame_{run_id}_target_vs_closest_dat"
        png_path = os.path.join(out_subdir, base + ".png")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(q_plot, y_tgt, linewidth=2.0, label="TARGET_FUNCTION", color="C0")
        ax.plot(
            q_plot,
            y_c_plot,
            linewidth=2.0,
            linestyle="--",
            label=os.path.basename(cand_path),
            color="C1",
        )
        ax.set_xlabel(r"q (Å$^{-1}$)", fontsize=label_fs)
        ax.set_ylabel("signal", fontsize=label_fs)
        ax.set_title(f"Timestep {run_id}: TARGET_FUNCTION vs {os.path.basename(cand_path)}", fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.legend(fontsize=legend_fs)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png_path}")

    if overlay_series:
        run_numeric: list[float] = []
        for s in overlay_series:
            try:
                run_numeric.append(float(s["run_id"]))
            except ValueError:
                run_numeric.append(float(len(run_numeric)))
        mn, mx = min(run_numeric), max(run_numeric)
        if mx <= mn:
            mx = mn + 1.0
        norm = plt.Normalize(vmin=mn, vmax=mx)
        cmap = plt.cm.viridis
        fig, ax = plt.subplots(figsize=(12, 7))
        for s, rn in zip(overlay_series, run_numeric):
            color = cmap(norm(rn))
            ax.plot(
                s["q"],
                s["y_tgt"],
                color=color,
                linewidth=1.5,
                alpha=0.92,
                linestyle="-",
            )
            ax.plot(
                s["q"],
                s["y_c"],
                color=color,
                linewidth=1.5,
                alpha=0.92,
                linestyle="--",
            )
        ax.set_xlabel(r"q (Å$^{-1}$)", fontsize=label_fs)
        ax.set_ylabel("signal", fontsize=label_fs)
        ax.set_title(
            "All timesteps: TARGET_FUNCTION (solid) vs closest candidate .dat (dashed), color = run id",
            fontsize=title_fs,
        )
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("run id (timestep)", fontsize=label_fs)
        legend_handles = [
            Line2D([0], [0], color="0.2", lw=2.0, linestyle="-", label="TARGET_FUNCTION"),
            Line2D([0], [0], color="0.2", lw=2.0, linestyle="--", label="closest candidate .dat"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=legend_fs)
        plt.tight_layout()
        overlay_png = os.path.join(out_subdir, "all_timesteps_target_overlay.png")
        plt.savefig(overlay_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {overlay_png}")

    print(f"Target comparison plots directory: {out_subdir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot per-timestep mean and closest-to-mean geometry "
            "for the topM best-fitting xyz files."
        ),
    )
    parser.add_argument(
        "directory",
        help="Directory containing *.xyz files (named like 01_000.12345678.xyz)",
    )
    parser.add_argument(
        "--topM",
        type=int,
        default=None,
        help="Keep only the M lowest-fit xyz files per timestep. Default: use all.",
    )
    parser.add_argument(
        "--bond",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "J"),
        help="Bond atom indices I J (0-indexed)",
    )
    parser.add_argument(
        "--angle",
        type=int,
        nargs=3,
        default=None,
        metavar=("I", "J", "K"),
        help="Angle atom indices I J K (0-indexed)",
    )
    parser.add_argument(
        "--dihedral",
        type=int,
        nargs=4,
        action="append",
        default=None,
        metavar=("I", "J", "K", "L"),
        help=(
            "Dihedral atom indices I J K L (0-indexed). "
            "May be provided multiple times to plot multiple dihedrals."
        ),
    )
    parser.add_argument(
        "--dihedral-offset",
        type=float,
        default=0.0,
        metavar="DEG",
        help=(
            "Degrees added to the dihedral column after computation; not folded modulo 360. "
            "Only valid with --dihedral. Changing this uses the same default output names as offset 0; "
            "pass --recompute to refresh a cached CSV."
        ),
    )
    parser.add_argument(
        "--dihedral-negate",
        action="store_true",
        help=(
            "After offset, multiply the dihedral column by -1 (last step). "
            "Only valid with --dihedral."
        ),
    )
    parser.add_argument(
        "--dihedral-wrap-360",
        action="store_true",
        help=(
            "After offset/negate, add 360 to any negative dihedral values "
            "(applies to plotted values + CSV). Only valid with --dihedral."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for all outputs (plot, CSV, xyz trajectories). "
            "Default: current directory."
        ),
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Output PNG filename (placed inside --output-dir). Default: auto-generated.",
    )
    parser.add_argument(
        "--show-individuals",
        action="store_true",
        help="Also plot each individual candidate as a faint line.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "Plot all requested coordinates on one shared axis (overlay), "
            "instead of one subplot per coordinate."
        ),
    )
    parser.add_argument(
        "--no-kabsch",
        action="store_true",
        help=(
            "Use plain centroid-aligned RMSD (no Kabsch rotation) for closest-to-mean selection. "
            "Much faster but assumes structures share a consistent orientation."
        ),
    )
    parser.add_argument(
        "--rmsd-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated atom indices (0-based) for RMSD in closest-to-mean selection "
            "(used when --closest-selection=rmsd). "
            "Example: '0,1,2,3,4,5'. Default: all atoms."
        ),
    )
    parser.add_argument(
        "--closest-selection",
        choices=["rmsd", "geometry"],
        default="rmsd",
        help=(
            "How to choose the per-timestep closest-to-mean representative frame: "
            "'rmsd' uses RMSD to the mean structure, "
            "'geometry' uses distance to mean of requested geometry values."
        ),
    )
    parser.add_argument(
        "--skip-closest",
        action="store_true",
        help=(
            "Skip computing the closest-to-mean representative frame entirely. "
            "The CSV will contain only mean and std columns (no closest_*), and "
            "no closest-to-mean XYZ trajectory / sidecar will be written."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation even if CSV from a previous run exists.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=20.0,
        metavar="T",
        help=(
            "Spacing between consecutive timestep indices on the time axis "
            "(same units as --time-units). Default: 20 (fs with default --time-units)."
        ),
    )
    parser.add_argument(
        "--time-origin",
        type=float,
        default=10.0,
        metavar="T0",
        help=(
            "With --time-basis plot-index: time at row i=0 (start of the axis). "
            "With --time-basis filename-step: time when the filename step id equals "
            "--timestep-anchor (or the smallest step id if unset). "
            "Default: 10 (fs, plot-index), matching the previous hard-coded mapping."
        ),
    )
    parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        metavar="T1",
        help=(
            "Optional end time (same units as --time-units). Only with --time-basis plot-index: "
            "use N equally spaced times from --time-origin to T1 (like numpy.linspace), "
            "so dt_eff = (T1 - T0) / max(N-1, 1). Overrides --dt for the time axis. "
            "Example: 99 frames from -1 ps to 4 ps → --time-units ps --time-origin -1 --time-end 4."
        ),
    )
    parser.add_argument(
        "--time-units",
        type=str,
        default="fs",
        metavar="UNIT",
        help=(
            "Label for the time axis and CSV time column (e.g. fs, ps, ns). "
            "Default: fs. Example for 0.05 ps spacing: --dt 0.05 --time-units ps --time-origin 0"
        ),
    )
    parser.add_argument(
        "--time-basis",
        choices=["plot-index", "filename-step"],
        default="plot-index",
        help=(
            "plot-index: time = --time-origin + i*--dt for row i (0..N-1), ignoring the integer "
            "prefix in filenames. filename-step: time = --time-origin + (t - anchor)*--dt where t "
            "is the prefix (e.g. 18_*.xyz -> 18). Anchor is --timestep-anchor or the smallest t. "
            "Use filename-step to match external data keyed by simulation step id."
        ),
    )
    parser.add_argument(
        "--timestep-anchor",
        type=int,
        default=None,
        metavar="T",
        help=(
            "With --time-basis filename-step: step id at which time equals --time-origin. "
            "Default: smallest timestep id in the xyz directory."
        ),
    )
    parser.add_argument(
        "--time-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Text file with one time value per timestep (same row order as sorted frames: "
            "one number per line, or first column if multiple). Lines starting with # are ignored. "
            "Overrides --dt, --time-end, and --time-basis for the time axis. "
            "Must have exactly as many values as timesteps (N rows in the output CSV)."
        ),
    )
    parser.add_argument(
        "--xmin", type=float, default=None, help="Minimum x-axis value (time in --time-units)."
    )
    parser.add_argument(
        "--xmax", type=float, default=None, help="Maximum x-axis value (time in --time-units)."
    )
    parser.add_argument(
        "--ymin", type=float, default=None, help="Minimum y-axis value."
    )
    parser.add_argument(
        "--ymax", type=float, default=None, help="Maximum y-axis value."
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale factor for plot text sizes (e.g. 2.0 for about double size).",
    )
    parser.add_argument(
        "--plot-target-comparison",
        action="store_true",
        help=(
            "After the geometry plot, write one PNG per timestep overlaying "
            "TARGET_FUNCTION_<run_id>.dat with the sibling .dat of the chosen structure "
            "(e.g. 18_000.12.xyz -> 18_000.12.dat), plus all_timesteps_target_overlay.png "
            "(all TARGET solid + all candidates dashed, color by run id). Requires "
            f"a sidecar file <plot_stem>_closest_dat_paths.txt (written when the "
            "closest-to-mean trajectory is built; use --recompute if missing)."
        ),
    )
    parser.add_argument(
        "--target-results-dir",
        type=str,
        default=None,
        help=(
            "Directory containing TARGET_FUNCTION_*.dat (default: same as the xyz "
            "directory positional argument)."
        ),
    )
    parser.add_argument(
        "--target-run-id-pad",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Zero-pad width for mapping timestep index to TARGET_FUNCTION_<id>.dat "
            "(default 2 → 01, 02, ...)."
        ),
    )

    args = parser.parse_args()

    if args.bond is None and args.angle is None and args.dihedral is None:
        parser.error("At least one of --bond, --angle, or --dihedral must be specified")
    if args.dihedral is None and args.dihedral_offset != 0.0:
        parser.error("--dihedral-offset requires --dihedral")
    if args.dihedral is None and args.dihedral_negate:
        parser.error("--dihedral-negate requires --dihedral")
    if args.dihedral is None and args.dihedral_wrap_360:
        parser.error("--dihedral-wrap-360 requires --dihedral")
    if args.dt <= 0 and not (
        args.time_basis == "plot-index" and args.time_end is not None
    ) and args.time_file is None:
        parser.error("--dt must be positive (or use --time-end with plot-index, or --time-file)")
    if args.time_end is not None and args.time_basis != "plot-index":
        parser.error("--time-end is only valid with --time-basis plot-index")
    if args.time_file is not None and not os.path.isfile(args.time_file):
        parser.error(f"--time-file: not a readable file: {args.time_file!r}")

    rmsd_indices: list[int] | None = None
    if args.rmsd_indices is not None:
        try:
            rmsd_indices = [int(x.strip()) for x in args.rmsd_indices.split(",") if x.strip()]
        except ValueError:
            parser.error("--rmsd-indices must be a comma-separated list of integers")
        if not rmsd_indices:
            parser.error("--rmsd-indices must contain at least one index")

    # --- Resolve output paths early so we can check for cached CSV ---
    if args.output_plot is None:
        parts = ["topM_geometry"]
        if args.bond is not None:
            parts.append(f"bond-{args.bond[0]}-{args.bond[1]}")
        if args.angle is not None:
            parts.append(f"angle-{args.angle[0]}-{args.angle[1]}-{args.angle[2]}")
        if args.dihedral is not None:
            for d in args.dihedral:
                parts.append(f"dihedral-{d[0]}-{d[1]}-{d[2]}-{d[3]}")
        if args.topM is not None:
            parts.append(f"topM-{args.topM}")
        args.output_plot = "_".join(parts) + ".png"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.dirname(args.output_plot) == "":
            args.output_plot = os.path.join(args.output_dir, args.output_plot)

    plot_stem = os.path.splitext(args.output_plot)[0]
    csv_path = plot_stem + ".csv"
    using_geometry_closest = args.closest_selection == "geometry"
    include_closest = not bool(args.skip_closest)
    closest_xyz_path = (
        plot_stem + "_closest_geometry_to_mean.xyz"
        if using_geometry_closest
        else plot_stem + "_closest_to_mean.xyz"
    )
    mean_xyz_path = plot_stem + "_mean.xyz"
    closest_dat_paths_txt = plot_stem + "_closest_dat_paths.txt"
    closest_dat_paths: list[str] = []

    col_labels = _column_labels(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )
    n_cols = len(col_labels)
    dihedral_cols = _dihedral_column_indices(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )

    # --- Try to reuse cached CSV to skip expensive computation ---
    loaded_from_csv = False
    if not args.recompute and os.path.exists(csv_path):
        try:
            raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            expected_cols = 1 + (3 * n_cols if include_closest else 2 * n_cols)
            if raw.shape[1] == expected_cols:
                x = raw[:, 0]
                means = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                stds = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                for ci in range(n_cols):
                    if include_closest:
                        base = 1 + ci * 3
                        means[:, ci] = raw[:, base]
                        # closest[:, ci] loaded below after header check
                        stds[:, ci] = raw[:, base + 2]
                    else:
                        base = 1 + ci * 2
                        means[:, ci] = raw[:, base]
                        stds[:, ci] = raw[:, base + 1]
                n_timesteps = raw.shape[0]
                closest_header_tag = "closest_geom_" if using_geometry_closest else "closest_rmsd_"
                try:
                    with open(csv_path, "r") as f:
                        header_line = f.readline().strip()
                    if include_closest:
                        loaded_from_csv = closest_header_tag in header_line
                    else:
                        loaded_from_csv = closest_header_tag not in header_line
                except OSError:
                    loaded_from_csv = False
                if loaded_from_csv:
                    if include_closest:
                        closest = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                        for ci in range(n_cols):
                            base = 1 + ci * 3
                            closest[:, ci] = raw[:, base + 1]
                    print(
                        f"Loaded cached data from {csv_path} ({n_timesteps} timesteps). "
                        f"Use --recompute to force recalculation."
                    )
                else:
                    print(
                        "Cached CSV does not match requested closest-selection mode "
                        "(or uses legacy headers), recomputing..."
                    )
            else:
                print(f"CSV column count mismatch ({raw.shape[1]} vs expected {expected_cols}), recomputing...")
        except Exception as e:
            print(f"Could not load cached CSV ({e}), recomputing...")

    layers = None
    closest_indices: list[int] = []
    all_geom: list[np.ndarray] = []
    timesteps: list[int] = []

    if not loaded_from_csv:
        print(f"Loading candidates from {args.directory}...")
        timesteps, layers = load_topM_candidates(args.directory, args.topM)

        n_timesteps = len(timesteps)
        layer_sizes = [len(layer) for layer in layers]
        print(
            f"Found {n_timesteps} timesteps, candidates per timestep: "
            f"min={min(layer_sizes)}, max={max(layer_sizes)}, "
            f"topM={'all' if args.topM is None else args.topM}"
        )

        if include_closest:
            if using_geometry_closest:
                print("Computing geometry and selecting geometry-closest-to-mean frames...")
            else:
                print("Computing geometry and selecting RMSD-closest-to-mean frames...")
        else:
            print("Computing geometry statistics (mean/std); skipping closest-to-mean selection...")
        means = np.zeros((n_timesteps, n_cols), dtype=np.float64)
        closest = np.zeros((n_timesteps, n_cols), dtype=np.float64) if include_closest else None
        stds = np.zeros((n_timesteps, n_cols), dtype=np.float64)

        for ti, layer in enumerate(layers):
            pct = (ti + 1) * 100 // n_timesteps
            print(f"\r  Timestep {ti + 1}/{n_timesteps} ({pct}%)", end="", flush=True)
            geom = compute_geometry_for_layer(
                layer, bond=args.bond, angle=args.angle, dihedral=args.dihedral
            )
            # Keep a copy of the raw values for circular statistics before we apply any
            # output transforms (offset/negate/wrap).
            geom_raw = geom.copy()
            all_geom.append(geom)

            for ci in range(n_cols):
                if dihedral_cols is not None and ci in dihedral_cols:
                    # Compute circular statistics on the *raw dihedral* (before offset/negate),
                    # then map back into the stored (offset/negate applied) representation.
                    raw = geom_raw[:, ci]
                    if args.dihedral_negate:
                        wrapped = -raw
                    else:
                        wrapped = raw
                    mean_d = circular_mean_deg(wrapped)
                    # Map mean into the output representation: apply offset/negate/wrap.
                    mean_d = mean_d + args.dihedral_offset
                    if args.dihedral_negate:
                        mean_d = -mean_d
                    if args.dihedral_wrap_360 and mean_d < 0.0:
                        mean_d = mean_d + 360.0
                    means[ti, ci] = mean_d
                    stds[ti, ci] = circular_std_deg(wrapped)
                else:
                    means[ti, ci] = np.mean(geom[:, ci])
                    stds[ti, ci] = np.std(geom[:, ci], ddof=0)

            # Apply output transforms to dihedral columns for storage/plotting/csv/closest selection.
            if dihedral_cols is not None:
                for dc in dihedral_cols:
                    if args.dihedral_offset != 0.0:
                        geom[:, dc] = geom[:, dc] + args.dihedral_offset
                    if args.dihedral_negate:
                        geom[:, dc] *= -1.0
                    if args.dihedral_wrap_360:
                        neg = geom[:, dc] < 0.0
                        if np.any(neg):
                            geom[neg, dc] = geom[neg, dc] + 360.0

            if include_closest:
                if using_geometry_closest:
                    closest_idx = select_closest_by_geometry(
                        geom, means[ti, :], dihedral_cols=dihedral_cols
                    )
                else:
                    closest_idx = select_closest_to_mean(
                        layer, use_kabsch=not args.no_kabsch, rmsd_indices=rmsd_indices
                    )
                closest_indices.append(closest_idx)
                assert closest is not None
                closest[ti, :] = geom[closest_idx, :]

        print()  # finish progress line

        if args.time_file is not None:
            x = load_time_values_from_file(args.time_file)
            if x.size != n_timesteps:
                raise SystemExit(
                    f"--time-file: need {n_timesteps} values (one per timestep), "
                    f"got {x.size} in {args.time_file!r}"
                )
            print(
                f"Time axis: {x.size} values from {args.time_file!r} "
                f"(first/last = {float(x[0]):g} / {float(x[-1]):g} {args.time_units})"
            )
        else:
            x = build_time_axis(
                time_basis=args.time_basis,
                n=n_timesteps,
                timestep_ids=timesteps,
                dt=float(args.dt),
                time_origin=float(args.time_origin),
                timestep_anchor=args.timestep_anchor,
                time_end=args.time_end,
            )
            if args.time_basis == "plot-index" and args.time_end is not None:
                eff = (float(args.time_end) - float(args.time_origin)) / max(n_timesteps - 1, 1)
                print(
                    f"Time axis (linspace): {n_timesteps} points from "
                    f"{float(args.time_origin):g} to {float(args.time_end):g} {args.time_units}, "
                    f"effective dt = {eff:g}"
                )
            elif args.time_basis == "filename-step":
                anchor = args.timestep_anchor if args.timestep_anchor is not None else timesteps[0]
                print(
                    f"Time axis (filename-step): anchor step id={anchor}, "
                    f"first/last time = {float(x[0]):g} / {float(x[-1]):g} {args.time_units}"
                )

        # Write CSV
        time_col = f"time_{args.time_units}"
        header_parts = [time_col]
        closest_prefix = "closest_geom_" if using_geometry_closest else "closest_rmsd_"
        for label in col_labels:
            short = label.split(" (")[0].replace(" ", "_")
            if include_closest:
                header_parts.extend([f"mean_{short}", f"{closest_prefix}{short}", f"std_{short}"])
            else:
                header_parts.extend([f"mean_{short}", f"std_{short}"])

        cols_out = [x]
        for i in range(n_cols):
            if include_closest:
                assert closest is not None
                cols_out.extend([means[:, i], closest[:, i], stds[:, i]])
            else:
                cols_out.extend([means[:, i], stds[:, i]])
        out_data = np.column_stack(cols_out)
        np.savetxt(
            csv_path,
            out_data,
            delimiter=",",
            header=",".join(header_parts),
            comments="",
            fmt="%.10g",
        )
        print(f"CSV written to: {csv_path}")

        # Write closest-to-mean and mean xyz trajectories
        if include_closest:
            print("Writing closest-to-mean trajectory...")
            closest_frames = []
            for ti, layer_i in enumerate(layers):
                mi = closest_indices[ti]
                xyz_path = layer_i[mi]["xyz"]
                closest_dat_paths.append(
                    os.path.abspath(os.path.splitext(xyz_path)[0] + ".dat")
                )
                n, comment, atoms, coords = read_xyz_frame(xyz_path)
                sel_tag = "closest-geometry-to-mean" if using_geometry_closest else "closest-rmsd-to-mean"
                comment = f"{comment} | {sel_tag} | timestep={timesteps[ti]}"
                closest_frames.append((n, comment, atoms, coords))
            write_xyz_trajectory(closest_frames, closest_xyz_path)
            print(f"Closest-to-mean trajectory written to: {closest_xyz_path}")
            with open(closest_dat_paths_txt, "w", encoding="utf-8") as f:
                for dp in closest_dat_paths:
                    f.write(dp + "\n")
            print(f"Wrote {closest_dat_paths_txt} (sibling .dat paths for each chosen xyz)")

        print("Writing mean trajectory...")
        mean_frames = []
        for ti, layer_i in enumerate(layers):
            all_coords = []
            atoms_ref = None
            natoms_ref = None
            for c in layer_i:
                n, _comment, atoms, coords = read_xyz_frame(c["xyz"])
                all_coords.append(coords)
                if atoms_ref is None:
                    atoms_ref = atoms
                    natoms_ref = n
            mean_coords = np.mean(np.stack(all_coords, axis=0), axis=0)
            comment = f"mean over {len(layer_i)} candidates | timestep={timesteps[ti]}"
            mean_frames.append((natoms_ref, comment, atoms_ref, mean_coords))
        write_xyz_trajectory(mean_frames, mean_xyz_path)
        print(f"Mean trajectory written to: {mean_xyz_path}")

    if loaded_from_csv and args.time_basis == "filename-step":
        t_ids = sorted_timestep_ids_from_directory(args.directory)
        if len(t_ids) != n_timesteps:
            raise SystemExit(
                f"--time-basis filename-step: found {len(t_ids)} timestep ids under "
                f"{args.directory!r} but CSV has {n_timesteps} rows. "
                "Use --recompute after changing inputs, or fix the directory."
            )
        x = build_time_axis(
            time_basis=args.time_basis,
            n=n_timesteps,
            timestep_ids=t_ids,
            dt=float(args.dt),
            time_origin=float(args.time_origin),
            timestep_anchor=args.timestep_anchor,
            time_end=None,
        )
        anchor = args.timestep_anchor if args.timestep_anchor is not None else t_ids[0]
        print(
            f"Time axis (filename-step, from directory): anchor step id={anchor}, "
            f"first/last time = {float(x[0]):g} / {float(x[-1]):g} {args.time_units}"
        )

    if loaded_from_csv and args.time_basis == "plot-index" and args.time_end is not None:
        x = build_time_axis(
            time_basis=args.time_basis,
            n=n_timesteps,
            timestep_ids=timesteps,
            dt=float(args.dt),
            time_origin=float(args.time_origin),
            timestep_anchor=args.timestep_anchor,
            time_end=args.time_end,
        )
        eff = (float(args.time_end) - float(args.time_origin)) / max(n_timesteps - 1, 1)
        print(
            f"Time axis (linspace, overriding CSV times): {n_timesteps} points "
            f"{float(args.time_origin):g} … {float(args.time_end):g} {args.time_units}, "
            f"effective dt = {eff:g}"
        )

    if args.time_file is not None and loaded_from_csv:
        tx = load_time_values_from_file(args.time_file)
        if tx.size != n_timesteps:
            raise SystemExit(
                f"--time-file: need {n_timesteps} values (one per timestep), "
                f"got {tx.size} in {args.time_file!r}"
            )
        x = tx
        print(
            f"Time axis: {tx.size} values from {args.time_file!r} (overriding CSV times), "
            f"first/last = {float(x[0]):g} / {float(x[-1]):g} {args.time_units}"
        )

    # --- Plot (always runs, using either fresh or cached data) ---
    print("Creating plot...")
    font_scale = float(args.font_scale)
    if font_scale <= 0:
        parser.error("--font-scale must be > 0")
    title_fs = 16.0 * font_scale
    label_fs = 12.0 * font_scale
    tick_fs = 10.0 * font_scale
    legend_fs = 10.0 * font_scale

    if n_cols == 1 or args.overlay:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3.2 * n_cols), sharex=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

    closest_plot_label = (
        "closest-geometry-to-mean" if using_geometry_closest else "closest-rmsd-to-mean"
    )

    if args.overlay and n_cols > 1:
        ax = axes[0]
        # Use consistent per-coordinate colors.
        for i in range(n_cols):
            color = f"C{i % 10}"

            if args.show_individuals and all_geom:
                max_candidates = max(g.shape[0] for g in all_geom)
                for ci in range(max_candidates):
                    ys = []
                    xs = []
                    for ti in range(n_timesteps):
                        if ci < all_geom[ti].shape[0]:
                            xs.append(x[ti])
                            ys.append(all_geom[ti][ci, i])
                    ax.plot(xs, ys, linewidth=0.25, alpha=0.06, color=color)

            ax.plot(
                x,
                means[:, i],
                linewidth=2,
                label=f"mean: {col_labels[i]}",
                color=color,
            )
            ax.fill_between(
                x,
                means[:, i] - stds[:, i],
                means[:, i] + stds[:, i],
                alpha=0.12,
                color=color,
                linewidth=0,
            )
            if include_closest:
                ax.plot(
                    x,
                    closest[:, i],
                    linewidth=2,
                    label=f"{closest_plot_label}: {col_labels[i]}",
                    color=color,
                    linestyle="--",
                    alpha=0.95,
                )

        ax.set_ylabel("value", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=legend_fs)
    else:
        for i in range(n_cols):
            ax = axes[i]

            if args.show_individuals and all_geom:
                max_candidates = max(g.shape[0] for g in all_geom)
                for ci in range(max_candidates):
                    ys = []
                    xs = []
                    for ti in range(n_timesteps):
                        if ci < all_geom[ti].shape[0]:
                            xs.append(x[ti])
                            ys.append(all_geom[ti][ci, i])
                    ax.plot(xs, ys, linewidth=0.3, alpha=0.15, color="grey")

            ax.plot(x, means[:, i], linewidth=2, label="mean", color="C0")
            ax.fill_between(
                x,
                means[:, i] - stds[:, i],
                means[:, i] + stds[:, i],
                alpha=0.2,
                color="C0",
                label="±1σ",
            )
            if include_closest:
                ax.plot(
                    x,
                    closest[:, i],
                    linewidth=2,
                    label=closest_plot_label,
                    color="C1",
                    linestyle="--",
                )
            ax.set_ylabel(col_labels[i], fontsize=label_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=legend_fs)

    axes[-1].set_xlabel(f"time ({args.time_units})", fontsize=label_fs)
    xleft = 0.0 if args.xmin is None else float(args.xmin)
    xright = None if args.xmax is None else float(args.xmax)
    for ax in axes:
        ax.set_xlim(left=xleft, right=xright)
        if args.ymin is not None or args.ymax is not None:
            ax.set_ylim(bottom=args.ymin, top=args.ymax)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {args.output_plot}")

    if args.plot_target_comparison:
        if not include_closest:
            raise SystemExit("--plot-target-comparison requires closest-to-mean selection (omit --skip-closest).")
        if not closest_dat_paths:
            if os.path.isfile(closest_dat_paths_txt):
                with open(closest_dat_paths_txt, encoding="utf-8") as f:
                    closest_dat_paths = [line.strip() for line in f if line.strip()]
        if len(closest_dat_paths) != n_timesteps:
            raise SystemExit(
                f"--plot-target-comparison needs {closest_dat_paths_txt} with exactly "
                f"{n_timesteps} lines (paths to candidate .dat files). "
                "Re-run without a stale CSV cache or pass --recompute to rebuild the "
                "closest-to-mean trajectory and sidecar."
            )
        plot_target_vs_closest_dat_files(
            args=args,
            n_timesteps=n_timesteps,
            closest_dat_paths=closest_dat_paths,
            plot_stem=plot_stem,
            font_scale=float(args.font_scale),
        )


if __name__ == "__main__":
    main()
