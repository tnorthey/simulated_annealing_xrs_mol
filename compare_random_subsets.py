#!/usr/bin/env python3
"""
compare_random_subsets.py

Runs optimal_path.py multiple times with different random subsets,
analyzes dihedral angles, and creates comparison plots.

Usage:
    python3 compare_random_subsets.py results/ [--n-subsets N] [--seed-start SEED]
"""

import argparse
import subprocess
import os
import sys
import tempfile
import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def run_optimal_path(
    directory,
    seed,
    output_dir,
    subset_idx,
    random_sample,
    topM,
    *,
    no_autoscale: bool,
    sample_after_prune: bool,
):
    """Run optimal_path.py with specified parameters."""
    output_xyz = os.path.join(output_dir, f"optimal_trajectory_subset_{subset_idx}.xyz")
    
    cmd = [
        "python3", "optimal_path.py", directory,
        "--random-sample", str(random_sample),
        "--topM", str(topM),
        "--fit-weight", "1",
        "--rmsd-weight", "1",
        "--rmsd-indices", "0,1,2,3,4,5",
        "--signal-weight", "1",
        "--seed", str(seed),
        "--xyz-out", output_xyz,
    ]
    # Default behavior (historical): random-sample first, then topM.
    # Optionally allow the alternative: topM first, then random-sample from that pool.
    if sample_after_prune:
        cmd.append("--sample-after-prune")
    if no_autoscale:
        cmd.append("--no-autoscale")
    
    print(f"\n{'='*60}")
    print(f"Running subset {subset_idx} with seed {seed}...")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running optimal_path.py:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    return output_xyz


def average_optimal_trajectories(
    xyz_files: Sequence[str],
    output_xyz: str,
    align: str = "kabsch",
    stat: str = "mean",
) -> bool:
    """
    Average multiple optimal-path XYZ trajectories into a mean trajectory.

    Uses the standalone average_xyz.py tool to:
      - average frame-by-frame across inputs (--per-frame)
      - optionally align each frame before averaging (--align kabsch)
    """
    if len(xyz_files) < 1:
        raise ValueError("No XYZ files provided for averaging.")

    cmd = [
        "python3",
        "average_xyz.py",
        *list(xyz_files),
        "--per-frame",
        "--align",
        align,
        "--stat",
        stat,
        "-o",
        output_xyz,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running average_xyz.py:")
        print(result.stderr)
        return False
    return True


def _read_xyz_trajectory_frames(path: str):
    """
    Read a multi-frame XYZ trajectory file.

    Returns: list of dicts with keys:
      - natoms (int)
      - comment (str)
      - body_lines (list[str])  # natoms lines including trailing newlines
      - coords (np.ndarray)     # (natoms, 3) float64
    """
    frames = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError:
                break
            comment = f.readline().rstrip("\n")
            body = []
            coords = np.zeros((natoms, 3), dtype=np.float64)
            for i in range(natoms):
                row = f.readline()
                if not row:
                    break
                body.append(row)  # preserve exact formatting
                parts = row.split()
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
                coords[i, 2] = float(parts[3])
            if len(body) != natoms:
                break
            frames.append(
                {
                    "natoms": natoms,
                    "comment": comment,
                    "body_lines": body,
                    "coords": coords,
                }
            )
    if not frames:
        raise ValueError(f"No frames found in XYZ trajectory: {path}")
    return frames


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two structures P and Q (N,3)."""
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


def write_median_trajectory_as_medoid(
    xyz_files: Sequence[str],
    output_xyz: str,
    *,
    smooth_weight: float = 0.0,
    label_suffix: str = "median=medoid",
) -> bool:
    """
    Write a "median" trajectory where each frame is an ACTUAL frame chosen from the inputs.

    For each frame index, we select the medoid across subsets: the input frame whose
    sum of Kabsch-RMSD distances to all other subset frames is minimal.

    This guarantees every output frame corresponds exactly to one of the subset trajectories
    (and therefore to an actual candidate chosen by optimal_path).
    """
    if len(xyz_files) < 1:
        raise ValueError("No XYZ files provided for median/medoid selection.")

    # Read all trajectories
    trajs = [_read_xyz_trajectory_frames(p) for p in xyz_files]
    min_frames = min(len(t) for t in trajs)
    if any(len(t) != min_frames for t in trajs):
        print(
            f"Warning: Not all subset trajectories have the same number of frames. "
            f"Truncating to min_frames={min_frames}."
        )

    # Validate natoms consistency
    natoms0 = trajs[0][0]["natoms"]
    for ti, tr in enumerate(trajs):
        for fi in range(min_frames):
            if tr[fi]["natoms"] != natoms0:
                raise ValueError(
                    f"Subset {ti} frame {fi} natoms={tr[fi]['natoms']} differs from first natoms={natoms0}"
                )

    # Precompute node costs: sum RMSD to all other subsets at the same frame
    K = len(trajs)
    node_cost = np.zeros((min_frames, K), dtype=np.float64)
    coords_by_frame = [[trajs[k][fi]["coords"] for k in range(K)] for fi in range(min_frames)]
    for fi in range(min_frames):
        coords_list = coords_by_frame[fi]
        for i in range(K):
            si = 0.0
            Pi = coords_list[i]
            for j in range(K):
                if i == j:
                    continue
                si += _kabsch_rmsd(Pi, coords_list[j])
            node_cost[fi, i] = si

    # If smooth_weight == 0, pick per-frame medoid (historical behavior)
    if smooth_weight <= 0.0:
        chosen_subset_indices = [int(np.argmin(node_cost[fi, :])) for fi in range(min_frames)]
    else:
        # DP: choose sequence minimizing node_cost + smooth_weight * transition_rmsd
        dp = np.full((min_frames, K), np.inf, dtype=np.float64)
        prev = np.full((min_frames, K), -1, dtype=np.int32)
        dp[0, :] = node_cost[0, :]

        for fi in range(1, min_frames):
            curr_coords = coords_by_frame[fi]
            prev_coords = coords_by_frame[fi - 1]
            for j in range(K):
                best_val = np.inf
                best_i = -1
                Pj = curr_coords[j]
                for i in range(K):
                    trans = smooth_weight * _kabsch_rmsd(prev_coords[i], Pj)
                    val = dp[fi - 1, i] + trans + node_cost[fi, j]
                    if val < best_val:
                        best_val = val
                        best_i = i
                dp[fi, j] = best_val
                prev[fi, j] = best_i

        # backtrack
        last = int(np.argmin(dp[min_frames - 1, :]))
        chosen_subset_indices = [last]
        for fi in range(min_frames - 1, 0, -1):
            last = int(prev[fi, last])
            chosen_subset_indices.append(last)
        chosen_subset_indices.reverse()

    # Write selected frames
    with open(output_xyz, "w") as out:
        for fi, best_i in enumerate(chosen_subset_indices):
            frame = trajs[best_i][fi]
            comment = frame["comment"]
            if label_suffix:
                comment = f"{comment} | {label_suffix} | medoid_subset={best_i}"
            if smooth_weight > 0.0:
                comment = f"{comment} | smooth_weight={smooth_weight:g}"

            out.write(f"{frame['natoms']}\n")
            out.write(f"{comment}\n")
            for line in frame["body_lines"]:
                out.write(line)

    print(
        f"Medoid-median trajectory written to {output_xyz} "
        f"(frames={min_frames}, subsets={len(trajs)})."
    )
    return True


def write_closest_to_mean_trajectory_per_frame(
    xyz_files: Sequence[str],
    mean_xyz: str,
    output_xyz: str,
    *,
    label_suffix: str = "closest_to_mean",
) -> bool:
    """
    Write a "closest-to-mean" trajectory where each frame is an ACTUAL frame chosen from the inputs.

    For each frame index, select the subset frame (from xyz_files) that minimizes Kabsch-RMSD
    to the mean trajectory frame (from mean_xyz).

    This allows choosing different subsets per frame (unlike the representative-subset method).
    """
    if len(xyz_files) < 1:
        raise ValueError("No XYZ files provided for closest-to-mean selection.")

    trajs = [_read_xyz_trajectory_frames(p) for p in xyz_files]
    mean_traj = _read_xyz_trajectory_frames(mean_xyz)

    min_frames = min(min(len(t) for t in trajs), len(mean_traj))
    if any(len(t) != min_frames for t in trajs) or len(mean_traj) != min_frames:
        print(
            "Warning: Not all trajectories have the same number of frames. "
            f"Truncating to min_frames={min_frames}."
        )

    natoms0 = trajs[0][0]["natoms"]
    for ti, tr in enumerate(trajs):
        for fi in range(min_frames):
            if tr[fi]["natoms"] != natoms0:
                raise ValueError(
                    f"Subset {ti} frame {fi} natoms={tr[fi]['natoms']} differs from first natoms={natoms0}"
                )
    for fi in range(min_frames):
        if mean_traj[fi]["natoms"] != natoms0:
            raise ValueError(
                f"Mean trajectory frame {fi} natoms={mean_traj[fi]['natoms']} differs from subsets natoms={natoms0}"
            )

    with open(output_xyz, "w") as out:
        for fi in range(min_frames):
            target = mean_traj[fi]["coords"]
            best_i = -1
            best_d = float("inf")
            for si, tr in enumerate(trajs):
                d = _kabsch_rmsd(tr[fi]["coords"], target)
                if d < best_d:
                    best_d = d
                    best_i = si

            frame = trajs[best_i][fi]
            comment = frame["comment"]
            if label_suffix:
                comment = f"{comment} | {label_suffix} | closest_subset={best_i} | rmsd_to_mean={best_d:.6g}"
            out.write(f"{frame['natoms']}\n")
            out.write(f"{comment}\n")
            for line in frame["body_lines"]:
                out.write(line)

    print(
        f"Closest-to-mean trajectory written to {output_xyz} "
        f"(frames={min_frames}, subsets={len(trajs)})."
    )
    return True


def analyze_geometry(xyz_file, output_csv, bond=None, angle=None, dihedral=None):
    """Run analyze_geometry.py to extract specified geometric parameters."""
    cmd = ["python3", "analyze_geometry.py", xyz_file]
    
    if bond is not None:
        cmd.extend(["--bond", str(bond[0]), str(bond[1])])
    if angle is not None:
        cmd.extend(["--angle", str(angle[0]), str(angle[1]), str(angle[2])])
    if dihedral is not None:
        cmd.extend(["--dihedral", str(dihedral[0]), str(dihedral[1]), 
                    str(dihedral[2]), str(dihedral[3])])
    
    cmd.extend(["--output", output_csv])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running analyze_geometry.py:")
        print(result.stderr)
        return False
    
    return True


def _read_csv_no_header(path: str) -> np.ndarray:
    """Read analyze_geometry.py output (numeric CSV, no header). Returns (n_frames, n_cols)."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data.astype(np.float64)


def _calc_column_labels(bond=None, angle=None, dihedral=None) -> List[str]:
    """Column order must match analyze_geometry.py: bonds first, then angles, then dihedrals."""
    labels: List[str] = []
    if bond is not None:
        labels.append(f"Bond {bond[0]}-{bond[1]} (Å)")
    if angle is not None:
        labels.append(f"Angle {angle[0]}-{angle[1]}-{angle[2]} (°)")
    if dihedral is not None:
        labels.append(f"Dihedral {dihedral[0]}-{dihedral[1]}-{dihedral[2]}-{dihedral[3]} (°)")
    return labels


def plot_aggregate_mean_std(
    csv_files: Sequence[str],
    output_png: str,
    random_sample: int,
    topM: int,
    bond=None,
    angle=None,
    dihedral=None,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    *,
    medoid_csv: str | None = None,
):
    """
    Plot mean ± std-dev across all subsets, per frame, for each requested column.
    Uses time axis: frame i at (10 + 20*i) fs, and starts x-axis at 0 for blank space.
    """
    arrays = [_read_csv_no_header(p) for p in csv_files]

    n_cols = arrays[0].shape[1]
    if any(a.shape[1] != n_cols for a in arrays):
        raise ValueError("Not all geometry CSV files have the same number of columns.")

    min_frames = min(a.shape[0] for a in arrays)
    if any(a.shape[0] != min_frames for a in arrays):
        print(
            f"Warning: Not all subsets have the same number of frames. "
            f"Truncating to min_frames={min_frames}."
        )

    Y = np.stack([a[:min_frames, :] for a in arrays], axis=0)  # (n_subsets, n_frames, n_cols)
    mean = np.mean(Y, axis=0)  # (n_frames, n_cols)
    std = np.std(Y, axis=0, ddof=0)  # population std-dev

    # Time axis: first frame at 10 fs, step 20 fs
    x = np.arange(min_frames, dtype=np.float64) * 20.0 + 10.0

    col_labels = _calc_column_labels(bond=bond, angle=angle, dihedral=dihedral)
    if len(col_labels) != n_cols:
        # Fallback if something unexpected happens
        col_labels = [f"Column {i}" for i in range(n_cols)]

    # Optional: overlay medoid trajectory curve(s) (must match columns requested).
    medoid_arr = None
    if medoid_csv is not None:
        medoid_arr = _read_csv_no_header(medoid_csv)
        if medoid_arr.shape[1] != n_cols:
            raise ValueError(
                f"Medoid CSV has {medoid_arr.shape[1]} columns but expected {n_cols} "
                f"(must match analyze_geometry.py output for requested bond/angle/dihedral)."
            )
        if medoid_arr.shape[0] < min_frames:
            min_frames = medoid_arr.shape[0]
            Y = Y[:, :min_frames, :]
            mean = mean[:min_frames, :]
            std = std[:min_frames, :]
            x = np.arange(min_frames, dtype=np.float64) * 20.0 + 10.0
        medoid_arr = medoid_arr[:min_frames, :]

    if n_cols == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        if medoid_arr is not None:
            ax.plot(x, medoid_arr[:, 0], linewidth=2.5, label="smooth-medoid")
            ax.fill_between(x, mean[:, 0] - std[:, 0], mean[:, 0] + std[:, 0], alpha=0.20, label="±1σ (subsets)")
        else:
            ax.plot(x, mean[:, 0], linewidth=2, label="mean")
            ax.fill_between(x, mean[:, 0] - std[:, 0], mean[:, 0] + std[:, 0], alpha=0.25, label="±1σ")
        ax.set_ylabel(col_labels[0])
        ax.legend()
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3.2 * n_cols), sharex=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for i in range(n_cols):
            ax = axes[i]
            if medoid_arr is not None:
                ax.plot(x, medoid_arr[:, i], linewidth=2.5, label="smooth-medoid")
                ax.fill_between(x, mean[:, i] - std[:, i], mean[:, i] + std[:, i], alpha=0.20, label="±1σ (subsets)")
            else:
                ax.plot(x, mean[:, i], linewidth=2, label="mean")
                ax.fill_between(x, mean[:, i] - std[:, i], mean[:, i] + std[:, i], alpha=0.25, label="±1σ")
            ax.set_ylabel(col_labels[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

    title_parts = []
    if bond is not None:
        title_parts.append(f"Bond {bond[0]}-{bond[1]}")
    if angle is not None:
        title_parts.append(f"Angle {angle[0]}-{angle[1]}-{angle[2]}")
    if dihedral is not None:
        title_parts.append(f"Dihedral {dihedral[0]}-{dihedral[1]}-{dihedral[2]}-{dihedral[3]}")
    title = " / ".join(title_parts) if title_parts else "Geometry"
    fig.suptitle(
        (
            f"{title} — smooth-medoid + ±std across {len(csv_files)} subsets "
            f"(random-sample={random_sample}, topM={topM})"
            if medoid_arr is not None
            else f"{title} — mean ± std across {len(csv_files)} subsets (random-sample={random_sample}, topM={topM})"
        )
    )

    # X axis
    axes[-1].set_xlabel("time (fs)")
    # start at 0 to show blank before first frame at 10 fs (unless user overrides)
    xleft = 0.0 if xmin is None else float(xmin)
    xright = None if xmax is None else float(xmax)
    for ax in axes:
        ax.set_xlim(left=xleft, right=xright)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAggregate plot saved to: {output_png}")


def _selected_column_indices(bond=None, angle=None, dihedral=None) -> List[int]:
    """Return column indices in analyze_geometry.py output for requested calculations."""
    idxs: List[int] = []
    col_idx = 0
    if bond is not None:
        idxs.append(col_idx)
        col_idx += 1
    if angle is not None:
        idxs.append(col_idx)
        col_idx += 1
    if dihedral is not None:
        idxs.append(col_idx)
        col_idx += 1
    return idxs


def _dihedral_column_index(bond=None, angle=None, dihedral=None) -> Optional[int]:
    """Return the dihedral column index if dihedral is requested; otherwise None."""
    if dihedral is None:
        return None
    idx = 0
    if bond is not None:
        idx += 1
    if angle is not None:
        idx += 1
    return idx


def select_representative_subset(
    records: Sequence[dict],
    bond=None,
    angle=None,
    dihedral=None,
) -> dict:
    """
    Pick the subset whose trajectory is closest to the mean (RMS over frames).

    If a dihedral is requested, closeness is computed using ONLY the dihedral column.
    Otherwise, closeness is computed using ALL requested columns jointly.
    """
    if len(records) == 0:
        raise ValueError("No subset records available for representative selection.")

    arrays = []
    for r in records:
        arr = _read_csv_no_header(r["csv_file"])
        arrays.append(arr)

    n_cols = arrays[0].shape[1]
    if any(a.shape[1] != n_cols for a in arrays):
        raise ValueError("Not all geometry CSV files have the same number of columns.")

    min_frames = min(a.shape[0] for a in arrays)
    Y = np.stack([a[:min_frames, :] for a in arrays], axis=0)  # (n_subsets, n_frames, n_cols)
    mean = np.mean(Y, axis=0)  # (n_frames, n_cols)

    dcol = _dihedral_column_index(bond=bond, angle=angle, dihedral=dihedral)
    if dcol is not None:
        diffs = Y[:, :, dcol] - mean[:, dcol]  # (n_subsets, n_frames)
        rms = np.sqrt(np.mean(diffs * diffs, axis=1))  # (n_subsets,)
    else:
        cols = _selected_column_indices(bond=bond, angle=angle, dihedral=dihedral)
        if len(cols) == 0:
            raise ValueError("No geometry columns selected.")
        diffs = Y[:, :, cols] - mean[:, cols]  # (n_subsets, n_frames, n_cols_sel)
        rms = np.sqrt(np.mean(diffs * diffs, axis=(1, 2)))  # (n_subsets,)

    best_i = int(np.argmin(rms))
    best = dict(records[best_i])
    best["representative_rms_to_mean"] = float(rms[best_i])
    best["representative_min_frames_used"] = int(min_frames)
    return best


def plot_comparison(
    csv_files,
    labels,
    output_png,
    random_sample,
    bond=None,
    angle=None,
    dihedral=None,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Run plot_geometry.py to create comparison plot."""
    # Build title and ylabel based on what's being plotted
    parts = []
    ylabel_parts = []
    columns = []
    col_idx = 0
    
    calc_types = []
    if bond is not None:
        parts.append(f"Bond {bond[0]}-{bond[1]}")
        ylabel_parts.append("Bond Length (Å)")
        columns.append(str(col_idx))
        calc_types.append("bond")
        col_idx += 1
    if angle is not None:
        parts.append(f"Angle {angle[0]}-{angle[1]}-{angle[2]}")
        ylabel_parts.append("Angle (°)")
        columns.append(str(col_idx))
        calc_types.append("angle")
        col_idx += 1
    if dihedral is not None:
        parts.append(f"Dihedral {dihedral[0]}-{dihedral[1]}-{dihedral[2]}-{dihedral[3]}")
        ylabel_parts.append("Dihedral Angle (°)")
        columns.append(str(col_idx))
        calc_types.append("dihedral")
        col_idx += 1
    
    title = f"{' / '.join(parts)} Comparison (Random Subsets of {random_sample})"
    ylabel = " / ".join(ylabel_parts) if ylabel_parts else "Value"
    
    # Build labels for each file-column combination
    # If multiple columns, each file needs labels for each column
    plot_labels = []
    if len(columns) == 1:
        # Single column: use subset labels as-is
        plot_labels = labels
    else:
        # Multiple columns: create labels like "Subset 0 (seed=0) - Bond"
        for label in labels:
            for calc_type in calc_types:
                calc_name = calc_type.capitalize()
                plot_labels.append(f"{label} - {calc_name}")
    
    cmd = [
        "python3", "plot_geometry.py",
    ] + csv_files + [
        output_png,
        "--columns",
    ] + columns + [
        "--labels",
    ] + plot_labels + [
        "--title", title,
        "--ylabel", ylabel,
        "--grid",
    ]

    # Axis limit pass-through (plot_geometry.py supports these)
    if xmin is not None:
        cmd.extend(["--xmin", str(xmin)])
    if xmax is not None:
        cmd.extend(["--xmax", str(xmax)])
    if ymin is not None:
        cmd.extend(["--ymin", str(ymin)])
    if ymax is not None:
        cmd.extend(["--ymax", str(ymax)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running plot_geometry.py:")
        print(result.stderr)
        return False
    
    print(f"\nPlot saved to: {output_png}")
    return True


def _sanitize_filename_component(s: str) -> str:
    # Keep filenames portable: alnum + a few safe separators
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    # collapse repeated underscores
    cleaned = "".join(out)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _auto_output_plot_name(args: argparse.Namespace) -> str:
    parts: List[str] = []

    # geometry selection
    if args.bond is not None:
        parts.append(f"bond-{args.bond[0]}-{args.bond[1]}")
    if args.angle is not None:
        parts.append(f"angle-{args.angle[0]}-{args.angle[1]}-{args.angle[2]}")
    if args.dihedral is not None:
        parts.append(
            f"dihedral-{args.dihedral[0]}-{args.dihedral[1]}-{args.dihedral[2]}-{args.dihedral[3]}"
        )

    # run parameters
    parts.append(f"nsub-{args.n_subsets}")
    parts.append(f"rs-{args.random_sample}")
    parts.append(f"topM-{args.topM}")
    parts.append(f"seed-{args.seed_start}")
    parts.append("agg" if args.aggregate else "all")

    stem = "_".join(_sanitize_filename_component(p) for p in parts if p)
    if not stem:
        stem = "geometry_comparison"
    return f"{stem}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple random subsets using optimal_path, analyze geometry, and plot results."
    )
    default_output_plot = "geometry_comparison.png"
    parser.add_argument(
        "directory",
        help="Directory containing *.dat and *.xyz files for optimal_path.py",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=5,
        help="Number of random subsets to compare (default: 5)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for random subsets (default: 0). Each subset uses seed_start + subset_index.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="subset_comparison",
        help="Output directory for intermediate files (default: subset_comparison)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=default_output_plot,
        help="Output plot filename. If left as default, a descriptive name is auto-generated.",
    )
    parser.add_argument(
        "--reuse-existing-trajectories",
        action="store_true",
        help="If output-dir already contains optimal_trajectory_subset_<i>.xyz for a subset index, reuse it instead of re-running optimal_path.py.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Plot mean ± std-dev across all subsets instead of plotting every subset curve.",
    )
    parser.add_argument(
        "--aggregate-plot-medoid",
        action="store_true",
        help=(
            "In --aggregate mode, plot the smooth-medoid trajectory curve(s) (from the median/medoid XYZ) "
            "instead of the mean curve. Still shows ±1σ envelope across subsets for context. "
            "Requires --write-median-trajectory."
        ),
    )
    parser.add_argument(
        "--write-median-trajectory",
        action="store_true",
        help="Also write a frame-wise median optimal-path trajectory across subsets (aligned), named like the plot but with '_median.xyz'.",
    )
    parser.add_argument(
        "--write-closest-to-mean-trajectory",
        action="store_true",
        help=(
            "Also write a frame-wise closest-to-mean trajectory across subsets (must be an actual subset frame), "
            "named like the plot but with '_closest_mean.xyz'. This can pick different subsets per frame."
        ),
    )
    parser.add_argument(
        "--median-smooth-weight",
        type=float,
        default=0.0,
        help=(
            "When writing the 'median' trajectory, add a smoothness penalty so the chosen subset "
            "doesn't flicker frame-to-frame. Objective is: sum(node_cost) + weight * sum(RMSD between consecutive chosen frames). "
            "0.0 means per-frame medoid (no smoothing)."
        ),
    )
    parser.add_argument(
        "--bond",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "J"),
        help="Bond atom indices I J (e.g., --bond 0 1)",
    )
    parser.add_argument(
        "--angle",
        type=int,
        nargs=3,
        default=None,
        metavar=("I", "J", "K"),
        help="Angle atom indices I J K (e.g., --angle 0 1 2)",
    )
    parser.add_argument(
        "--dihedral",
        type=int,
        nargs=4,
        default=None,
        metavar=("I", "J", "K", "L"),
        help="Dihedral atom indices I J K L (e.g., --dihedral 2 3 4 5)",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=500,
        help="Number of files to randomly sample for each subset (default: 500)",
    )
    parser.add_argument(
        "--topM",
        type=int,
        default=50,
        help="Number of lowest-fit candidates to keep per timestep (default: 50)",
    )
    parser.add_argument(
        "--sample-after-prune",
        action="store_true",
        help=(
            "Change candidate selection order for optimal_path: apply topM/delta pruning first, "
            "then randomly sample from that pool. Default is the historical behavior "
            "(random subset first, then topM)."
        ),
    )
    parser.add_argument(
        "--no-autoscale",
        action="store_true",
        help="Disable optimal_path.py automatic scaling of fit/signal/RMSD cost terms.",
    )

    # Plot axis limits (applies to both aggregate and non-aggregate plots)
    parser.add_argument("--xmin", type=float, default=None, help="Minimum x-axis value (time in fs). Default: 0")
    parser.add_argument("--xmax", type=float, default=None, help="Maximum x-axis value (time in fs). Default: auto")
    parser.add_argument("--ymin", type=float, default=None, help="Minimum y-axis value. Default: auto")
    parser.add_argument("--ymax", type=float, default=None, help="Maximum y-axis value. Default: auto")
    
    args = parser.parse_args()
    
    # Validate that at least one calculation type is specified
    if args.bond is None and args.angle is None and args.dihedral is None:
        parser.error("At least one of --bond, --angle, or --dihedral must be specified")
    if args.aggregate_plot_medoid and not args.write_median_trajectory:
        parser.error("--aggregate-plot-medoid requires --write-median-trajectory")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-name output plot if user didn't override the default.
    if args.output_plot == default_output_plot:
        args.output_plot = _auto_output_plot_name(args)

    # Default save location for plot/representative outputs is output-dir, unless user supplies a path.
    if not os.path.isabs(args.output_plot) and os.path.dirname(args.output_plot) == "":
        args.output_plot = os.path.join(args.output_dir, args.output_plot)
    
    # Lists to store results
    records = []
    
    # Build description of calculations
    calc_parts = []
    if args.bond is not None:
        calc_parts.append(f"Bond {args.bond[0]}-{args.bond[1]}")
    if args.angle is not None:
        calc_parts.append(f"Angle {args.angle[0]}-{args.angle[1]}-{args.angle[2]}")
    if args.dihedral is not None:
        calc_parts.append(f"Dihedral {args.dihedral[0]}-{args.dihedral[1]}-{args.dihedral[2]}-{args.dihedral[3]}")
    calc_desc = ", ".join(calc_parts) if calc_parts else "None"
    
    print(f"\n{'='*60}")
    print(f"Comparing {args.n_subsets} random subsets")
    print(f"Directory: {args.directory}")
    print(f"Random sample size: {args.random_sample}")
    print(f"TopM: {args.topM}")
    print(f"Calculations: {calc_desc}")
    print(f"Output directory: {args.output_dir}")
    print(f"Aggregate mode: {args.aggregate}")
    print(f"{'='*60}\n")
    
    # Run optimal_path for each subset
    for i in range(args.n_subsets):
        seed = args.seed_start + i

        expected_xyz = os.path.join(args.output_dir, f"optimal_trajectory_subset_{i}.xyz")
        reused = False
        if args.reuse_existing_trajectories and os.path.exists(expected_xyz):
            xyz_file = expected_xyz
            reused = True
            print(f"\nReusing existing trajectory for subset {i}: {xyz_file}")
        else:
            xyz_file = run_optimal_path(
                args.directory,
                seed,
                args.output_dir,
                i,
                args.random_sample,
                args.topM,
                no_autoscale=bool(args.no_autoscale),
                sample_after_prune=bool(args.sample_after_prune),
            )
        
        if xyz_file is None:
            print(f"Warning: Failed to generate trajectory for subset {i}, skipping...")
            continue
        
        if not os.path.exists(xyz_file):
            print(f"Warning: Output file {xyz_file} not found, skipping...")
            continue
        
        # Analyze geometry
        csv_file = os.path.join(args.output_dir, f"geometry_subset_{i}.csv")
        print(f"\nAnalyzing geometry for subset {i}...")
        
        if not analyze_geometry(xyz_file, csv_file, args.bond, args.angle, args.dihedral):
            print(f"Warning: Failed to analyze geometry for subset {i}, skipping...")
            continue
        
        if not os.path.exists(csv_file):
            print(f"Warning: Analysis output {csv_file} not found, skipping...")
            continue
        
        records.append(
            {
                "subset_index": i,
                "seed": None if reused else seed,
                "reused": reused,
                "xyz_file": xyz_file,
                "csv_file": csv_file,
            }
        )
    
    if len(records) == 0:
        print("\nError: No valid subsets were generated. Exiting.")
        sys.exit(1)

    # Convenience lists derived from records
    csv_files = [r["csv_file"] for r in records]
    labels = []
    for r in records:
        s = f"Subset {r['subset_index']}"
        if r.get("seed") is not None:
            s += f" (seed={r['seed']})"
        if r.get("reused"):
            s += " (reused)"
        labels.append(s)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(csv_files)} valid subsets")
    print(f"{'='*60}\n")
    
    # Create comparison plot
    print(f"Creating comparison plot...")
    if args.aggregate:
        try:
            medoid_csv = None
            if args.aggregate_plot_medoid:
                # Ensure the smooth-medoid trajectory exists and analyze it to get a matching CSV.
                median_out = os.path.splitext(args.output_plot)[0] + "_median.xyz"
                xyz_files_for_medoid = [r["xyz_file"] for r in records]
                print("Preparing smooth-medoid trajectory for aggregate plot overlay...")
                write_median_trajectory_as_medoid(
                    xyz_files_for_medoid,
                    median_out,
                    smooth_weight=float(args.median_smooth_weight),
                )
                medoid_csv = os.path.join(args.output_dir, "geometry_smooth_medoid.csv")
                if not analyze_geometry(median_out, medoid_csv, args.bond, args.angle, args.dihedral):
                    raise RuntimeError("Failed to analyze geometry for smooth-medoid trajectory.")

            plot_aggregate_mean_std(
                csv_files,
                args.output_plot,
                random_sample=args.random_sample,
                topM=args.topM,
                bond=args.bond,
                angle=args.angle,
                dihedral=args.dihedral,
                xmin=args.xmin,
                xmax=args.xmax,
                ymin=args.ymin,
                ymax=args.ymax,
                medoid_csv=medoid_csv,
            )
            ok = True
        except Exception as e:
            print(f"\nError: Failed to create aggregate plot: {type(e).__name__}: {e}")
            ok = False
    else:
        ok = plot_comparison(
            csv_files,
            labels,
            args.output_plot,
            args.random_sample,
            args.bond,
            args.angle,
            args.dihedral,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )

    if ok:
        print(f"\n{'='*60}")
        print(f"Success! Comparison plot saved to: {args.output_plot}")
        print(f"Intermediate files saved in: {args.output_dir}")
        print(f"{'='*60}\n")

        # Representative subset selection: closest to mean dihedral (if requested),
        # otherwise closest to mean across all selected columns.
        try:
            rep = select_representative_subset(
                records, bond=args.bond, angle=args.angle, dihedral=args.dihedral
            )
            rep_xyz = rep["xyz_file"]
            rep_out = os.path.splitext(args.output_plot)[0] + ".xyz"
            shutil.copyfile(rep_xyz, rep_out)

            print(f"\n{'='*60}")
            print("Representative subset (closest to mean):")
            print(f"  Subset index: {rep['subset_index']}")
            print(f"  Seed: {rep['seed']}")
            print(f"  RMS to mean: {rep['representative_rms_to_mean']:.6g} "
                  f"(frames used: {rep['representative_min_frames_used']})")
            print(f"  Representative XYZ: {rep_out}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\nWarning: Failed to select representative subset: {type(e).__name__}: {e}")

        # Mean optimal-path trajectory across subsets (frame-by-frame average)
        try:
            mean_out = os.path.splitext(args.output_plot)[0] + "_mean.xyz"
            xyz_files = [r["xyz_file"] for r in records]
            print("Averaging optimal-path trajectories across subsets...")
            if average_optimal_trajectories(xyz_files, mean_out, align="kabsch", stat="mean"):
                print(f"Mean optimal-path XYZ: {mean_out}")
            else:
                print("Warning: Failed to write mean optimal-path XYZ.")
        except Exception as e:
            print(f"\nWarning: Failed to create mean optimal-path trajectory: {type(e).__name__}: {e}")

        # Closest-to-mean trajectory across subsets (per-frame, chooses actual subset frame)
        if args.write_closest_to_mean_trajectory:
            try:
                closest_out = os.path.splitext(args.output_plot)[0] + "_closest_mean.xyz"
                xyz_files = [r["xyz_file"] for r in records]
                mean_out = os.path.splitext(args.output_plot)[0] + "_mean.xyz"
                if not os.path.exists(mean_out):
                    raise FileNotFoundError(
                        f"Mean trajectory not found at {mean_out}. "
                        "It is required for closest-to-mean selection."
                    )
                print("Computing closest-to-mean trajectory (per-frame selection from subset trajectories)...")
                if write_closest_to_mean_trajectory_per_frame(xyz_files, mean_out, closest_out):
                    print(f"Closest-to-mean optimal-path XYZ: {closest_out}")
                else:
                    print("Warning: Failed to write closest-to-mean optimal-path XYZ.")
            except Exception as e:
                print(
                    f"\nWarning: Failed to create closest-to-mean optimal-path trajectory: "
                    f"{type(e).__name__}: {e}"
                )

        # Median optimal-path trajectory across subsets (frame-by-frame median)
        if args.write_median_trajectory:
            try:
                median_out = os.path.splitext(args.output_plot)[0] + "_median.xyz"
                xyz_files = [r["xyz_file"] for r in records]
                print("Computing 'median' trajectory as a per-frame medoid (must be an actual subset frame)...")
                if write_median_trajectory_as_medoid(
                    xyz_files, median_out, smooth_weight=float(args.median_smooth_weight)
                ):
                    print(f"Median (medoid) optimal-path XYZ: {median_out}")
                else:
                    print("Warning: Failed to write median (medoid) optimal-path XYZ.")
            except Exception as e:
                print(f"\nWarning: Failed to create median optimal-path trajectory: {type(e).__name__}: {e}")
    else:
        print("\nError: Failed to create comparison plot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
