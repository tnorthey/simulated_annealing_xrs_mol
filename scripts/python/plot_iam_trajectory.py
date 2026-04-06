#!/usr/bin/env python3
"""
Compute the raw IAM signal I(q,t) for an XYZ trajectory and write a PNG for debugging.

Default output is a single heatmap-style PNG showing IAM intensity across q and time
(one row per frame), with a colorbar.

By default, the q-grid is taken from the first column of --q-dat (useful to match a
reference scattering file). You can override with --qmin/--qmax/--qlen.

Examples:
    python3 plot_iam_trajectory.py path/to/trajectory.xyz \
        --q-dat data/chd_reference.dat \
        --inelastic --ion-mode --output-plot iam_heatmap.png

    python3 plot_iam_trajectory.py path/to/trajectory.xyz \
        --qmin 0.4 --qmax 8.0 --qlen 200 \
        --elastic --log --output-plot iam_log.png --output-npz iam.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

import modules.mol as mol
import modules.x as xray


def read_xyz_trajectory(filename: str):
    """Read XYZ file(s) - handles both single structure and trajectory."""
    structures = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except (ValueError, AttributeError):
                break

            comment = f.readline().strip()
            atomlist = []
            xyzmatrix = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 4:
                    break
                atomlist.append(parts[0])
                xyzmatrix.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(atomlist) != natoms:
                break

            structures.append(
                (
                    natoms,
                    comment,
                    np.array(atomlist, dtype=str),
                    np.array(xyzmatrix, dtype=float),
                )
            )

    if not structures:
        raise ValueError(f"No valid structures found in {filename}")
    return structures


def write_xyz_frame(path: str, atomlist: np.ndarray, xyz: np.ndarray, comment: str):
    """Write a single-frame XYZ."""
    natoms = int(len(atomlist))
    with open(path, "w") as f:
        f.write(f"{natoms}\n")
        f.write(f"{comment}\n")
        for sym, (x, y, z) in zip(atomlist.tolist(), xyz):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


def load_q_from_dat(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        # 1-column file: treat as I only, q becomes index
        return np.arange(data.size, dtype=float)
    if data.shape[1] < 1:
        raise ValueError(f"DAT must have at least 1 column; got shape {data.shape} from {path}")
    return np.asarray(data[:, 0], dtype=float)


def load_reference_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference scattering from a DAT file.

    Expected formats:
      - 2+ columns: q, I_ref, ... (uses 2nd column as I_ref)
      - 1 column: I_ref only (q treated as integer index)
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        ref_I = np.asarray(data, dtype=float)
        ref_q = np.arange(ref_I.size, dtype=float)
        return ref_q, ref_I
    if data.shape[1] < 2:
        raise ValueError(f"Reference DAT must have 1 or 2+ columns; got shape {data.shape} from {path}")
    ref_q = np.asarray(data[:, 0], dtype=float)
    ref_I = np.asarray(data[:, 1], dtype=float)
    return ref_q, ref_I


def compute_pcd(iam_tq: np.ndarray, qvector: np.ndarray, ref_q: np.ndarray, ref_I: np.ndarray) -> np.ndarray:
    """Compute PCD(t,q) = 100 * (IAM/I_ref - 1), interpolating I_ref onto qvector."""
    ref_I_interp = np.interp(qvector, ref_q, ref_I)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcd = 100.0 * (iam_tq / ref_I_interp[None, :] - 1.0)
    return np.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)


def load_dat_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DAT-like file as (x, y).

    Supported formats:
      - 2+ columns: x, y, ... (uses first two columns)
      - 1 column: y only (x treated as integer index)
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        y = np.asarray(data, dtype=float)
        x = np.arange(y.size, dtype=float)
        return x, y
    if data.shape[1] < 2:
        raise ValueError(f"DAT must have 1 or 2+ columns; got shape {data.shape} from {path}")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    return x, y


def load_exp_dat(path: str, qvector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load experimental data for overlay.

    Supported formats:
      - 2+ columns: q, y, ... (uses first two columns)
      - 1 column: y only (assumes the same q-grid as the theory data *iff* lengths match)
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        y = np.asarray(data, dtype=float)
        if y.size != np.asarray(qvector).size:
            raise ValueError(
                f"Experimental DAT '{path}' has 1 column (y only) with length {y.size}, "
                f"but theory q-grid has length {np.asarray(qvector).size}. "
                "For 1-column experimental data, lengths must match so the theory q-grid can be reused. "
                "Otherwise, provide a 2-column file with q and y."
            )
        return np.asarray(qvector, dtype=float), y
    if data.shape[1] < 2:
        raise ValueError(f"Experimental DAT must have 1 or 2+ columns; got shape {data.shape} from {path}")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    return x, y


def align_exp_to_qvector(
    qvector: np.ndarray,
    exp_q: np.ndarray,
    exp_y: np.ndarray,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> np.ndarray:
    """
    Align experimental y-values onto the script's q-grid.

    Policy:
      - If len(exp_q) == len(qvector): require q grids to match (within tolerance) and return exp_y.
      - Else: interpolate exp_y(exp_q) onto qvector.
    """
    exp_q = np.asarray(exp_q, dtype=float)
    exp_y = np.asarray(exp_y, dtype=float)
    qvector = np.asarray(qvector, dtype=float)

    if exp_q.size == qvector.size:
        if not np.allclose(exp_q, qvector, atol=atol, rtol=rtol):
            raise ValueError(
                "Experimental DAT has the same length as the calculation q-grid, but q values do not match. "
                "Refusing to interpolate. Ensure the experimental file uses the same q grid."
            )
        return exp_y

    return np.interp(qvector, exp_q, exp_y)


def find_first_peak_q(
    q: np.ndarray,
    y: np.ndarray,
    *,
    min_q: float | None = None,
    mode: str = "max",
) -> float | None:
    """
    Find the first local maximum in y(q) (smallest q where y[i-1] < y[i] > y[i+1]).

    - If min_q is provided, search only for q >= min_q.
    - mode:
        - "max": find peaks in y
        - "abs": find peaks in abs(y)
    Returns q_peak or None if no peak found.
    """
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if q.size < 3:
        return None
    if mode not in {"max", "abs"}:
        raise ValueError(f"Invalid peak mode: {mode}. Expected 'max' or 'abs'.")

    yy = np.abs(y) if mode == "abs" else y

    start = 1
    if min_q is not None:
        # first index with q >= min_q, clamped to [1, n-2]
        idx = int(np.searchsorted(q, float(min_q), side="left"))
        start = max(1, min(idx, q.size - 2))

    for i in range(start, q.size - 1):
        if yy[i - 1] < yy[i] and yy[i] > yy[i + 1]:
            return float(q[i])
    return None


def compute_iam_trajectory(
    structures, qvector: np.ndarray, include_inelastic: bool, *, ion_mode: bool
) -> np.ndarray:
    """Return IAM(q,t) with shape (n_frames, len(qvector))."""
    m = mol.Xyz()
    x = xray.Xray()

    natoms0, _, atomlist0, _ = structures[0]
    atomic_numbers = [m.periodic_table(sym) for sym in atomlist0]
    if any(z is None for z in atomic_numbers):
        bad = [sym for sym, z in zip(atomlist0, atomic_numbers) if z is None]
        raise ValueError(f"Unknown element symbols in XYZ: {sorted(set(bad))}")

    for i, (natoms, _, atomlist, _) in enumerate(structures):
        if natoms != natoms0:
            raise ValueError(f"Frame {i} natoms={natoms} differs from first frame natoms={natoms0}")
        if not np.array_equal(atomlist, atomlist0):
            raise ValueError(f"Frame {i} atom ordering/types differ from first frame; cannot compute consistent IAM")

    compton_array = None
    if include_inelastic:
        try:
            compton_array = x.compton_spline(atomic_numbers, qvector)
        except FileNotFoundError:
            print("Warning: Compton data file not found; continuing with elastic scattering only.", file=sys.stderr)
            include_inelastic = False

    out = np.zeros((len(structures), len(qvector)), dtype=float)
    for i, (_, _, _, xyz) in enumerate(structures):
        iam, _, _, _, _ = x.iam_calc(
            atomic_numbers,
            xyz,
            qvector,
            ion=ion_mode,
            electron_mode=False,
            inelastic=include_inelastic,
            compton_array=compton_array if compton_array is not None else np.zeros(0),
        )
        out[i, :] = iam
    return out


def plot_iam_heatmap(
    time_fs: np.ndarray,
    qvector: np.ndarray,
    iam_tq: np.ndarray,
    output_plot: str,
    log_scale: bool,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
    cbar_label: str | None,
    zero_centered: bool,
    figsize: tuple[float, float],
    dpi: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # imshow expects rows=y, cols=x. We'll map x=q and y=time.
    extent = [float(np.min(qvector)), float(np.max(qvector)), float(np.min(time_fs)), float(np.max(time_fs))]

    if log_scale and zero_centered:
        raise ValueError("Cannot use zero-centered colormap with log scale.")

    if log_scale:
        # Avoid log(0) by clipping to small positive values.
        arr = np.clip(iam_tq, a_min=np.finfo(float).tiny, a_max=None)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(arr, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm)
        default_cbar_label = "IAM (log scale)"
    else:
        if zero_centered:
            data_min = float(np.nanmin(iam_tq))
            data_max = float(np.nanmax(iam_tq))
            if vmin is None and vmax is None:
                bound = max(abs(data_min), abs(data_max))
                if bound == 0.0:
                    bound = 1.0
                vmin_ = -bound
                vmax_ = bound
            elif vmin is None and vmax is not None:
                vmax_ = float(vmax)
                vmin_ = -abs(vmax_)
            elif vmax is None and vmin is not None:
                vmin_ = float(vmin)
                vmax_ = abs(vmin_)
            else:
                vmin_ = float(vmin)
                vmax_ = float(vmax)

            if not (vmin_ < 0.0 < vmax_):
                raise ValueError(
                    f"Zero-centered colormap requires vmin < 0 < vmax, got vmin={vmin_}, vmax={vmax_}."
                )
            norm = TwoSlopeNorm(vmin=vmin_, vcenter=0.0, vmax=vmax_)
            im = ax.imshow(iam_tq, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm)
        else:
            im = ax.imshow(iam_tq, aspect="auto", origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        default_cbar_label = "IAM"

    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel("time (fs)")
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(default_cbar_label if cbar_label is None else cbar_label)

    fig.tight_layout()
    fig.savefig(output_plot, bbox_inches="tight")
    plt.close(fig)


def plot_iam_frame(
    qvector: np.ndarray,
    iam_q: np.ndarray,
    output_png: str,
    time_fs: float | None,
    logy: bool,
    title: str | None,
    ylabel: str,
    exp_q: np.ndarray | None,
    exp_y: np.ndarray | None,
    exp_label: str | None,
    calc_label: str | None,
    show_residual: bool,
    mark_first_peak: str,
    peak_min_q: float | None,
    peak_mode: str,
    frame_xlim: tuple[float, float] | None,
    frame_ylim: tuple[float, float] | None,
    figsize: tuple[float, float],
    dpi: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(qvector, iam_q, linewidth=1.2, label=(calc_label or "calculated"))
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if frame_xlim is not None:
        ax.set_xlim(frame_xlim[0], frame_xlim[1])
    if frame_ylim is not None:
        ax.set_ylim(frame_ylim[0], frame_ylim[1])

    if exp_q is not None and exp_y is not None:
        # If the experimental file has the same number of points as the calculated curve,
        # require that its q-grid matches (no interpolation allowed in that case).
        if np.asarray(exp_q).size == np.asarray(qvector).size:
            _ = align_exp_to_qvector(qvector, exp_q, exp_y)  # validates q match
            ax.plot(qvector, exp_y, linewidth=1.2, alpha=0.85, label=(exp_label or "experimental"))
        else:
            ax.plot(exp_q, exp_y, linewidth=1.2, alpha=0.85, label=(exp_label or "experimental"))
        if show_residual:
            # Plot residual on a secondary axis to avoid confusing scales.
            ax2 = ax.twinx()
            exp_interp = align_exp_to_qvector(qvector, exp_q, exp_y)
            resid = iam_q - exp_interp
            ax2.plot(qvector, resid, linewidth=1.0, alpha=0.6, color="tab:red", label="residual (calc-exp)")
            ax2.set_ylabel("residual")
            # Merge legends
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="best", frameon=False)
        else:
            ax.legend(loc="best", frameon=False)

    # Peak marker (per-frame)
    if mark_first_peak != "none":
        if mark_first_peak == "calc":
            q_for_peak = qvector
            y_for_peak = iam_q
        elif mark_first_peak == "exp":
            if exp_q is None or exp_y is None:
                q_for_peak = y_for_peak = None
            else:
                q_for_peak = exp_q
                y_for_peak = exp_y
        else:
            raise ValueError(f"Invalid --mark-first-peak value: {mark_first_peak}")

        if q_for_peak is not None and y_for_peak is not None:
            q_peak = find_first_peak_q(q_for_peak, y_for_peak, min_q=peak_min_q, mode=peak_mode)
            if q_peak is not None:
                ax.axvline(q_peak, linestyle="--", linewidth=1.0, color="k", alpha=0.8)
                # Annotate near top of axes
                ymin, ymax = ax.get_ylim()
                y_text = ymin + 0.92 * (ymax - ymin)
                ax.text(
                    q_peak,
                    y_text,
                    f"{q_peak:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
                )

    if title is not None:
        ax.set_title(title)
    elif time_fs is not None:
        ax.set_title(f"IAM frame (t = {time_fs:g} fs)")

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute and plot raw IAM I(q,t) from an XYZ trajectory.")
    parser.add_argument("input_xyz", help="XYZ file (single structure or trajectory).")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index (0-based, inclusive) for processing/plotting. Default: 0",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame index (0-based, exclusive) for processing/plotting. Default: process to last frame",
    )
    parser.add_argument(
        "--q-dat",
        default="data/chd_reference.dat",
        help="DAT file used only to define q-grid (uses column 1). Default: data/chd_reference.dat",
    )
    parser.add_argument("--qmin", type=float, default=None, help="Override q-grid: min q (A^-1).")
    parser.add_argument("--qmax", type=float, default=None, help="Override q-grid: max q (A^-1).")
    parser.add_argument("--qlen", type=int, default=None, help="Override q-grid: number of q points.")
    # To avoid user error / ambiguity, require explicitly choosing elastic vs inelastic.
    scatter_mode = parser.add_mutually_exclusive_group(required=True)
    scatter_mode.add_argument(
        "--inelastic",
        dest="inelastic",
        action="store_true",
        help="Include inelastic (Compton) scattering (if available).",
    )
    scatter_mode.add_argument(
        "--elastic",
        dest="inelastic",
        action="store_false",
        help="Elastic-only scattering (disable Compton).",
    )
    parser.add_argument(
        "--ion-mode",
        dest="ion_mode",
        action="store_true",
        help="Use ion-corrected atomic scattering factors (see modules/x.py).",
    )
    # Backwards-compatible alias
    parser.add_argument(
        "--ion",
        dest="ion_mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dt-fs", type=float, default=20.0, help="Time per frame in fs. Default: 20")
    parser.add_argument("--t0-fs", type=float, default=10.0, help="Time of first frame in fs. Default: 10")
    parser.add_argument("--log", action="store_true", help="Use log color scale for IAM (useful for dynamic range).")
    parser.add_argument(
        "--pcd",
        action="store_true",
        help="Compute and plot PCD per frame relative to --ref-dat (uses 2nd column as I_ref).",
    )
    parser.add_argument(
        "--ref-dat",
        type=str,
        default=None,
        help="Reference DAT file for PCD. Required if --pcd is set.",
    )
    parser.add_argument(
        "--pcd-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to PCD (e.g., excitation fraction). Default: 1.0",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap. Default: viridis")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min. Default: auto")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max. Default: auto")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--figsize", type=str, default="10,6", help='Figure size "W,H" in inches. Default: "10,6"')
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150")
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output PNG. Default: <input_basename>_iam_heatmap.png",
    )
    parser.add_argument(
        "--output-dat",
        default=None,
        help="Optional output DAT with columns: q, IAM_frame1, ... (or PCD_frame1, ... if --pcd).",
    )
    parser.add_argument(
        "--output-npz",
        default=None,
        help="Optional output NPZ containing q, time_fs, iam (t,q).",
    )
    parser.add_argument(
        "--output-first-peak-dat",
        default=None,
        help=(
            "Optional output DAT with 2 columns: time_fs, q_first_peak. "
            "Peak is computed from the calculated curve for each frame using the same "
            "--peak-min-q/--peak-mode logic. Uses PCD curve if --pcd is set, else IAM."
        ),
    )
    parser.add_argument(
        "--write-frame-pngs",
        action="store_true",
        help='Also write one PNG per frame (IAM vs q), named "frame_XX.png" in --frames-dir.',
    )
    parser.add_argument(
        "--write-frame-dats",
        action="store_true",
        help='Also write one DAT per frame (q, y), named "frame_XX.dat" in --frames-dir (y is IAM or PCD).',
    )
    parser.add_argument(
        "--write-frame-xyzs",
        action="store_true",
        help='Also write one XYZ per frame, named "frame_XX.xyz" in --frames-dir.',
    )
    parser.add_argument(
        "--exp-dat",
        type=str,
        default=None,
        help="Optional experimental DAT to overlay on per-frame plots (2+ cols: q, y). Applied to all frames.",
    )
    parser.add_argument(
        "--exp-dat-template",
        type=str,
        default=None,
        help=(
            'Optional per-frame experimental DAT template using Python format, e.g. '
            '"data/eirik_data_{exp_frame:02d}.dat". Available fields: '
            "{frame} (theory absolute frame index), "
            "{exp_frame} (mapped experimental frame index), "
            "{rel_frame} (0..N-1)."
        ),
    )
    parser.add_argument(
        "--exp-start-frame",
        type=int,
        default=0,
        help=(
            "Experimental frame index mapping using: exp_frame = frame - exp-start-frame. "
            "Default: 0. (You can pass negative values.)"
        ),
    )
    parser.add_argument(
        "--exp-frame-offset",
        type=int,
        default=None,
        help=(
            "Alternative experimental mapping using: exp_frame = frame + exp-frame-offset. "
            "If set, this overrides --exp-start-frame."
        ),
    )
    parser.add_argument(
        "--exp-missing",
        choices=["ignore", "warn", "error"],
        default="warn",
        help=(
            "Behavior when an experimental per-frame file (from --exp-dat-template) is missing. "
            "Default: warn"
        ),
    )
    parser.add_argument(
        "--exp-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to experimental y before plotting/writing. Default: 1.0",
    )
    parser.add_argument(
        "--exp-label",
        type=str,
        default="experimental",
        help='Legend label for experimental curve. Default: "experimental"',
    )
    parser.add_argument(
        "--calc-label",
        type=str,
        default="calculated",
        help='Legend label for calculated curve. Default: "calculated"',
    )
    parser.add_argument(
        "--show-residual",
        action="store_true",
        help="On per-frame plots, also show residual (calc - exp) on a secondary axis (requires exp data).",
    )
    parser.add_argument(
        "--mark-first-peak",
        choices=["none", "calc", "exp"],
        default="none",
        help='On per-frame PNGs, mark the first peak and label its q value (choose "calc" or "exp"). Default: none',
    )
    parser.add_argument(
        "--peak-min-q",
        type=float,
        default=None,
        help="Minimum q (A^-1) to start searching for the first peak. Default: None",
    )
    parser.add_argument(
        "--peak-mode",
        choices=["max", "abs"],
        default="max",
        help='Peak detection mode: "max" uses y, "abs" uses |y|. Default: max',
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help='Directory for per-frame PNGs. Default: "<output_plot_dir>/frames" if --write-frame-pngs is set.',
    )
    parser.add_argument(
        "--frame-pad-width",
        type=int,
        default=None,
        help="Zero-pad width for per-frame PNG numbering. Default: max(2, digits in n_frames-1).",
    )
    parser.add_argument(
        "--frame-logy",
        action="store_true",
        help="Use log y-axis for per-frame IAM line plots.",
    )
    parser.add_argument(
        "--frame-xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Per-frame PNG x-axis limits as two numbers (e.g. --frame-xlim 0.4 8.0). Default: auto.",
    )
    parser.add_argument(
        "--frame-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Per-frame PNG y-axis limits as two numbers (e.g. --frame-ylim -10 10). Default: auto.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_xyz):
        raise FileNotFoundError(f"Input XYZ not found: {args.input_xyz}")

    if args.dt_fs <= 0:
        raise ValueError("--dt-fs must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    try:
        w_str, h_str = (x.strip() for x in str(args.figsize).split(","))
        figsize = (float(w_str), float(h_str))
    except Exception as e:
        raise ValueError(f'Invalid --figsize "{args.figsize}". Expected "W,H" (e.g. "10,6").') from e
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise ValueError("--figsize values must be > 0")

    if args.pcd:
        if args.ref_dat is None:
            raise ValueError("--pcd requires --ref-dat")
        if not os.path.exists(args.ref_dat):
            raise FileNotFoundError(f"Reference DAT not found: {args.ref_dat}")
        if args.log:
            raise ValueError("--log is only supported for IAM heatmaps; disable --log when using --pcd.")
        if args.frame_logy:
            raise ValueError("--frame-logy is not supported for PCD (PCD can be negative).")

    def _validate_pair(arg, *, name: str) -> tuple[float, float] | None:
        if arg is None:
            return None
        a = float(arg[0])
        b = float(arg[1])
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(f"{name} values must be finite numbers.")
        if b <= a:
            raise ValueError(f"{name} must satisfy max > min; got {a},{b}")
        return (a, b)

    frame_xlim = _validate_pair(args.frame_xlim, name="--frame-xlim")
    frame_ylim = _validate_pair(args.frame_ylim, name="--frame-ylim")
    if args.frame_logy and frame_ylim is not None and frame_ylim[0] <= 0:
        raise ValueError("--frame-ylim lower bound must be > 0 when using --frame-logy.")

    if args.exp_dat is not None and args.exp_dat_template is not None:
        raise ValueError("Use only one of --exp-dat or --exp-dat-template (not both).")
    if args.show_residual and args.exp_dat is None and args.exp_dat_template is None:
        raise ValueError("--show-residual requires --exp-dat or --exp-dat-template")

    # Determine q-grid
    if args.qmin is not None or args.qmax is not None or args.qlen is not None:
        if args.qmin is None or args.qmax is None or args.qlen is None:
            raise ValueError("If overriding q-grid, you must provide --qmin, --qmax, and --qlen together.")
        if args.qmin >= args.qmax:
            raise ValueError("--qmin must be < --qmax")
        if args.qlen <= 1:
            raise ValueError("--qlen must be > 1")
        qvector = np.linspace(float(args.qmin), float(args.qmax), int(args.qlen), endpoint=True)
    else:
        if not os.path.exists(args.q_dat):
            raise FileNotFoundError(f"q DAT file not found: {args.q_dat}")
        qvector = load_q_from_dat(args.q_dat)

    structures = read_xyz_trajectory(args.input_xyz)
    n_frames_total = len(structures)
    start = int(args.start_frame)
    end = n_frames_total if args.end_frame is None else int(args.end_frame)
    if start < 0 or start >= n_frames_total:
        raise ValueError(f"--start-frame must be in [0, {n_frames_total - 1}]")
    if end < 0 or end > n_frames_total:
        raise ValueError(f"--end-frame must be in [0, {n_frames_total}]")
    if end <= start:
        raise ValueError("--end-frame must be > --start-frame")

    # Select subset, but keep absolute frame indices for naming and time axis.
    frame_indices = np.arange(n_frames_total, dtype=int)
    sel_indices = frame_indices[start:end]
    structures = [structures[i] for i in sel_indices.tolist()]
    n_frames = len(structures)
    time_fs = sel_indices.astype(float) * float(args.dt_fs) + float(args.t0_fs)

    iam_tq = compute_iam_trajectory(
        structures, qvector, include_inelastic=args.inelastic, ion_mode=bool(args.ion_mode)
    )

    ref_q = ref_I = None
    pcd_tq = None
    if args.pcd:
        ref_q, ref_I = load_reference_dat(args.ref_dat)
        pcd_tq = compute_pcd(iam_tq, qvector=qvector, ref_q=ref_q, ref_I=ref_I) * float(args.pcd_scale)

    base = os.path.splitext(os.path.basename(args.input_xyz))[0]
    if args.pcd:
        default_plot = f"{base}_pcd_heatmap.png"
    else:
        default_plot = f"{base}_iam_heatmap.png"
    output_plot = args.output_plot or default_plot

    if args.output_dat:
        if args.pcd:
            out = np.column_stack([qvector] + [pcd_tq[i, :] for i in range(n_frames)])
            header = "q(A^-1)  " + "  ".join([f"PCD_frame{int(fi)}" for fi in sel_indices.tolist()])
        else:
            out = np.column_stack([qvector] + [iam_tq[i, :] for i in range(n_frames)])
            header = "q(A^-1)  " + "  ".join([f"IAM_frame{int(fi)}" for fi in sel_indices.tolist()])
        np.savetxt(args.output_dat, out, fmt="%.6e", header=header)

    if args.output_npz:
        payload = {"q": qvector, "time_fs": time_fs, "iam": iam_tq}
        if args.pcd:
            payload["pcd"] = pcd_tq
            payload["ref_q"] = ref_q
            payload["ref_I"] = ref_I
            payload["pcd_scale"] = float(args.pcd_scale)
        np.savez_compressed(args.output_npz, **payload)

    pcd_label = f"PCD × {args.pcd_scale:g} (%)"
    plot_iam_heatmap(
        time_fs=time_fs,
        qvector=qvector,
        iam_tq=pcd_tq if args.pcd else iam_tq,
        output_plot=output_plot,
        log_scale=bool(args.log) if not args.pcd else False,
        cmap=("bwr" if args.pcd and str(args.cmap) == "viridis" else str(args.cmap)),
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        cbar_label=(pcd_label if args.pcd else None),
        zero_centered=bool(args.pcd),
        figsize=figsize,
        dpi=int(args.dpi),
    )

    print(f"Wrote plot: {output_plot}")

    # Optional: write first-peak q position vs time (uses calculated curve per frame)
    if args.output_first_peak_dat is not None:
        y_tq = pcd_tq if args.pcd else iam_tq
        q_peaks = np.full((n_frames,), np.nan, dtype=float)
        for i in range(n_frames):
            q_peak = find_first_peak_q(
                qvector,
                y_tq[i, :],
                min_q=args.peak_min_q,
                mode=str(args.peak_mode),
            )
            q_peaks[i] = np.nan if q_peak is None else float(q_peak)
        out = np.column_stack([time_fs, q_peaks])
        np.savetxt(
            args.output_first_peak_dat,
            out,
            fmt="%.6e",
            header="time_fs  q_first_peak_A^-1",
        )
        print(f"Wrote first-peak DAT: {args.output_first_peak_dat}")

    if args.write_frame_pngs or args.write_frame_dats or args.write_frame_xyzs:
        out_dir = args.frames_dir
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(output_plot)) or ".", "frames")
        os.makedirs(out_dir, exist_ok=True)

        pad = args.frame_pad_width
        if pad is None:
            pad = max(2, len(str(int(np.max(sel_indices)))))
        if pad < 1:
            raise ValueError("--frame-pad-width must be >= 1")

        for rel_i, abs_i in enumerate(sel_indices.tolist()):
            y = pcd_tq[rel_i, :] if args.pcd else iam_tq[rel_i, :]
            stem = f"frame_{abs_i:0{pad}d}"

            exp_q = exp_y = None
            if args.exp_dat is not None:
                if not os.path.exists(args.exp_dat):
                    raise FileNotFoundError(f"Experimental DAT not found: {args.exp_dat}")
                exp_q, exp_y = load_exp_dat(args.exp_dat, qvector=qvector)
            elif args.exp_dat_template is not None:
                # Use Python format mini-language with `frame` and `i`.
                if args.exp_frame_offset is not None:
                    exp_frame = abs_i + int(args.exp_frame_offset)
                else:
                    exp_frame = abs_i - int(args.exp_start_frame)
                exp_path = args.exp_dat_template.format(frame=abs_i, i=abs_i, exp_frame=exp_frame, rel_frame=rel_i)
                if exp_frame < 0:
                    msg = (
                        f"Mapped exp_frame={exp_frame} for theory frame {abs_i} is negative; "
                        f"cannot load exp file. (Tip: you may need --start-frame {abs_i - exp_frame}.)"
                    )
                    if args.exp_missing == "error":
                        raise ValueError(msg)
                    if args.exp_missing == "warn":
                        print(f"Warning: {msg}", file=sys.stderr)
                    exp_q = exp_y = None
                elif os.path.exists(exp_path):
                    exp_q, exp_y = load_exp_dat(exp_path, qvector=qvector)
                else:
                    msg = f"Experimental DAT missing for theory frame {abs_i}: expected '{exp_path}'"
                    if args.exp_missing == "error":
                        raise FileNotFoundError(msg)
                    if args.exp_missing == "warn":
                        print(f"Warning: {msg}", file=sys.stderr)
                    exp_q = exp_y = None

            if exp_y is not None:
                exp_y = np.asarray(exp_y, dtype=float) * float(args.exp_scale)

            if args.write_frame_pngs:
                out_png = os.path.join(out_dir, f"{stem}.png")
                plot_iam_frame(
                    qvector=qvector,
                    iam_q=y,
                    output_png=out_png,
                    time_fs=float(time_fs[rel_i]),
                    logy=bool(args.frame_logy),
                    title=None,
                    ylabel=(pcd_label if args.pcd else "IAM"),
                    exp_q=exp_q,
                    exp_y=exp_y,
                    exp_label=args.exp_label,
                    calc_label=args.calc_label,
                    show_residual=bool(args.show_residual),
                    mark_first_peak=str(args.mark_first_peak),
                    peak_min_q=args.peak_min_q,
                    peak_mode=str(args.peak_mode),
                    frame_xlim=frame_xlim,
                    frame_ylim=frame_ylim,
                    figsize=figsize,
                    dpi=int(args.dpi),
                )

            if args.write_frame_dats:
                out_dat = os.path.join(out_dir, f"{stem}.dat")
                if exp_q is not None and exp_y is not None:
                    exp_interp = align_exp_to_qvector(qvector, exp_q, exp_y)
                    header = f"q(A^-1)  {'PCD' if args.pcd else 'IAM'}  exp"
                    np.savetxt(out_dat, np.column_stack([qvector, y, exp_interp]), fmt="%.6e", header=header)
                else:
                    header = f"q(A^-1)  {'PCD' if args.pcd else 'IAM'}"
                    np.savetxt(out_dat, np.column_stack([qvector, y]), fmt="%.6e", header=header)

            if args.write_frame_xyzs:
                out_xyz = os.path.join(out_dir, f"{stem}.xyz")
                _, comment, atomlist, xyz = structures[rel_i]
                c = f"{comment} | frame={abs_i} | t={float(time_fs[rel_i]):g} fs"
                write_xyz_frame(out_xyz, atomlist=atomlist, xyz=xyz, comment=c)

        if args.write_frame_pngs:
            print(f"Wrote per-frame PNGs: {out_dir}/frame_*.png")
        if args.write_frame_dats:
            print(f"Wrote per-frame DATs: {out_dir}/frame_*.dat")
        if args.write_frame_xyzs:
            print(f"Wrote per-frame XYZs: {out_dir}/frame_*.xyz")

    if args.output_dat:
        print(f"Wrote DAT:  {args.output_dat}")
    if args.output_npz:
        print(f"Wrote NPZ:  {args.output_npz}")


if __name__ == "__main__":
    main()

