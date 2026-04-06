#!/usr/bin/env python3
"""
plot_fit_histograms.py

Plot histograms of fit values from filenames for each timestep separately.
Helps identify the distribution of fit values and where "converged" runs fall.

Usage:
    python3 plot_fit_histograms.py results/ [--output OUTPUT] [--bins BINS]
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Filename parsing pattern
# Example: 01_000.17533577.dat
NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path):
    """Parse timestep and fit value from filename."""
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    t = int(m.group("t"))
    fit = float(m.group("fit"))
    return t, fit


def collect_fit_values(directory, dat_ext=".dat", xyz_ext=".xyz"):
    """Collect fit values grouped by timestep."""
    dats = glob.glob(os.path.join(directory, f"*{dat_ext}"))
    xyzs = glob.glob(os.path.join(directory, f"*{xyz_ext}"))
    
    # Map by (timestep, fit) -> file
    dat_map = {}
    xyz_map = {}
    for p in dats:
        parsed = parse_name(p)
        if parsed:
            dat_map[parsed] = p
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            xyz_map[parsed] = p
    
    # Keep only keys that have both dat and xyz
    keys = sorted(set(dat_map.keys()) & set(xyz_map.keys()), key=lambda x: (x[0], x[1]))
    if not keys:
        raise RuntimeError(
            "No matched (timestep,fit) pairs found with both .dat and .xyz. "
            "Expected names like 01_0.12345678.dat and 01_0.12345678.xyz"
        )
    
    # Group fit values by timestep
    by_timestep = defaultdict(list)
    for (t, fit) in keys:
        by_timestep[t].append(fit)
    
    # Sort timesteps and fit values
    timesteps = sorted(by_timestep.keys())
    for t in timesteps:
        by_timestep[t].sort()
    
    return timesteps, by_timestep


def plot_histograms(timesteps, by_timestep, output_file, bins=50):
    """Plot histograms for each timestep."""
    n_timesteps = len(timesteps)
    
    # Determine layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_timesteps)))
    n_rows = int(np.ceil(n_timesteps / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    
    # Flatten axes if needed
    if n_timesteps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot histogram for each timestep
    for idx, t in enumerate(timesteps):
        fits = np.array(by_timestep[t])
        
        ax = axes[idx]
        ax.hist(fits, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Fit Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Timestep {t:02d} (n={len(fits)})')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_fit = np.mean(fits)
        median_fit = np.median(fits)
        min_fit = np.min(fits)
        max_fit = np.max(fits)
        q25 = np.percentile(fits, 25)
        q75 = np.percentile(fits, 75)
        
        stats_text = (
            f'Min: {min_fit:.4f}\n'
            f'Q25: {q25:.4f}\n'
            f'Median: {median_fit:.4f}\n'
            f'Mean: {mean_fit:.4f}\n'
            f'Q75: {q75:.4f}\n'
            f'Max: {max_fit:.4f}'
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8, family='monospace')
    
    # Hide unused subplots
    for idx in range(n_timesteps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot to: {output_file}")
    
    # Also print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics (across {n_timesteps} timesteps):")
    print(f"{'='*60}")
    print(f"{'Timestep':<12} {'Count':<10} {'Min':<12} {'Q25':<12} {'Median':<12} {'Q75':<12} {'Max':<12}")
    print(f"{'-'*60}")
    for t in timesteps:
        fits = np.array(by_timestep[t])
        print(f"{t:02d}          {len(fits):<10} {np.min(fits):<12.6f} "
              f"{np.percentile(fits, 25):<12.6f} {np.median(fits):<12.6f} "
              f"{np.percentile(fits, 75):<12.6f} {np.max(fits):<12.6f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of fit values for each timestep separately."
    )
    parser.add_argument(
        "directory",
        help="Directory containing *.dat and *.xyz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fit_histograms.png",
        help="Output plot filename (default: fit_histograms.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histograms (default: 50)",
    )
    parser.add_argument(
        "--dat-ext",
        type=str,
        default=".dat",
        help="Extension for DAT files (default: .dat)",
    )
    parser.add_argument(
        "--xyz-ext",
        type=str,
        default=".xyz",
        help="Extension for XYZ files (default: .xyz)",
    )
    
    args = parser.parse_args()
    
    print(f"Collecting fit values from {args.directory}...")
    timesteps, by_timestep = collect_fit_values(
        args.directory,
        dat_ext=args.dat_ext,
        xyz_ext=args.xyz_ext
    )
    
    print(f"Found {len(timesteps)} timesteps")
    total_files = sum(len(fits) for fits in by_timestep.values())
    print(f"Total files: {total_files}")
    
    print(f"\nPlotting histograms...")
    plot_histograms(timesteps, by_timestep, args.output, bins=args.bins)


if __name__ == "__main__":
    main()
