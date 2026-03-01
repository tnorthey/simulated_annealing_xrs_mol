# Simulated Annealing XRS Molecular

Simulated annealing optimization for X-ray scattering molecular structure refinement.

## Installation (via Conda)

```sh
conda create --name sa python=3.10
conda activate sa
conda install -c conda-forge pyscf numba rdkit openbabel openmm openff-toolkit openff-forcefields pytest
```

Optional (CUDA backend):
```sh
conda install -c conda-forge cupy
```

## Quick Start

Run with default configuration:
```bash
python3 run.py
```

Parameters are documented in `input.toml`. Command-line arguments override TOML defaults.

## Scripts

Each script supports `--help` to display all available options.

| Script | Purpose |
|--------|---------|
| `run.py` | Main simulated annealing optimization |
| `optimal_path.py` | Find smoothest trajectory through candidate structures using DP |
| `compare_random_subsets.py` | Compare multiple random subsets via `optimal_path.py`, analyze geometry, and plot |
| `calculate_iam.py` | Standalone IAM scattering calculation from XYZ files |
| `analyze_geometry.py` | Extract bond lengths, angles, and dihedrals from XYZ files |
| `plot_geometry.py` | Plot geometry CSVs produced by `analyze_geometry.py` |
| `plot_fit_histograms.py` | Histogram of fit values per timestep |
| `average_xyz.py` | Frame-wise averaging of XYZ trajectories |

## Testing

```bash
pytest tests/
```
