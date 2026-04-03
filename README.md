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

Optional multi-chain CUDA execution (independent SA chains in parallel):
- Set `gpu_backend = "cuda"` and `gpu_chains = <N>` in `input.toml`
- Or use CLI: `python3 run.py --gpu-backend cuda --gpu-chains 8`
- Outputs include one `.xyz/.dat` per chain (`..._c000`, `..._c001`, etc.) plus the best-chain file for compatibility
- For timestep workflows, `run_start.sh` runs an initial step from a chosen XYZ; `run_gpu_from_previous_timestep.sh` averages the previous step’s best structures into `results/<N−1>_mean.xyz` (when launching run-id `N`), then runs one GPU job (default 1024 chains).
- If `ab_initio_scattering_file` is set: two columns `q`, `I` (ab-initio **total** intensity at **reference_xyz**). The `files.ab_initio_correction_mode` setting selects the denominator: **`elastic`** (default) computes \(c(q)=I_{\mathrm{ab\ initio}}/I_{\mathrm{IAM,\ elastic}}(\mathrm{ref})\), writes `reference_iam_scattering.dat`, and SA uses **\(c\times I_{\mathrm{elastic}}\)**; **`total`** uses \(c(I)=I_{\mathrm{ab\ initio}}/I_{\mathrm{IAM,\ total}}(\mathrm{ref})\) (including Compton when `inelastic` is on), writes `reference_iam_total_scattering.dat`, and SA uses **\(c\times I_{\mathrm{total}}\)** (legacy multiply path). In **PCD** mode, `reference_dat_file` is **required** for the PCD baseline \(I_{\mathrm{ref}}(q)\). Isotropic q only (not with Ewald mode). If unset, \(c\equiv 1\). CLI: `--ab-initio-correction-mode {elastic,total}`. The standalone `calculate_iam.py` has `--ab-initio-correction-mode` (with `--pcd` and `--ab-initio-scattering`, pass `--reference` and `--reference-dat`).

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
| `run_gpu_from_previous_timestep.sh` | Kabsch mean of best `TOP_N` structures from timestep *N−1*, save as `results/<N−1>_mean.xyz`, then one CUDA `run.py` with `GPU_CHAINS` (default 1024). See `--help`. Env: `RESULTS_DIR`, `TOP_N`, `PREV_STEP`, `STARTING_XYZ`, `TARGET_FILE`, `ALIGN_INDICES`. |
| `run_cpu_from_previous_timestep.sh` | Same as `run_gpu_from_previous_timestep.sh`, but runs `run.py` with `--gpu-backend cpu`. If `N_WORKERS>1`, delegates to `run_parallel.sh` (launches N workers starting from the computed mean). See `--help`. Env: `RESULTS_DIR`, `TOP_N`, `PREV_STEP`, `STARTING_XYZ`, `TARGET_FILE`, `ALIGN_INDICES`, `N_WORKERS`, `CPU_CHAINS`. |
| `run_start.sh` | First (or any) timestep from a **fixed** starting XYZ: copies it to `results/NN_mean.xyz`, then same CUDA `run.py` as above. Pass `starting_xyz` as the second argument, or set `STARTING_XYZ`. See `--help`. |

## Testing

```bash
pytest tests/
```
