# Simulated Annealing XRS Molecular

Simulated annealing optimization for X-ray scattering molecular structure refinement.

---

## Installation (via Conda)

We recommend using **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** or **Anaconda** for environment setup.

1. Create a new conda environment:
   ```sh
   conda create --name sa python=3.10
   conda activate sa
   ```

2. Install the following required packages from `conda-forge`:
   ```sh
   conda install -c conda-forge pyscf numba rdkit openbabel openmm openff-toolkit openff-forcefields pytest
   ```
---


## Quick Start

### Basic Usage

Run with default configuration (uses `input.toml`):
```bash
python3 run.py
```

> **Tip**: The `input.toml` file contains detailed comments explaining all parameters. See that file for comprehensive parameter documentation.

Run with a different config file:
```bash
python3 run.py --config my_config.toml
```

Show help message:
```bash
python3 run.py --help
```

## Command Line Examples

### Basic Run with Overrides
```bash
python3 run.py --mode test --molecule chd --run-id test_run
```

### Override File Paths
```bash
python3 run.py \
    --start-xyz-file xyz/my_start.xyz \
    --reference-xyz-file xyz/my_reference.xyz \
    --target-file xyz/my_target.xyz \
    --forcefield-file forcefields/my_forcefield.offxml
```

### Enable PySCF Normal Modes Calculation
```bash
python3 run.py --run-pyscf-modes --pyscf-basis 6-31g
```

### Enable Verbose Output
```bash
python3 run.py --verbose
```

### Configure Scattering Parameters
```bash
python3 run.py \
    --qmin 0.1 \
    --qmax 8.0 \
    --qlen 81 \
    --inelastic \
    --pcd-mode
```

### Configure Simulated Annealing Parameters
```bash
python3 run.py \
    --sa-starting-temp 1.0 \
    --sa-nsteps 4000 \
    --sa-step-size 0.012 \
    --greedy-algorithm \
    --ga-nsteps 20000 \
    --ga-step-size 0.012
```

### Enable Bonds, Angles, and Torsions
```bash
python3 run.py --bonds --angles --torsions
```

### Configure Restart Parameters
```bash
python3 run.py \
    --nrestarts 5 \
    --ntotalruns 5
```

### Enable Ewald Mode for Scattering
```bash
python3 run.py --ewald-mode
```

### Configure Theta and Phi Parameters (for 2D/3D scattering)
```bash
python3 run.py \
    --tmin 0.0 \
    --tmax 1.0 \
    --tlen 21 \
    --pmin 0.0 \
    --pmax 2.0 \
    --plen 21
```

### Enable Sampling with Boltzmann Temperature
```bash
python3 run.py --sampling --boltzmann-temperature 300.0
```

### Configure Noise Parameters
```bash
python3 run.py \
    --noise-value 0.05 \
    --noise-data-file noise/my_noise.dat
```

### Tuning Parameters for Simulated Annealing
```bash
python3 run.py \
    --tuning-ratio-target 0.9 \
    --c-tuning-initial 0.001
```

### Enable Non-Hydrogen Modes Only
```bash
python3 run.py --non-h-modes-only
```

### Enable PySCF HF Energy Calculation
```bash
python3 run.py --hf-energy
```

### Write DAT File Output
```bash
python3 run.py --write-dat-file
```

### Complete Example: Full Configuration Override
```bash
python3 run.py \
    --config input.toml \
    --mode test \
    --molecule chd \
    --run-id production_run_001 \
    --results-dir results/production \
    --start-xyz-file xyz/start.xyz \
    --reference-xyz-file xyz/chd_reference.xyz \
    --target-file xyz/target.xyz \
    --forcefield-file forcefields/openff_unconstrained-2.0.0.offxml \
    --run-pyscf-modes \
    --pyscf-basis 6-31g* \
    --verbose \
    --inelastic \
    --pcd-mode \
    --ewald-mode \
    --qmin 1e-9 \
    --qmax 8.0 \
    --qlen 81 \
    --tmin 0.0 \
    --tmax 1.0 \
    --tlen 21 \
    --pmin 0.0 \
    --pmax 2.0 \
    --plen 21 \
    --sa-starting-temp 1.0 \
    --sa-nsteps 4000 \
    --greedy-algorithm \
    --ga-nsteps 20000 \
    --sa-step-size 0.012 \
    --ga-step-size 0.012 \
    --nrestarts 5 \
    --ntotalruns 5 \
    --bonds \
    --angles \
    --torsions \
    --tuning-ratio-target 0.9 \
    --c-tuning-initial 0.001
```

## Standalone IAM Calculation Tool

The `calculate_iam.py` script is a standalone tool for calculating IAM (Independent Atom Model) scattering signals from XYZ files without running the full simulated annealing optimization. This is useful for:

- **Quick IAM calculations** from single structures or trajectories
- **Generating reference signals** for comparison
- **Testing scattering calculations** independently
- **Batch processing** multiple structures

### Basic Usage

Calculate IAM signal from a single XYZ file:
```bash
python3 calculate_iam.py input.xyz output.dat
```

Calculate IAM from an XYZ trajectory (multiple structures):
```bash
python3 calculate_iam.py trajectory.xyz output.dat
```

### Q-Vector Parameters

Configure the q-vector range and resolution:
```bash
python3 calculate_iam.py input.xyz output.dat \
    --qmin 0.1 \
    --qmax 8.0 \
    --qlen 100
```

### PCD Mode (Percentage Change Difference)

Calculate PCD signal relative to a reference structure:
```bash
python3 calculate_iam.py input.xyz output.dat \
    --reference reference.xyz \
    --pcd
```

PCD is calculated as: `PCD = 100 × (IAM / reference_IAM - 1)`

### Inelastic Scattering

Include Compton (inelastic) scattering contributions:
```bash
python3 calculate_iam.py input.xyz output.dat --inelastic
```

**Note**: Requires `data/Compton_Scattering_Intensities.npz` file. If not found, the script will continue without inelastic scattering.

### Ewald Sphere Mode

Calculate 3D scattering in Ewald sphere mode:
```bash
python3 calculate_iam.py input.xyz output.dat --ewald
```

Configure Ewald parameters (theta and phi angles):
```bash
python3 calculate_iam.py input.xyz output.dat \
    --ewald \
    --tmin 0.0 \
    --tmax 1.0 \
    --tlen 21 \
    --pmin 0.0 \
    --pmax 2.0 \
    --plen 21
```

**Note**: Theta and phi are specified in units of π (e.g., `tmax=1.0` means π radians).

### Combined Options

Example with all options enabled:
```bash
python3 calculate_iam.py input.xyz output.dat \
    --reference reference.xyz \
    --pcd \
    --inelastic \
    --ewald \
    --qmin 0.1 \
    --qmax 8.0 \
    --qlen 100 \
    --tmin 0.0 \
    --tmax 1.0 \
    --tlen 21 \
    --pmin 0.0 \
    --pmax 2.0 \
    --plen 21
```

### Output Format

- **Single structure**: Output file contains two columns: `q (Å⁻¹)` and `IAM signal`
- **Trajectory**: Output file contains multiple columns: `q (Å⁻¹)`, `IAM_1`, `IAM_2`, `IAM_3`, ...

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_xyz` | required | - | Input XYZ file (single structure or trajectory) |
| `output_dat` | required | - | Output DAT file with IAM signal |
| `--qmin` | float | 0.1 | Minimum q value (Å⁻¹) |
| `--qmax` | float | 8.0 | Maximum q value (Å⁻¹) |
| `--qlen` | int | 100 | Number of q points |
| `--reference` | str | None | Reference XYZ file for PCD calculation |
| `--pcd` | flag | False | Calculate PCD instead of IAM |
| `--inelastic` | flag | False | Include inelastic (Compton) scattering |
| `--ewald` | flag | False | Use Ewald sphere mode (3D scattering) |
| `--tmin` | float | 0.0 | Minimum theta (units of π) |
| `--tmax` | float | 1.0 | Maximum theta (units of π) |
| `--tlen` | int | 21 | Number of theta points |
| `--pmin` | float | 0.0 | Minimum phi (units of π) |
| `--pmax` | float | 2.0 | Maximum phi (units of π) |
| `--plen` | int | 21 | Number of phi points |

### Help

View all available options:
```bash
python3 calculate_iam.py --help
```

## Optimal Path Trajectory Tool

The `optimal_path.py` script finds the globally optimal "smoothest" trajectory through multiple candidate structures at each timestep. It uses dynamic programming to minimize a weighted sum of:

- **Fit factor**: Per-candidate fit quality (parsed from filename)
- **Signal delta**: Signal difference between consecutive timesteps (MSE on interpolated common q-grid)
- **Structure delta**: Structural difference between consecutive timesteps (Kabsch-aligned RMSD)

This is useful for:
- **Selecting optimal trajectories** from multiple candidate structures per timestep
- **Smoothing trajectories** by minimizing structural and signal jumps
- **Post-processing simulation results** to find the best path through candidate structures

### File Naming Convention

The script expects files with a specific naming pattern:
```
01_000.17533577.dat
01_000.17533577.xyz
02_000.46654335.dat
02_000.46654335.xyz
```

Where:
- First number (`01`, `02`, etc.) is the timestep
- Second number (`0.17533577`, etc.) is the fit factor (lower is better)
- Each timestep should have matching `.dat` and `.xyz` files

### Basic Usage

Find optimal path in a directory containing candidate files:
```bash
python3 optimal_path.py results/
```

This will:
1. Load all candidate structures and signals
2. Find the optimal path through timesteps
3. Write the selected trajectory to `optimal_trajectory.xyz`
4. Print the chosen path and statistics

### Random Sampling

**Randomly sample files per timestep before pruning** (useful for testing or exploring subsets):

```bash
python3 optimal_path.py results/ --random-sample 500
```

This randomly selects up to 500 files **per timestep** before applying topM/delta pruning. Useful for:
- Testing with smaller datasets
- Reducing computational cost
- Exploring different random subsets
- Quick prototyping
- Ensuring consistent coverage across all timesteps

**Note**: Sampling happens **per timestep** after grouping, so each timestep gets up to N files. If a timestep has fewer than N files, all files are kept. The same `--seed` is used for reproducibility.

**Combine with other options**:
```bash
python3 optimal_path.py results/ --random-sample 1000 --topM 50 --delta 0.1
```

This will:
1. Group files by timestep
2. Randomly sample up to 1000 files per timestep (seed from `--seed`)
3. Apply delta pruning (fit <= best_fit + 0.1)
4. Apply topM pruning (keep top 50)

### Pruning Options

Limit the number of candidates considered per timestep:

**Keep top M candidates** (by fit factor):
```bash
python3 optimal_path.py results/ --topM 50
```

**Keep candidates within a fit window**:
```bash
python3 optimal_path.py results/ --delta 0.1
```

This keeps candidates with `fit <= best_fit + 0.1` at each timestep.

**Combine both** (delta applied first, then topM):
```bash
python3 optimal_path.py results/ --delta 0.1 --topM 50
```

### Weight Configuration

Control the relative importance of fit, signal, and RMSD terms:

```bash
python3 optimal_path.py results/ \
    --fit-weight 1.0 \
    --signal-weight 1.0 \
    --rmsd-weight 1.0
```

**Auto-scaling** (enabled by default) normalizes weights so they're comparable:
```bash
python3 optimal_path.py results/  # Auto-scaling enabled
```

**Disable auto-scaling** to use raw weights:
```bash
python3 optimal_path.py results/ --no-autoscale
```

### RMSD Atom Selection

Select which atoms to include in RMSD calculations:

```bash
python3 optimal_path.py results/ --rmsd-indices "0,1,2,5"
```

Use comma-separated atom indices (0-based). If not specified, all atoms are used.

**Example**: Calculate RMSD using only heavy atoms or specific regions:
```bash
python3 optimal_path.py results/ --rmsd-indices "3,5,6,10,12"
```

### Output Options

**Custom output filename**:
```bash
python3 optimal_path.py results/ --xyz-out my_trajectory.xyz
```

**Auto-scaling parameters**:
```bash
python3 optimal_path.py results/ \
    --edge-sample-cap 5000 \
    --seed 42
```

- `--edge-sample-cap`: Number of random edges sampled for auto-scaling (default: 3000)
- `--seed`: Random seed for edge sampling (default: 0)

### Complete Example

```bash
python3 optimal_path.py results/ \
    --random-sample 2000 \
    --topM 100 \
    --delta 0.2 \
    --fit-weight 1.0 \
    --signal-weight 1.0 \
    --rmsd-weight 1.0 \
    --rmsd-indices "3,5,6,10,12" \
    --xyz-out optimal_trajectory.xyz \
    --edge-sample-cap 5000 \
    --seed 42
```

### Output

The script provides:
1. **Console output**: 
   - Auto-scaling information (if enabled)
   - Progress for each timestep transition
   - Optimal path with timestep, fit, and filenames
   - Unweighted totals (sum of fit factors, signal MSE, RMSD)
   - Best weighted cost

2. **XYZ trajectory file**: Multi-frame XYZ file containing the selected structures, with comment lines recording timestep and fit values.

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | required | - | Directory containing `*.dat` and `*.xyz` files |
| `--random-sample` | int | None | Randomly sample N files before pruning (uses `--seed` for reproducibility) |
| `--topM` | int | 100 | Number of lowest-fit candidates to keep per timestep |
| `--delta` | float | None | Fit window: keep candidates with `fit <= best_fit + delta` |
| `--fit-weight` | float | 1.0 | Weight for fit factor term |
| `--signal-weight` | float | 1.0 | Weight for signal MSE term |
| `--rmsd-weight` | float | 1.0 | Weight for RMSD term |
| `--no-autoscale` | flag | False | Disable automatic scaling of cost terms |
| `--rmsd-indices` | str | None | Comma-separated atom indices (0-based) for RMSD calculation |
| `--xyz-out` | str | optimal_trajectory.xyz | Output XYZ trajectory filename |
| `--edge-sample-cap` | int | 3000 | Number of edges sampled for auto-scaling |
| `--seed` | int | 0 | RNG seed for random sampling and auto-scale edge sampling |

### Help

View all available options:
```bash
python3 optimal_path.py --help
```

## Geometry Analysis Tool

The `analyze_geometry.py` script is a standalone tool for analyzing molecular geometry from XYZ files or trajectories. It can calculate:

- **Bond lengths**: Distance between two atoms
- **Angles**: Angle between three atoms (i-j-k)
- **Dihedral angles**: Torsion angle between four atoms (i-j-k-l)

This is useful for:
- **Analyzing molecular structures** from simulations or optimizations
- **Tracking geometric parameters** over trajectories
- **Comparing structures** by specific geometric features
- **Extracting key structural parameters** for analysis

### Basic Usage

Calculate bond length between atoms 0 and 1:
```bash
python3 analyze_geometry.py input.xyz --bond 0 1
```

Calculate angle between atoms 0, 1, and 2:
```bash
python3 analyze_geometry.py input.xyz --angle 0 1 2
```

Calculate dihedral angle between atoms 0, 1, 2, and 3:
```bash
python3 analyze_geometry.py input.xyz --dihedral 0 1 2 3
```

### Multiple Calculations

Calculate multiple geometric parameters in one run:
```bash
python3 analyze_geometry.py input.xyz \
    --bond 0 1 \
    --bond 1 2 \
    --angle 0 1 2 \
    --dihedral 0 1 2 3
```

### Trajectory Analysis

For XYZ trajectories (multiple structures), output to CSV:
```bash
python3 analyze_geometry.py trajectory.xyz \
    --bond 0 1 \
    --angle 0 1 2 \
    --output results.csv
```

The CSV file will contain one row per structure with all calculated values.

### Units

Bond lengths can be specified in Angstrom (default) or Bohr:
```bash
python3 analyze_geometry.py input.xyz --bond 0 1 --units bohr
```

Angles and dihedral angles are always reported in degrees.

### Output Format

**Single structure** (no `--output` specified):
- Human-readable format printed to stdout
- Shows structure comment and all calculated values

**Trajectory or with `--output`**:
- CSV format with one column per calculation (no header, no frame/comment columns)
- Each row represents one structure in the trajectory
- Column order matches the order of calculations specified (bonds first, then angles, then dihedrals)

### Atom Indices

**Important**: Atom indices are **0-based** (first atom is index 0, second is index 1, etc.).

To determine atom indices:
1. Open the XYZ file in a text editor
2. Count atoms starting from 0 (first atom = 0, second = 1, etc.)
3. Or use a molecular viewer that shows atom indices

### Examples

**Analyze a single structure**:
```bash
python3 analyze_geometry.py optimized.xyz \
    --bond 0 1 \
    --angle 0 1 2 \
    --dihedral 0 1 2 3
```

**Analyze a trajectory**:
```bash
python3 analyze_geometry.py simulation.xyz \
    --bond 0 1 \
    --angle 0 1 2 \
    --dihedral 0 1 2 3 \
    --output geometry_analysis.csv
```

**Multiple bonds and angles**:
```bash
python3 analyze_geometry.py structure.xyz \
    --bond 0 1 \
    --bond 1 2 \
    --bond 2 3 \
    --angle 0 1 2 \
    --angle 1 2 3 \
    --dihedral 0 1 2 3
```

### Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input_xyz` | required | Input XYZ file (single structure or trajectory) |
| `--bond I J` | int, int | Calculate bond length between atoms I and J (0-indexed). Can be specified multiple times. |
| `--angle I J K` | int, int, int | Calculate angle I-J-K in degrees (0-indexed). Can be specified multiple times. |
| `--dihedral I J K L` | int, int, int, int | Calculate dihedral angle I-J-K-L in degrees (0-indexed). Can be specified multiple times. |
| `--output OUTPUT` | str | Output file (CSV format). If not specified, prints to stdout. |
| `--units {angstrom,bohr}` | str | Units for bond lengths (default: angstrom) |

### Help

View all available options:
```bash
python3 analyze_geometry.py --help
```

## Random Subset Comparison Tool

The `compare_random_subsets.py` script automates the process of comparing multiple random subsets from your results directory. It:

1. Runs `optimal_path.py` multiple times with different random seeds
2. Analyzes specified geometric parameters (bonds, angles, dihedrals) from each optimal trajectory
3. Creates a comparison plot showing all subsets together

This is useful for:
- **Evaluating robustness**: See how different random subsets affect the optimal path
- **Statistical analysis**: Compare multiple trajectories from different data subsets
- **Quick comparisons**: Automate the workflow of running optimal_path → analyze → plot

### Basic Usage

Compare 5 random subsets (default) with dihedral 2-3-4-5 and random sample size 500:
```bash
python3 compare_random_subsets.py results/ --dihedral 2 3 4 5
```

Compare 10 random subsets:
```bash
python3 compare_random_subsets.py results/ --n-subsets 10 --dihedral 2 3 4 5
```

### Bonds, Angles, and Dihedrals

You can analyze bonds, angles, and/or dihedrals. At least one must be specified:

**Bond length**:
```bash
python3 compare_random_subsets.py results/ --bond 0 1
```

**Angle**:
```bash
python3 compare_random_subsets.py results/ --angle 0 1 2
```

**Dihedral angle**:
```bash
python3 compare_random_subsets.py results/ --dihedral 2 3 4 5
```

**Multiple calculations** (all will be plotted):
```bash
python3 compare_random_subsets.py results/ \
    --bond 0 1 \
    --angle 0 1 2 \
    --dihedral 2 3 4 5
```

### Aggregate (mean ± std) Plot

If you plan to run many subsets (e.g. 100s) and want a **single summary curve** instead of plotting every subset, use `--aggregate`.

This computes the **mean** and **standard deviation** across subsets at each timepoint and plots **mean ± 1σ** as a shaded band.

```bash
python3 compare_random_subsets.py results/ \
    --n-subsets 200 \
    --random-sample 500 \
    --topM 50 \
    --dihedral 2 3 4 5 \
    --aggregate \
    --output-plot dihedral_mean_std.png
```

After plotting, the script also selects a **representative subset**: the subset whose dihedral trajectory is closest (smallest RMS difference) to the mean trajectory. The corresponding optimal-path XYZ is saved to:

- `subset_comparison/representative_trajectory.xyz` (or your chosen `--output-dir`)

### Custom Random Sample Size

Use a different random sample size (default is 500):
```bash
python3 compare_random_subsets.py results/ --random-sample 1000 --dihedral 2 3 4 5
```

### Custom Output

Specify output directory and plot filename:
```bash
python3 compare_random_subsets.py results/ \
    --output-dir my_comparison \
    --output-plot my_plot.png
```

If you **do not** set `--output-plot` (i.e. you leave it as the default), the script will auto-generate a descriptive filename based on your settings (e.g. `n-subsets`, `random-sample`, `topM`, geometry indices, `seed-start`, and whether `--aggregate` is enabled).

### Seed Control

Control the starting seed for reproducibility:
```bash
python3 compare_random_subsets.py results/ \
    --n-subsets 5 \
    --seed-start 42
```

Each subset uses `seed_start + subset_index` (e.g., 42, 43, 44, 45, 46).

### Complete Example

```bash
python3 compare_random_subsets.py results/ \
    --n-subsets 8 \
    --random-sample 500 \
    --topM 50 \
    --seed-start 0 \
    --bond 0 1 \
    --angle 0 1 2 \
    --dihedral 2 3 4 5 \
    --output-dir subset_analysis \
    --output-plot geometry_comparison.png
```

### What It Does

1. **Runs optimal_path.py** for each subset with:
   - `--random-sample N` (configurable via `--random-sample`, default: 500)
   - `--topM M` (configurable via `--topM`, default: 50)
   - `--fit-weight 0`
   - `--rmsd-weight 1`
   - `--signal-weight 1`
   - `--no-autoscale`
   - Different seed for each subset

2. **Analyzes geometry** using `analyze_geometry.py`:
   - Extracts the specified bonds, angles, and/or dihedrals from each trajectory
   - Saves results as CSV files (one column per calculation)

3. **Creates comparison plot** using `plot_geometry.py`:
   - Plots all subsets on the same graph
   - If multiple calculations are specified, plots each calculation type separately
   - Labels each subset with its seed and calculation type
   - Saves as PNG

### Output

- **Trajectory files**: `optimal_trajectory_subset_0.xyz`, `optimal_trajectory_subset_1.xyz`, ...
- **Analysis files**: `geometry_subset_0.csv`, `geometry_subset_1.csv`, ... (contains one column per calculation)
- **Comparison plot**: `geometry_comparison.png` (or custom filename)

All files are saved in the output directory (default: `subset_comparison/`).

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | required | - | Directory containing `*.dat` and `*.xyz` files |
| `--n-subsets` | int | 5 | Number of random subsets to compare |
| `--random-sample` | int | 500 | Number of files to randomly sample for each subset |
| `--topM` | int | 50 | Number of lowest-fit candidates to keep per timestep |
| `--seed-start` | int | 0 | Starting seed (each subset uses seed_start + index) |
| `--output-dir` | str | subset_comparison | Output directory for intermediate files |
| `--output-plot` | str | geometry_comparison.png | Output plot filename |
| `--bond` | int, int | None | Bond atom indices (I J). At least one of --bond, --angle, or --dihedral must be specified. |
| `--angle` | int, int, int | None | Angle atom indices (I J K). At least one of --bond, --angle, or --dihedral must be specified. |
| `--dihedral` | int, int, int, int | None | Dihedral atom indices (I J K L). At least one of --bond, --angle, or --dihedral must be specified. |

### Help

View all available options:
```bash
python3 compare_random_subsets.py --help
```

## Fit Value Histogram Tool

The `plot_fit_histograms.py` script creates histograms of fit values from filenames for each timestep separately. This helps you:

- **Visualize fit distributions** across timesteps
- **Identify converged regions** where most good runs fall
- **Determine outlier thresholds** for filtering
- **Compare fit distributions** between timesteps

### Basic Usage

Plot histograms for all timesteps:
```bash
python3 plot_fit_histograms.py results/
```

### Custom Output

Specify output filename:
```bash
python3 plot_fit_histograms.py results/ --output my_histograms.png
```

### Adjust Bin Count

Change the number of histogram bins:
```bash
python3 plot_fit_histograms.py results/ --bins 100
```

### Output

The script creates:
1. **Histogram plot**: A grid of histograms, one per timestep, showing:
   - Distribution of fit values
   - Statistics box (min, Q25, median, mean, Q75, max)
   - Number of files per timestep

2. **Summary statistics**: Printed to console with:
   - Count of files per timestep
   - Min, Q25, median, Q75, max fit values per timestep

### Example Output

```
Summary Statistics (across 20 timesteps):
============================================================
Timestep     Count      Min          Q25          Median       Q75          Max         
------------------------------------------------------------
01           1500       0.123456     0.234567     0.345678     0.456789     0.987654
02           1500       0.112345     0.223456     0.334567     0.445678     0.876543
...
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | required | - | Directory containing `*.dat` and `*.xyz` files |
| `--output` | str | fit_histograms.png | Output plot filename |
| `--bins` | int | 50 | Number of bins for histograms |
| `--dat-ext` | str | .dat | Extension for DAT files |
| `--xyz-ext` | str | .xyz | Extension for XYZ files |

### Help

View all available options:
```bash
python3 plot_fit_histograms.py --help
```

## Geometry Plotting Tool

The `plot_geometry.py` script creates plots from CSV files generated by `analyze_geometry.py`. It can visualize geometric parameters over time (or frame number) and supports multiple customization options. **It can plot multiple files simultaneously for comparison.**

### Basic Usage

Plot column 0 from a single CSV file:
```bash
python3 plot_geometry.py data.csv output.png
```

Plot specific columns (0-indexed) from a single file:
```bash
python3 plot_geometry.py data.csv output.png --columns 0 1
```

### Multiple File Comparison

Plot the same column from multiple files for comparison:
```bash
python3 plot_geometry.py file1.csv file2.csv file3.csv output.png
```

This will plot column 0 from each file. Use `--columns` to specify which column(s) to plot from all files:
```bash
python3 plot_geometry.py file1.csv file2.csv output.png --columns 0
```

Custom labels for each file:
```bash
python3 plot_geometry.py run1.csv run2.csv run3.csv output.png \
    --labels "Run 1" "Run 2" "Run 3" \
    --title "Comparison of Multiple Runs"
```

### Customization

**Labels and titles**:
```bash
python3 plot_geometry.py data.csv output.png \
    --title "Bond Length Evolution" \
    --xlabel "time (fs)" \
    --ylabel "Bond Length (Å)" \
    --labels "C-C bond" "C-H bond"
```

**Note**: The x-axis defaults to "time (fs)" where each frame represents 20 fs, with the first frame at 10 fs. The plot starts at 0 fs to show blank space before the first data point. You can override the x-axis label with `--xlabel` if needed.

**Axis limits**:
```bash
python3 plot_geometry.py data.csv output.png \
    --xmin 0 --xmax 100 \
    --ymin 1.0 --ymax 2.0
```

**Plot style**:
```bash
# Line plot (default)
python3 plot_geometry.py data.csv output.png --style line

# Scatter plot
python3 plot_geometry.py data.csv output.png --style scatter

# Both line and scatter
python3 plot_geometry.py data.csv output.png --style both
```

**Figure size and resolution**:
```bash
python3 plot_geometry.py data.csv output.png \
    --figsize 12 8 \
    --dpi 300
```

**Add grid**:
```bash
python3 plot_geometry.py data.csv output.png --grid
```

### Complete Example

Analyze geometry and plot results:
```bash
# Step 1: Analyze trajectory
python3 analyze_geometry.py trajectory.xyz \
    --bond 0 1 \
    --angle 0 1 2 \
    --output geometry_data.csv

# Step 2: Plot the results
python3 plot_geometry.py geometry_data.csv geometry_plot.png \
    --title "Molecular Geometry Evolution" \
    --xlabel "time (fs)" \
    --ylabel "Value" \
    --labels "Bond 0-1 (Å)" "Angle 0-1-2 (°)" \
    --grid \
    --style line
```

Compare multiple trajectories:
```bash
# Analyze multiple trajectories
python3 analyze_geometry.py traj1.xyz --bond 0 1 --output traj1.csv
python3 analyze_geometry.py traj2.xyz --bond 0 1 --output traj2.csv
python3 analyze_geometry.py traj3.xyz --bond 0 1 --output traj3.csv

# Plot all together for comparison
python3 plot_geometry.py traj1.csv traj2.csv traj3.csv comparison.png \
    --title "Bond Length Comparison" \
    --ylabel "Bond Length (Å)" \
    --labels "Run 1" "Run 2" "Run 3" \
    --grid
```

### Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input_csv` | required | Input CSV file(s) (from analyze_geometry.py, no header). Can specify multiple files for comparison. |
| `output_png` | required | Output PNG file |
| `--columns` | int+ | Column indices to plot (0-indexed). Applies to all files. If not specified, plots column 0 from each file. |
| `--xlabel` | str | X-axis label (default: "time (fs)") |
| `--ylabel` | str | Y-axis label (default: "Value") |
| `--labels` | str+ | Labels for each column (legend). Number should match number of columns plotted. |
| `--title` | str | Plot title |
| `--xmin`, `--xmax` | float | X-axis limits |
| `--ymin`, `--ymax` | float | Y-axis limits |
| `--figsize` | float, float | Figure size in inches (default: 10 6) |
| `--dpi` | int | Output resolution in DPI (default: 300) |
| `--style` | str | Plot style: line, scatter, or both (default: line) |
| `--grid` | flag | Show grid on plot |

### Help

View all available options:
```bash
python3 plot_geometry.py --help
```

## Parameter Groups

> **Note**: Detailed explanations of all parameters, including their units, purpose, and relationships, are provided as comments in the `input.toml` configuration file. Refer to that file for comprehensive parameter documentation.

### Run Parameters
- `--run-id`: Run identifier
- `--molecule`: Molecule name (must match entry in `molecule_params` section)
- `--results-dir`: Directory for output files

### File Parameters
- `--forcefield-file`: Force field file path
- `--start-xyz-file`: Starting XYZ structure
- `--start-sdf-file`: Starting SDF structure
- `--reference-xyz-file`: Reference XYZ structure
- `--target-file`: Target XYZ or DAT file

### Options
- `--run-pyscf-modes`: Enable PySCF normal modes calculation
- `--pyscf-basis`: Basis set for PySCF (e.g., "6-31g*", "sto-3g")
- `--verbose`: Enable verbose output
- `--write-dat-file`: Write DAT file output
- `--mm-param-method`: MM parameter retrieval method (see "MM Parameter Retrieval Methods" below)

### Sampling Parameters
- `--sampling`: Enable sampling
- `--boltzmann-temperature`: Temperature for Boltzmann sampling

### Scattering Parameters
- `--inelastic`: Enable inelastic scattering
- `--pcd-mode`: Enable PCD mode
- `--excitation-factor`: Excitation factor
- `--ewald-mode`: Enable Ewald mode
- `--qmin`, `--qmax`, `--qlen`: Q-vector parameters
- `--tmin`, `--tmax`, `--tlen`: Theta parameters
- `--pmin`, `--pmax`, `--plen`: Phi parameters
- `--noise-value`: Noise value
- `--noise-data-file`: Noise data file path

### Simulated Annealing Parameters
- `--sa-starting-temp`: Starting temperature
- `--sa-nsteps`: Number of SA steps
- `--greedy-algorithm`: Use greedy algorithm
- `--ga-nsteps`: Number of GA steps
- `--sa-step-size`: Step size for SA
- `--ga-step-size`: Step size for GA
- `--nrestarts`: Number of restarts
- `--ntotalruns`: Total number of runs
- `--bonds`: Enable bond constraints
- `--angles`: Enable angle constraints
- `--torsions`: Enable torsion constraints
- `--tuning-ratio-target`: Target tuning ratio
- `--c-tuning-initial`: Initial C tuning value
- `--non-h-modes-only`: Only use non-hydrogen modes
- `--hf-energy`: Run PySCF HF energy calculation

## MM Parameter Retrieval Methods

The program supports two methods for retrieving molecular mechanics (MM) force field parameters:

### Method Selection

You can choose the method using the `mm_param_method` parameter in `input.toml`:

```toml
[options]
mm_param_method = "sdf"  # or "basic"
```

Or via command line:
```bash
python3 run.py --mm-param-method sdf
```

### Method 1: SDF Method (`mm_param_method = "sdf"`)

**Default method.** This method attempts to use force field parameters from OpenFF Toolkit by working with SDF files.

**Process:**
1. **Robust SDF method**: Tries to create an OpenMM system from the SDF file with:
   - Automatic radical fixing (removes radicals that OpenFF doesn't support)
   - Bond order simplification (converts double/triple/aromatic bonds to single bonds if needed)
   - Direct RDKit-to-OpenFF conversion
2. **Original SDF method**: If robust SDF fails, tries the original SDF method (standard OpenFF parameterization)
3. **Basic method fallback**: If both SDF methods fail, automatically falls back to the basic method (geometry extraction with generic force constants)

**When to use:**
- **Default choice** for most molecules
- When you want to use accurate force field parameters from OpenFF
- When your molecule is compatible with OpenFF (or can be made compatible with radical fixing/bond simplification)
- When you have an SDF file or can create one from your XYZ file

**Advantages:**
- Uses accurate, atom-type-specific force field parameters
- Handles radicals and unusual bond orders automatically
- Falls back gracefully to basic method if needed

**Requirements:**
- SDF file (will be created from XYZ if not present)
- OpenFF Toolkit installed
- Force field file (e.g., `openff_unconstrained-2.0.0.offxml`)

### Method 2: Basic Method (`mm_param_method = "basic"`)

This method extracts parameters directly from the molecular geometry using generic force constants, bypassing OpenFF entirely.

**Process:**
1. Reads the starting XYZ structure
2. Identifies bonds, angles, and torsions from atomic connectivity
3. Uses current geometry as equilibrium values
4. Applies generic force constants (see "Generic Force Constants" section below)

**When to use:**
- When OpenFF parameterization consistently fails
- When you want to skip SDF file creation/conversion
- When you need guaranteed parameter extraction (no dependency on OpenFF)
- For quick testing or when approximate parameters are sufficient
- For molecules with unusual structures that OpenFF cannot handle

**Advantages:**
- **Always works**: No dependency on OpenFF compatibility
- **Fast**: No SDF conversion or OpenFF parameterization
- **Simple**: Direct geometry-based extraction
- **Reliable**: Guaranteed to produce parameters

**Limitations:**
- Uses generic force constants (not atom-type-specific)
- Less accurate than force field parameters
- See "Generic Force Constants" section for details

### Comparison

| Feature | SDF Method | Basic Method |
|---------|------------|--------------|
| **Accuracy** | High (force field parameters) | Moderate (generic constants) |
| **Speed** | Slower (SDF conversion + OpenFF) | Faster (direct extraction) |
| **Reliability** | May fail for unusual molecules | Always works |
| **Dependencies** | Requires OpenFF + SDF | No special dependencies |
| **Fallback** | Falls back to basic if needed | No fallback needed |
| **Use case** | Production runs, accurate parameters | Quick testing, guaranteed extraction |

### Recommendations

- **Start with SDF method** (`mm_param_method = "sdf"`): It will automatically fall back to basic method if needed
- **Use basic method** if:
  - You consistently get errors with SDF method
  - You're doing quick tests and don't need high accuracy
  - You want to avoid SDF file dependencies
  - Your molecule has unusual structures that OpenFF cannot handle

## Generic Force Constants (Fallback Method)

When the program cannot obtain molecular mechanics (MM) force field parameters from OpenFF Toolkit (e.g., due to radicals, unusual bond orders, or other compatibility issues), it automatically falls back to a **geometry-based parameter extraction method** that uses generic force constants.

### When Are Generic Force Constants Used?

The generic force constants are used in two scenarios:

1. **When `mm_param_method = "basic"`**: Used directly as the primary method
2. **When `mm_param_method = "sdf"`**: Used as a fallback when:
   - The robust SDF method fails (after radical fixing and bond simplification)
   - The original SDF method fails

If the SDF method succeeds, the program uses the force field parameters from OpenFF Toolkit instead of generic constants.

### What Are the Generic Force Constants?

The fallback method extracts parameters directly from your starting geometry:

- **Equilibrium values**: Calculated from the current XYZ coordinates
  - Bond lengths: Current distances between bonded atoms
  - Bond angles: Current angles between three connected atoms
  - Torsion angles: Current dihedral angles between four connected atoms

- **Force constants**: Generic values based on typical molecular mechanics force fields:

#### Bond Force Constants
```
k_bond = 400.0 + 50.0 × (average atomic number)  [kcal/mol/Å²]
```

**Examples:**
- C-C bond (both atomic number 6): `400 + 50×6 = 700 kcal/mol/Å²`
- C-H bond (avg atomic number ~3.5): `400 + 50×3.5 = 575 kcal/mol/Å²`
- C-O bond (avg atomic number ~7): `400 + 50×7 = 750 kcal/mol/Å²`

**Reference**: Typical C-C bonds in OpenFF are ~529 kcal/mol/Å², so these values are in a similar range (slightly higher for safety).

#### Angle Force Constants
```
k_angle = 60.0  [kcal/mol/rad²]
```

This is a fixed value for all angles. Typical range in force fields: 50-100 kcal/mol/rad².

#### Torsion Force Constants
```
k_torsion = 2.0  [kcal/mol]
```

This is a fixed value for all torsions. Typical range in force fields: 0.1-5 kcal/mol.

### Where Do These Values Come From?

These generic force constants are based on typical values from standard molecular mechanics force fields:

- **AMBER**: Bond constants ~300-600 kcal/mol/Å², angle constants ~40-100 kcal/mol/rad², torsion constants ~0.1-3 kcal/mol
- **CHARMM**: Similar ranges to AMBER
- **OpenFF**: Bond constants ~400-800 kcal/mol/Å², angle constants ~50-100 kcal/mol/rad², torsion constants ~0.1-5 kcal/mol

The values chosen are:
- **Conservative** (slightly higher than typical) to provide reasonable constraints
- **Within typical ranges** used in standard force fields
- **Simple heuristics** that work for most organic molecules

### Limitations

The generic force constants are **approximations** and have the following limitations:

1. **Not atom-type specific**: All C-C bonds get the same force constant, regardless of hybridization (sp³ vs sp²)
2. **Not bond-order specific**: Single, double, and triple bonds are treated the same
3. **Not environment-specific**: Aromatic vs aliphatic bonds are treated the same
4. **Fixed angle/torsion constants**: All angles and torsions use the same force constant regardless of atom types

### Why They Work

Despite these limitations, the generic force constants work well for simulated annealing because:

1. **Reasonable constraints**: They provide sufficient constraints to keep the structure near the starting geometry
2. **Typical ranges**: The values are within typical force field ranges
3. **Equilibrium from geometry**: The equilibrium values come from your actual starting structure, so they're appropriate for your molecule
4. **Simulated annealing tolerance**: For optimization purposes, approximate constraints are often sufficient

### How to Know If Generic Force Constants Are Being Used

When the fallback method is used, you'll see output like:
```
Both robust method and SDF method failed!
Attempting final fallback: extracting parameters directly from geometry...
(This uses starting geometry as equilibrium values with generic force constants)
Successfully extracted parameters from geometry!
Found X bonds, Y angles, Z torsions
Note: Using generic force constants and starting geometry as equilibrium values.
```

If you see this message, the program is using generic force constants instead of OpenFF parameters.

## Notes

1. **Default Values**: All parameters have default values defined in the TOML config file (`input.toml` by default). Command-line arguments override these defaults.

2. **Boolean Flags**: Boolean parameters (like `--verbose`, `--inelastic`) are enabled by including the flag. They default to `False` if not specified.

3. **Parameter Overrides**: Only parameters explicitly provided on the command line override the TOML file values. Unspecified parameters use TOML defaults.

4. **Mode Validation**: The `--mode` argument must be either `"normal"` or `"test"` (case-insensitive).

5. **Molecule Parameters**: The `--molecule` argument must match a section in `molecule_params` in the TOML file (e.g., `[molecule_params.chd]`).

## Viewing All Options

To see all available command-line options with descriptions:
```bash
python3 run.py --help
```

This will display all arguments organized by group with their descriptions and default values.

## Testing

Run the test suite:
```bash
pytest tests/
```

For more information on tests, see `tests/README.md`.
