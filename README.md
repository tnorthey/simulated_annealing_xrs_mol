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
| `--topM` | int | 100 | Number of lowest-fit candidates to keep per timestep |
| `--delta` | float | None | Fit window: keep candidates with `fit <= best_fit + delta` |
| `--fit-weight` | float | 1.0 | Weight for fit factor term |
| `--signal-weight` | float | 1.0 | Weight for signal MSE term |
| `--rmsd-weight` | float | 1.0 | Weight for RMSD term |
| `--no-autoscale` | flag | False | Disable automatic scaling of cost terms |
| `--rmsd-indices` | str | None | Comma-separated atom indices (0-based) for RMSD calculation |
| `--xyz-out` | str | optimal_trajectory.xyz | Output XYZ trajectory filename |
| `--edge-sample-cap` | int | 3000 | Number of edges sampled for auto-scaling |
| `--seed` | int | 0 | RNG seed for auto-scale edge sampling |

### Help

View all available options:
```bash
python3 optimal_path.py --help
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

## Generic Force Constants (Fallback Method)

When the program cannot obtain molecular mechanics (MM) force field parameters from OpenFF Toolkit (e.g., due to radicals, unusual bond orders, or other compatibility issues), it automatically falls back to a **geometry-based parameter extraction method** that uses generic force constants.

### When Are Generic Force Constants Used?

The generic force constants are used as a **last-resort fallback** when:
1. The robust XYZ-to-OpenFF method fails (after radical fixing and bond simplification)
2. The robust SDF method fails (after radical fixing and bond simplification)
3. The original SDF method fails

If any of these methods succeed, the program uses the force field parameters from OpenFF Toolkit instead.

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
