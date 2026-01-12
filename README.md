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
