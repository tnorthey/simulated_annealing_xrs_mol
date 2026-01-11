# Command Line Usage Examples

This document provides examples of how to run `run.py` with various command-line arguments.

## Basic Usage

### Run with default configuration (uses `input.toml`)
```bash
python3 run.py
```

### Run with a different config file
```bash
python3 run.py --config my_config.toml
```

### Show help message
```bash
python3 run.py --help
```

## Common Examples

### 1. Basic run with mode and molecule override
```bash
python3 run.py --mode xyz --molecule chd --run-id test_run
```

### 2. Override file paths
```bash
python3 run.py \
    --start-xyz-file xyz/my_start.xyz \
    --reference-xyz-file xyz/my_reference.xyz \
    --target-file xyz/my_target.xyz \
    --forcefield-file forcefields/my_forcefield.offxml
```

### 3. Enable PySCF normal modes calculation
```bash
python3 run.py --run-pyscf-modes --pyscf-basis 6-31g
```

### 4. Enable verbose output
```bash
python3 run.py --verbose
```

### 5. Configure scattering parameters
```bash
python3 run.py \
    --qmin 0.1 \
    --qmax 8.0 \
    --qlen 81 \
    --inelastic \
    --pcd-mode
```

### 6. Configure simulated annealing parameters
```bash
python3 run.py \
    --sa-starting-temp 1.0 \
    --sa-nsteps 4000 \
    --sa-step-size 0.012 \
    --greedy-algorithm \
    --ga-nsteps 20000 \
    --ga-step-size 0.012
```

### 7. Enable bonds, angles, and torsions
```bash
python3 run.py --bonds --angles --torsions
```

### 8. Configure restart parameters
```bash
python3 run.py \
    --nrestarts 5 \
    --ntotalruns 5
```

### 9. Enable Ewald mode for scattering
```bash
python3 run.py --ewald-mode
```

### 10. Configure theta and phi parameters (for 2D/3D scattering)
```bash
python3 run.py \
    --tmin 0.0 \
    --tmax 1.0 \
    --tlen 21 \
    --pmin 0.0 \
    --pmax 2.0 \
    --plen 21
```

### 11. Enable sampling with Boltzmann temperature
```bash
python3 run.py --sampling --boltzmann-temperature 300.0
```

### 12. Configure noise parameters
```bash
python3 run.py \
    --noise-value 0.05 \
    --noise-data-file noise/my_noise.dat
```

### 13. Tuning parameters for simulated annealing
```bash
python3 run.py \
    --tuning-ratio-target 0.9 \
    --c-tuning-initial 0.001
```

### 14. Enable non-hydrogen modes only
```bash
python3 run.py --non-h-modes-only
```

### 15. Enable PySCF HF energy calculation
```bash
python3 run.py --hf-energy
```

### 16. Write DAT file output
```bash
python3 run.py --write-dat-file
```

## Complete Example: Full Configuration Override

```bash
python3 run.py \
    --config input.toml \
    --mode xyz \
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

Arguments are organized into logical groups:

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

## Notes

1. **Default Values**: All parameters have default values defined in the TOML config file (`input.toml` by default). Command-line arguments override these defaults.

2. **Boolean Flags**: Boolean parameters (like `--verbose`, `--inelastic`) are enabled by including the flag. They default to `False` if not specified.

3. **Parameter Overrides**: Only parameters explicitly provided on the command line override the TOML file values. Unspecified parameters use TOML defaults.

4. **Mode Validation**: The `--mode` argument must be either `"dat"` or `"xyz"` (case-insensitive).

5. **Molecule Parameters**: The `--molecule` argument must match a section in `molecule_params` in the TOML file (e.g., `[molecule_params.chd]`).

## Quick Reference

View all available options:
```bash
python3 run.py --help
```

This will show all available arguments organized by group with their descriptions and default values.
