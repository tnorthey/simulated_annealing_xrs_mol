# import math
import numpy as np
import sys
import pprint
import os

# Try to use tomllib (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Please install 'tomli' package: pip install tomli")


######
class Input_to_params:
    """read parameters for simulated annealing"""

    def __init__(self, input_toml_file, overrides=None):
        """initialise with TOML file and optional argparse overrides
        
        Args:
            input_toml_file: Path to TOML input file
            overrides: Dict of parameter overrides from argparse (optional)
        """

        ###################################
        # Load input TOML
        try:
            with open(input_toml_file, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            print(f"\n{'='*60}")
            print("ERROR: Input file not found")
            print(f"{'='*60}")
            print(f"  File: {input_toml_file}")
            print(f"  Suggestion: Check the file path is correct")
            print(f"{'='*60}\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n{'='*60}")
            print("ERROR: Failed to parse TOML file")
            print(f"{'='*60}")
            print(f"  File: {input_toml_file}")
            print(f"  Error: {type(e).__name__}: {e}")
            print(f"  Suggestion: Check the TOML file syntax is correct")
            print(f"{'='*60}\n")
            sys.exit(1)
        
        # Apply overrides if provided (merge into data structure)
        if overrides:
            self._apply_overrides(data, overrides)
        
        # Handle reference_dat_file: ensure it's None if empty string
        if "files" in data and "reference_dat_file" in data["files"]:
            ref_dat = str(data["files"]["reference_dat_file"])
            if ref_dat == "":
                data["files"]["reference_dat_file"] = None
        
        ### Parameters
        # mode
        try:
            self.mode = str(data["mode"])
        except KeyError:
            print(f"\n{'='*60}")
            print("ERROR: Missing required parameter in TOML file")
            print(f"{'='*60}")
            print(f"  Missing key: 'mode'")
            print(f"  Suggestion: Add 'mode = \"test\"' or 'mode = \"normal\"' to input file")
            print(f"{'='*60}\n")
            sys.exit(1)
        
        mode = self.mode.lower()  # lower case mode string
        ## handle the case when mode does not equal "normal" or "test"
        try:
            if not (mode == "normal" or mode == "test"):
                raise ValueError(
                    'mode value must equal "normal" or "test"! (case insensitive). Exiting...'
                )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  # exit program with error message.
        
        # run params
        try:
            self.run_id = str(data["run_params"]["run_id"])
            self.molecule = str(data["run_params"]["molecule"])
            self.results_dir = str(data["run_params"]["results_dir"])
        except KeyError as e:
            print(f"\n{'='*60}")
            print("ERROR: Missing required parameter in TOML file")
            print(f"{'='*60}")
            print(f"  Missing key: {e}")
            print(f"  Suggestion: Add [run_params] section with run_id, molecule, and results_dir")
            print(f"{'='*60}\n")
            sys.exit(1)
        # options
        self.run_pyscf_modes_bool = bool(data["options"]["run_pyscf_modes_bool"])
        self.pyscf_basis = str(data["options"]["pyscf_basis"])
        self.verbose_bool = bool(data["options"]["verbose_bool"])
        self.write_dat_file_bool = bool(data["options"]["write_dat_file_bool"])
        self.mm_param_method = str(data["options"]["mm_param_method"]).lower()
        self.use_pre_molecular = bool(data["options"].get("use_pre_molecular", False))
        # Validate mm_param_method
        if self.mm_param_method not in ["sdf", "basic"]:
            print(f"\n{'='*60}")
            print("ERROR: Invalid mm_param_method value")
            print(f"{'='*60}")
            print(f"  Value: {data['options']['mm_param_method']}")
            print(f"  Allowed values: 'sdf' or 'basic'")
            print(f"  Suggestion: Set mm_param_method = 'sdf' or 'basic' in [options] section")
            print(f"{'='*60}\n")
            sys.exit(1)
        # sampling options
        self.sampling_bool = bool(data["sampling"]["sampling_bool"])
        self.boltzmann_temperature = float(data["sampling"]["boltzmann_temperature"])
        # file params
        self.forcefield_file = str(data["files"]["forcefield_file"])
        self.start_xyz_file = str(data["files"]["start_xyz_file"])
        self.start_sdf_file = str(data["files"]["start_sdf_file"])
        self.reference_xyz_file = str(data["files"]["reference_xyz_file"])
        # reference_dat_file is optional
        if "reference_dat_file" in data["files"]:
            self.reference_dat_file = str(data["files"]["reference_dat_file"])
            # Treat empty string as None
            if self.reference_dat_file == "":
                self.reference_dat_file = None
        else:
            self.reference_dat_file = None
        self.target_file = str(data["files"]["target_file"])
        # scattering_params params
        self.inelastic = bool(data["scattering_params"]["inelastic_bool"])
        self.pcd_mode = bool(data["scattering_params"]["pcd_mode_bool"])
        self.excitation_factor = float(data["scattering_params"]["excitation_factor"])
        # ewald params
        self.ewald_mode = bool(data["scattering_params"]["ewald"]["ewald_mode_bool"])
        # radial q params
        self.qmin = float(data["scattering_params"]["q"]["qmin"])
        self.qmax = float(data["scattering_params"]["q"]["qmax"])
        self.qlen = int(data["scattering_params"]["q"]["qlen"])
        # theta params (from ewald section)
        self.tmin = float(data["scattering_params"]["ewald"]["th"]["tmin"])
        self.tmax = float(data["scattering_params"]["ewald"]["th"]["tmax"])
        self.tlen = int(data["scattering_params"]["ewald"]["th"]["tlen"])
        # phi params (from ewald section)
        self.pmin = float(data["scattering_params"]["ewald"]["ph"]["pmin"])
        self.pmax = float(data["scattering_params"]["ewald"]["ph"]["pmax"])
        self.plen = int(data["scattering_params"]["ewald"]["ph"]["plen"])
        # noise params
        self.noise_value = float(data["scattering_params"]["noise"]["noise_value"])
        self.noise_data_file = str(
            data["scattering_params"]["noise"]["noise_data_file"]
        )
        # simulated annealing params
        self.sa_starting_temp = float(
            data["simulated_annealing_params"]["sa_starting_temp"]
        )
        self.sa_nsteps = int(data["simulated_annealing_params"]["sa_nsteps"])
        self.greedy_algorithm_bool = bool(
            data["simulated_annealing_params"]["greedy_algorithm_bool"]
        )
        self.ga_nsteps = int(data["simulated_annealing_params"]["ga_nsteps"])
        self.sa_step_size = float(data["simulated_annealing_params"]["sa_step_size"])
        self.ga_step_size = float(data["simulated_annealing_params"]["ga_step_size"])
        self.ntotalruns = int(
            data["simulated_annealing_params"]["ntotalruns"]
        )  # repeat everything from the start n_totalruns times; this is here for efficiency, equivalent to setting this to 1 and rerunning the program many times
        self.nrestarts = int(
            data["simulated_annealing_params"]["nrestarts"]
        )  # it restarts from the xyz_best of the previous restart
        self.bonds_bool = bool(data["simulated_annealing_params"]["bonds_bool"])
        self.angles_bool = bool(data["simulated_annealing_params"]["angles_bool"])
        self.torsions_bool = bool(data["simulated_annealing_params"]["torsions_bool"])
        self.tuning_ratio_target = float(
            data["simulated_annealing_params"]["tuning_ratio_target"]
        )
        self.c_tuning_initial = float(
            data["simulated_annealing_params"]["c_tuning_initial"]
        )
        self.non_h_modes_only = bool(
            data["simulated_annealing_params"]["non_h_modes_only_bool"]
        )  # only include "non-hydrogen" modes
        self.hydrogen_mode_damping_factor = float(
            data["simulated_annealing_params"]["hydrogen_mode_damping_factor"]
        )  # damping factor for hydrogen modes
        self.hf_energy = bool(
            data["simulated_annealing_params"]["hf_energy_bool"]
        )  # run PySCF HF energy

        # molecule params
        molecule = self.molecule
        try:
            self.natoms = int(data["molecule_params"][molecule]["natoms"])
            self.nmodes = int(data["molecule_params"][molecule]["nmodes"])
            self.hydrogen_mode_range = np.array(
                data["molecule_params"][molecule]["hydrogen_mode_range"]
            )
            self.sa_mode_range = np.array(
                data["molecule_params"][molecule]["sa_mode_range"]
            )
            self.ga_mode_range = np.array(
                data["molecule_params"][molecule]["ga_mode_range"]
            )
            self.bond_ignore_array = np.array(
                data["molecule_params"][molecule]["bond_ignore_array"]
            )
            self.angle_ignore_array = np.array(
                data["molecule_params"][molecule]["angle_ignore_array"]
            )
            self.torsion_ignore_array = np.array(
                data["molecule_params"][molecule]["torsion_ignore_array"]
            )
            self.rmsd_indices = np.array(data["molecule_params"][molecule]["rmsd_indices"])
            self.bond_indices = np.array(data["molecule_params"][molecule]["bond_indices"])
            self.angle_indices = np.array(
                data["molecule_params"][molecule]["angle_indices"]
            )
            self.dihedral_indices = np.array(
                data["molecule_params"][molecule]["dihedral_indices"]
            )
        except KeyError as e:
            print(f"\n{'='*60}")
            print("ERROR: Missing required parameter in TOML file")
            print(f"{'='*60}")
            print(f"  Missing key: {e}")
            if "molecule_params" not in data:
                print(f"  Suggestion: Add [molecule_params.{molecule}] section to input file")
            elif molecule not in data["molecule_params"]:
                print(f"  Suggestion: Add molecule '{molecule}' to molecule_params section")
                print(f"  Available molecules: {list(data.get('molecule_params', {}).keys())}")
            else:
                print(f"  Suggestion: Add missing parameter to [molecule_params.{molecule}] section")
            print(f"{'='*60}\n")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            print(f"\n{'='*60}")
            print("ERROR: Invalid parameter value in TOML file")
            print(f"{'='*60}")
            print(f"  Error: {type(e).__name__}: {e}")
            print(f"  Molecule: {molecule}")
            print(f"  Suggestion: Check parameter types and values in [molecule_params.{molecule}] section")
            print(f"{'='*60}\n")
            sys.exit(1)

        ### Define other variables
        # qvector
        self.qvector = np.linspace(self.qmin, self.qmax, self.qlen, endpoint=True)
        # theta (units of pi)
        self.th = np.pi * np.linspace(self.tmin, self.tmax, self.tlen, endpoint=True)
        # phi (units of pi), note f(0) = f(2pi) so endpoint=False
        self.ph = np.pi * np.linspace(self.pmin, self.pmax, self.plen, endpoint=False)
        # specific mode indices
        self.hydrogen_mode_indices = np.arange(
            self.hydrogen_mode_range[0], self.hydrogen_mode_range[1]
        )
        self.sa_mode_indices = np.arange(self.sa_mode_range[0], self.sa_mode_range[1])
        self.ga_mode_indices = np.arange(self.ga_mode_range[0], self.ga_mode_range[1])

        # Validate all parameters
        self._validate_parameters(data)

        ### print out all attributes
        if self.verbose_bool:
            print("##################################################")
            print("### Initialised with the following parameters: ###")
            print("##################################################")
            pprint.pprint(vars(self))
            print("##################################################")

        ###################################
    
    def _apply_overrides(self, data, overrides):
        """Apply argparse overrides to data dictionary using nested key paths"""
        for key_path, value in overrides.items():
            if value is None:
                continue
            # Split the dotted path (e.g., "run_params.run_id")
            parts = key_path.split('.')
            current = data
            # Navigate to the parent dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            # Set the value
            current[parts[-1]] = value

    def _validate_parameters(self, data):
        """Validate all parameters and raise helpful errors if invalid"""
        errors = []
        warnings = []
        
        # 1. Check if molecule exists in molecule_params
        molecule = self.molecule
        if "molecule_params" not in data or molecule not in data["molecule_params"]:
            errors.append(
                f'Error: Molecule "{molecule}" not found in molecule_params section.\n'
                f'  Available molecules: {list(data.get("molecule_params", {}).keys())}\n'
                f'  Suggestion: Check spelling or add molecule section to input file.'
            )
        
        # 2. Validate file paths (warn if they don't exist yet)
        #
        # These files are often generated/placed later (or provided via CLI overrides),
        # so missing paths should not prevent configuration parsing.
        file_params = {
            "forcefield_file": self.forcefield_file,
            "start_xyz_file": self.start_xyz_file,
            "reference_xyz_file": self.reference_xyz_file,
            "target_file": self.target_file,
        }
        for param_name, file_path in file_params.items():
            if not os.path.exists(file_path):
                warnings.append(
                    f'Warning: File not found: {param_name} = "{file_path}"\n'
                    f'  Suggestion: This is OK for config validation/tests; ensure the file exists before running.'
                )
        
        # 3. Validate numeric ranges for scattering parameters
        if self.qmin >= self.qmax:
            errors.append(
                f'Error: qmin ({self.qmin}) must be less than qmax ({self.qmax})\n'
                f'  Suggestion: Set qmin < qmax (e.g., qmin=0.1, qmax=8.0)'
            )
        if self.qmin < 0:
            errors.append(
                f'Error: qmin ({self.qmin}) must be non-negative\n'
                f'  Suggestion: Set qmin >= 0 (typically qmin >= 1e-9)'
            )
        if self.qlen < 2:
            errors.append(
                f'Error: qlen ({self.qlen}) must be at least 2\n'
                f'  Suggestion: Set qlen >= 2 (typically 50-100)'
            )
        
        if self.ewald_mode:
            if self.tmin >= self.tmax:
                errors.append(
                    f'Error: tmin ({self.tmin}) must be less than tmax ({self.tmax})\n'
                    f'  Suggestion: Set tmin < tmax (e.g., tmin=0.0, tmax=1.0)'
                )
            if self.tlen < 2:
                errors.append(
                    f'Error: tlen ({self.tlen}) must be at least 2\n'
                    f'  Suggestion: Set tlen >= 2 (typically 10-50)'
                )
            if self.pmin >= self.pmax:
                errors.append(
                    f'Error: pmin ({self.pmin}) must be less than pmax ({self.pmax})\n'
                    f'  Suggestion: Set pmin < pmax (e.g., pmin=0.0, pmax=2.0)'
                )
            if self.plen < 2:
                errors.append(
                    f'Error: plen ({self.plen}) must be at least 2\n'
                    f'  Suggestion: Set plen >= 2 (typically 10-50)'
                )
        
        # 4. Validate simulated annealing parameters
        if self.sa_starting_temp <= 0:
            errors.append(
                f'Error: sa_starting_temp ({self.sa_starting_temp}) must be positive\n'
                f'  Suggestion: Set sa_starting_temp > 0 (typically 0.1-10.0)'
            )
        if self.sa_nsteps < 1:
            errors.append(
                f'Error: sa_nsteps ({self.sa_nsteps}) must be at least 1\n'
                f'  Suggestion: Set sa_nsteps >= 1 (typically 1000-10000)'
            )
        if self.ga_nsteps < 1:
            errors.append(
                f'Error: ga_nsteps ({self.ga_nsteps}) must be at least 1\n'
                f'  Suggestion: Set ga_nsteps >= 1 (typically 1000-20000)'
            )
        if self.sa_step_size <= 0:
            errors.append(
                f'Error: sa_step_size ({self.sa_step_size}) must be positive\n'
                f'  Suggestion: Set sa_step_size > 0 (typically 0.001-0.1)'
            )
        if self.ga_step_size <= 0:
            errors.append(
                f'Error: ga_step_size ({self.ga_step_size}) must be positive\n'
                f'  Suggestion: Set ga_step_size > 0 (typically 0.001-0.1)'
            )
        if self.nrestarts < 1:
            errors.append(
                f'Error: nrestarts ({self.nrestarts}) must be at least 1\n'
                f'  Suggestion: Set nrestarts >= 1 (typically 1-10)'
            )
        if self.ntotalruns < 1:
            errors.append(
                f'Error: ntotalruns ({self.ntotalruns}) must be at least 1\n'
                f'  Suggestion: Set ntotalruns >= 1 (typically 1-10)'
            )
        if self.tuning_ratio_target < 0 or self.tuning_ratio_target > 1:
            errors.append(
                f'Error: tuning_ratio_target ({self.tuning_ratio_target}) must be between 0 and 1\n'
                f'  Suggestion: Set 0 <= tuning_ratio_target <= 1 (typically 0.3-0.7)'
            )
        if self.c_tuning_initial <= 0:
            errors.append(
                f'Error: c_tuning_initial ({self.c_tuning_initial}) must be positive\n'
                f'  Suggestion: Set c_tuning_initial > 0 (typically 0.001-0.1)'
            )
        if self.hydrogen_mode_damping_factor < 0:
            errors.append(
                f'Error: hydrogen_mode_damping_factor ({self.hydrogen_mode_damping_factor}) must be non-negative\n'
                f'  Suggestion: Set hydrogen_mode_damping_factor >= 0 (typically 0.1-0.5)'
            )
        
        # 5. Validate sampling parameters
        if self.sampling_bool and self.boltzmann_temperature <= 0:
            errors.append(
                f'Error: boltzmann_temperature ({self.boltzmann_temperature}) must be positive when sampling is enabled\n'
                f'  Suggestion: Set boltzmann_temperature > 0 (typically 100-500 K)'
            )
        
        # 6. Validate excitation factor
        if self.excitation_factor <= 0:
            errors.append(
                f'Error: excitation_factor ({self.excitation_factor}) must be positive\n'
                f'  Suggestion: Set excitation_factor > 0 (typically 0.1-10.0)'
            )
        
        # 7. Validate noise parameters
        if self.noise_value < 0:
            errors.append(
                f'Error: noise_value ({self.noise_value}) must be non-negative\n'
                f'  Suggestion: Set noise_value >= 0 (typically 0.0-0.1)'
            )
        
        # 8. Validate molecule-specific parameters (if molecule section exists)
        if "molecule_params" in data and molecule in data["molecule_params"]:
            mol_data = data["molecule_params"][molecule]
            
            # Check natoms and nmodes
            if self.natoms < 1:
                errors.append(
                    f'Error: natoms ({self.natoms}) must be at least 1\n'
                    f'  Suggestion: Set natoms >= 1'
                )
            if self.nmodes < 0:
                errors.append(
                    f'Error: nmodes ({self.nmodes}) must be non-negative\n'
                    f'  Suggestion: For non-linear molecules, nmodes = 3*natoms - 6'
                )
            if self.nmodes > 3 * self.natoms:
                warnings.append(
                    f'Warning: nmodes ({self.nmodes}) is unusually large for natoms ({self.natoms})\n'
                    f'  Expected: nmodes = 3*natoms - 6 = {3*self.natoms - 6} for non-linear molecules'
                )
            
            # Validate mode ranges
            if len(self.hydrogen_mode_range) != 2:
                errors.append(
                    f'Error: hydrogen_mode_range must have exactly 2 elements, got {len(self.hydrogen_mode_range)}\n'
                    f'  Suggestion: Use format [start, end) (e.g., [28, 36])'
                )
            else:
                h_start, h_end = self.hydrogen_mode_range
                if h_start < 0 or h_start >= self.nmodes:
                    errors.append(
                        f'Error: hydrogen_mode_range[0] ({h_start}) must be in [0, nmodes) = [0, {self.nmodes})\n'
                        f'  Suggestion: Set hydrogen_mode_range[0] in valid range'
                    )
                if h_end < 0 or h_end > self.nmodes:
                    errors.append(
                        f'Error: hydrogen_mode_range[1] ({h_end}) must be in [0, nmodes] = [0, {self.nmodes}]\n'
                        f'  Suggestion: Set hydrogen_mode_range[1] in valid range'
                    )
                if h_start >= h_end:
                    errors.append(
                        f'Error: hydrogen_mode_range[0] ({h_start}) must be less than hydrogen_mode_range[1] ({h_end})\n'
                        f'  Suggestion: Set hydrogen_mode_range[0] < hydrogen_mode_range[1]'
                    )
            
            if len(self.sa_mode_range) != 2:
                errors.append(
                    f'Error: sa_mode_range must have exactly 2 elements, got {len(self.sa_mode_range)}\n'
                    f'  Suggestion: Use format [start, end) (e.g., [0, 36])'
                )
            else:
                sa_start, sa_end = self.sa_mode_range
                if sa_start < 0 or sa_start >= self.nmodes:
                    errors.append(
                        f'Error: sa_mode_range[0] ({sa_start}) must be in [0, nmodes) = [0, {self.nmodes})\n'
                        f'  Suggestion: Set sa_mode_range[0] in valid range'
                    )
                if sa_end < 0 or sa_end > self.nmodes:
                    errors.append(
                        f'Error: sa_mode_range[1] ({sa_end}) must be in [0, nmodes] = [0, {self.nmodes}]\n'
                        f'  Suggestion: Set sa_mode_range[1] in valid range'
                    )
                if sa_start >= sa_end:
                    errors.append(
                        f'Error: sa_mode_range[0] ({sa_start}) must be less than sa_mode_range[1] ({sa_end})\n'
                        f'  Suggestion: Set sa_mode_range[0] < sa_mode_range[1]'
                    )
            
            if len(self.ga_mode_range) != 2:
                errors.append(
                    f'Error: ga_mode_range must have exactly 2 elements, got {len(self.ga_mode_range)}\n'
                    f'  Suggestion: Use format [start, end) (e.g., [0, 36])'
                )
            else:
                ga_start, ga_end = self.ga_mode_range
                if ga_start < 0 or ga_start >= self.nmodes:
                    errors.append(
                        f'Error: ga_mode_range[0] ({ga_start}) must be in [0, nmodes) = [0, {self.nmodes})\n'
                        f'  Suggestion: Set ga_mode_range[0] in valid range'
                    )
                if ga_end < 0 or ga_end > self.nmodes:
                    errors.append(
                        f'Error: ga_mode_range[1] ({ga_end}) must be in [0, nmodes] = [0, {self.nmodes}]\n'
                        f'  Suggestion: Set ga_mode_range[1] in valid range'
                    )
                if ga_start >= ga_end:
                    errors.append(
                        f'Error: ga_mode_range[0] ({ga_start}) must be less than ga_mode_range[1] ({ga_end})\n'
                        f'  Suggestion: Set ga_mode_range[0] < ga_mode_range[1]'
                    )
            
            # Validate index arrays
            if len(self.bond_indices) != 2:
                errors.append(
                    f'Error: bond_indices must have exactly 2 elements, got {len(self.bond_indices)}\n'
                    f'  Suggestion: Use format [atom1_idx, atom2_idx]'
                )
            else:
                for idx in self.bond_indices:
                    if idx < 0 or idx >= self.natoms:
                        errors.append(
                            f'Error: bond_indices contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                            f'  Suggestion: Check bond_indices values are valid atom indices'
                        )
            
            # angle_indices/dihedral_indices are optional analysis targets; allow empty.
            if len(self.angle_indices) not in (0, 3):
                errors.append(
                    f'Error: angle_indices must have exactly 3 elements (or be empty), got {len(self.angle_indices)}\n'
                    f'  Suggestion: Use format [atom1_idx, atom2_idx, atom3_idx] or []'
                )
            elif len(self.angle_indices) == 3:
                for idx in self.angle_indices:
                    if idx < 0 or idx >= self.natoms:
                        errors.append(
                            f'Error: angle_indices contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                            f'  Suggestion: Check angle_indices values are valid atom indices'
                        )
            
            if len(self.dihedral_indices) not in (0, 4):
                errors.append(
                    f'Error: dihedral_indices must have exactly 4 elements (or be empty), got {len(self.dihedral_indices)}\n'
                    f'  Suggestion: Use format [atom1_idx, atom2_idx, atom3_idx, atom4_idx] or []'
                )
            elif len(self.dihedral_indices) == 4:
                for idx in self.dihedral_indices:
                    if idx < 0 or idx >= self.natoms:
                        errors.append(
                            f'Error: dihedral_indices contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                            f'  Suggestion: Check dihedral_indices values are valid atom indices'
                        )
            
            # Validate ignore arrays (check atom indices)
            for i, bond_pair in enumerate(self.bond_ignore_array):
                if len(bond_pair) != 2:
                    errors.append(
                        f'Error: bond_ignore_array[{i}] must have exactly 2 elements, got {len(bond_pair)}\n'
                        f'  Suggestion: Use format [atom1_idx, atom2_idx]'
                    )
                else:
                    for idx in bond_pair:
                        if idx < 0 or idx >= self.natoms:
                            errors.append(
                                f'Error: bond_ignore_array[{i}] contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                                f'  Suggestion: Check bond_ignore_array values are valid atom indices'
                            )
            
            for i, angle_triple in enumerate(self.angle_ignore_array):
                if len(angle_triple) != 3:
                    errors.append(
                        f'Error: angle_ignore_array[{i}] must have exactly 3 elements, got {len(angle_triple)}\n'
                        f'  Suggestion: Use format [atom1_idx, atom2_idx, atom3_idx]'
                    )
                else:
                    for idx in angle_triple:
                        if idx < 0 or idx >= self.natoms:
                            errors.append(
                                f'Error: angle_ignore_array[{i}] contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                                f'  Suggestion: Check angle_ignore_array values are valid atom indices'
                            )
            
            for i, torsion_quad in enumerate(self.torsion_ignore_array):
                if len(torsion_quad) != 4:
                    errors.append(
                        f'Error: torsion_ignore_array[{i}] must have exactly 4 elements, got {len(torsion_quad)}\n'
                        f'  Suggestion: Use format [atom1_idx, atom2_idx, atom3_idx, atom4_idx]'
                    )
                else:
                    for idx in torsion_quad:
                        if idx < 0 or idx >= self.natoms:
                            errors.append(
                                f'Error: torsion_ignore_array[{i}] contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                                f'  Suggestion: Check torsion_ignore_array values are valid atom indices'
                            )
            
            # Validate rmsd_indices
            for idx in self.rmsd_indices:
                if idx < 0 or idx >= self.natoms:
                    errors.append(
                        f'Error: rmsd_indices contains invalid atom index {idx} (must be in [0, natoms) = [0, {self.natoms}))\n'
                        f'  Suggestion: Check rmsd_indices values are valid atom indices'
                    )
        
        # 9. Check for logical inconsistencies
        if not self.bonds_bool and not self.angles_bool and not self.torsions_bool:
            warnings.append(
                'Warning: All geometric constraints are disabled (bonds_bool, angles_bool, torsions_bool all False)\n'
                '  Suggestion: Consider enabling at least one constraint type for better structure preservation'
            )
        
        if self.sa_nsteps == 0 and self.ga_nsteps == 0:
            errors.append(
                'Error: Both sa_nsteps and ga_nsteps are 0, no optimization will occur\n'
                '  Suggestion: Set at least one of sa_nsteps or ga_nsteps > 0'
            )
        
        # Print warnings first
        if warnings:
            print("\n" + "="*60)
            print("PARAMETER VALIDATION WARNINGS:")
            print("="*60)
            for warning in warnings:
                print(f"  ⚠ {warning}")
            print("="*60 + "\n")
        
        # Print errors and exit if any
        if errors:
            print("\n" + "="*60)
            print("PARAMETER VALIDATION ERRORS:")
            print("="*60)
            for i, error in enumerate(errors, 1):
                print(f"\nError {i}:")
                print(f"  {error}")
            print("\n" + "="*60)
            print("Please fix the errors above and try again.")
            print("="*60 + "\n")
            sys.exit(1)
