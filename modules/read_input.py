# import math
import numpy as np
import json
import sys
import pprint


######
class Input_to_params:
    """read parameters for simulated annealing"""

    def __init__(self, input_json_file):
        """initialise with hard-coded params"""

        ###################################
        # Load input JSON
        with open(input_json_file, "r") as f:
            data = json.load(f)
        ### Parameters
        # mode
        self.mode = str(data["mode"])
        mode = self.mode.lower()  # lower case mode string
        ## handle the case when mode does not equal "dat" or "xyz"
        try:
            if not (mode == "dat" or mode == "xyz"):
                raise ValueError(
                    'mode value must equal "dat" or "xyz"! (case insensitive). Exiting...'
                )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  # exit program with error message.
        # run params
        self.run_id = str(data["run_params"]["run_id"])
        self.molecule = str(data["run_params"]["molecule"])
        self.results_dir = str(data["run_params"]["results_dir"])
        # options
        self.run_pyscf_modes_bool = bool(data["options"]["run_pyscf_modes_bool"])
        self.pyscf_basis = str(data["options"]["pyscf_basis"])
        self.verbose_bool = bool(data["options"]["verbose_bool"])
        self.write_dat_file_bool = bool(data["options"]["write_dat_file_bool"])
        # sampling options
        self.sampling_bool = bool(data["sampling"]["sampling_bool"])
        self.boltzmann_temperature = bool(data["sampling"]["boltzmann_temperature"])
        # file params
        self.forcefield_file = str(data["files"]["forcefield_file"])
        self.start_xyz_file = str(data["files"]["start_xyz_file"])
        self.start_sdf_file = str(data["files"]["start_sdf_file"])
        self.reference_xyz_file = str(data["files"]["reference_xyz_file"])
        self.target_file = str(data["files"]["target_file"])
        # scattering_params params
        self.inelastic = bool(data["scattering_params"]["inelastic_bool"])
        self.pcd_mode = bool(data["scattering_params"]["pcd_mode_bool"])
        self.excitation_factor = float(data["scattering_params"]["excitation_factor"])
        self.ewald_mode = bool(data["scattering_params"]["ewald_mode_bool"])
        # radial q params
        self.qmin = float(data["scattering_params"]["q"]["qmin"])
        self.qmax = float(data["scattering_params"]["q"]["qmax"])
        self.qlen = int(data["scattering_params"]["q"]["qlen"])
        # theta params
        self.tmin = float(data["scattering_params"]["th"]["tmin"])
        self.tmax = float(data["scattering_params"]["th"]["tmax"])
        self.tlen = int(data["scattering_params"]["th"]["tlen"])
        # phi params
        self.pmin = float(data["scattering_params"]["ph"]["pmin"])
        self.pmax = float(data["scattering_params"]["ph"]["pmax"])
        self.plen = int(data["scattering_params"]["ph"]["plen"])
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
        self.hf_energy = bool(
            data["simulated_annealing_params"]["hf_energy_bool"]
        )  # run PySCF HF energy

        # molecule params
        molecule = self.molecule
        self.natoms = int(data["molecule_params"][molecule]["natoms"])
        self.nmodes = int(data["molecule_params"][molecule]["nmodes"])
        self.nmfile = str(data["molecule_params"][molecule]["nmfile"])
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

        ### print out all attributes
        print("")
        print("##################################################")
        print(" ___ ___ _______ _______ _______ _______ _____   ")
        print("|   |   |     __|   _   |   |   |       |     |_ ")
        print("|-     -|__     |       |       |   -   |       |")
        print("|___|___|_______|___|___|__|_|__|_______|_______|")
        print("")
        print("##################################################")
        print("### Initialised with the following parameters: ###")
        print("##################################################")
        if self.verbose_bool:
            pprint.pprint(vars(self))
        print("##################################################")

        ###################################
