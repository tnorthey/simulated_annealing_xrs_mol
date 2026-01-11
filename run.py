"""
Run simulated annealing
"""

# run example: python3 run.py
# Override defaults: python3 run.py --mode xyz --molecule chd --run-id test_run

import argparse
from timeit import default_timer

start = default_timer()

# my modules
import modules.mol as mol
import modules.wrap as wrap
import modules.read_input as read_input

def create_parser():
    """Create argparse parser with all parameters"""
    parser = argparse.ArgumentParser(
        description='Run simulated annealing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', type=str, default='input.toml',
                       help='Path to TOML config file')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['dat', 'xyz'],
                       help='Mode: "dat" or "xyz"')
    
    # Run params
    run_group = parser.add_argument_group('run_params', 'Run parameters')
    run_group.add_argument('--run-id', type=str, dest='run_params.run_id',
                          help='Run ID')
    run_group.add_argument('--molecule', type=str, dest='run_params.molecule',
                          help='Molecule name')
    run_group.add_argument('--results-dir', type=str, dest='run_params.results_dir',
                          help='Results directory')
    
    # Files
    files_group = parser.add_argument_group('files', 'File paths')
    files_group.add_argument('--forcefield-file', type=str, dest='files.forcefield_file',
                            help='Force field file')
    files_group.add_argument('--start-xyz-file', type=str, dest='files.start_xyz_file',
                            help='Start XYZ file')
    files_group.add_argument('--start-sdf-file', type=str, dest='files.start_sdf_file',
                            help='Start SDF file')
    files_group.add_argument('--reference-xyz-file', type=str, dest='files.reference_xyz_file',
                            help='Reference XYZ file')
    files_group.add_argument('--target-file', type=str, dest='files.target_file',
                            help='Target file')
    
    # Options
    options_group = parser.add_argument_group('options', 'Options')
    options_group.add_argument('--run-pyscf-modes', action='store_true',
                              dest='options.run_pyscf_modes_bool',
                              help='Run normal modes calculation with PySCF')
    options_group.add_argument('--pyscf-basis', type=str, dest='options.pyscf_basis',
                              help='PySCF basis set')
    options_group.add_argument('--verbose', action='store_true',
                              dest='options.verbose_bool', help='Verbose output')
    options_group.add_argument('--write-dat-file', action='store_true',
                              dest='options.write_dat_file_bool', help='Write DAT file')
    
    # Sampling
    sampling_group = parser.add_argument_group('sampling', 'Sampling parameters')
    sampling_group.add_argument('--sampling', action='store_true',
                               dest='sampling.sampling_bool', help='Enable sampling')
    sampling_group.add_argument('--boltzmann-temperature', type=float,
                               dest='sampling.boltzmann_temperature',
                               help='Boltzmann temperature')
    
    # Scattering params
    scatter_group = parser.add_argument_group('scattering_params', 'Scattering parameters')
    scatter_group.add_argument('--inelastic', action='store_true',
                              dest='scattering_params.inelastic_bool',
                              help='Inelastic scattering')
    scatter_group.add_argument('--pcd-mode', action='store_true',
                              dest='scattering_params.pcd_mode_bool', help='PCD mode')
    scatter_group.add_argument('--excitation-factor', type=float,
                              dest='scattering_params.excitation_factor',
                              help='Excitation factor')
    scatter_group.add_argument('--ewald-mode', action='store_true',
                              dest='scattering_params.ewald_mode_bool', help='Ewald mode')
    
    # Q params
    scatter_group.add_argument('--qmin', type=float, dest='scattering_params.q.qmin',
                              help='Q minimum')
    scatter_group.add_argument('--qmax', type=float, dest='scattering_params.q.qmax',
                              help='Q maximum')
    scatter_group.add_argument('--qlen', type=int, dest='scattering_params.q.qlen',
                              help='Q length')
    
    # Theta params
    scatter_group.add_argument('--tmin', type=float, dest='scattering_params.th.tmin',
                              help='Theta minimum')
    scatter_group.add_argument('--tmax', type=float, dest='scattering_params.th.tmax',
                              help='Theta maximum')
    scatter_group.add_argument('--tlen', type=int, dest='scattering_params.th.tlen',
                              help='Theta length')
    
    # Phi params
    scatter_group.add_argument('--pmin', type=float, dest='scattering_params.ph.pmin',
                              help='Phi minimum')
    scatter_group.add_argument('--pmax', type=float, dest='scattering_params.ph.pmax',
                              help='Phi maximum')
    scatter_group.add_argument('--plen', type=int, dest='scattering_params.ph.plen',
                              help='Phi length')
    
    # Noise params
    scatter_group.add_argument('--noise-value', type=float,
                              dest='scattering_params.noise.noise_value',
                              help='Noise value')
    scatter_group.add_argument('--noise-data-file', type=str,
                              dest='scattering_params.noise.noise_data_file',
                              help='Noise data file')
    
    # Simulated annealing params
    sa_group = parser.add_argument_group('simulated_annealing_params',
                                        'Simulated annealing parameters')
    sa_group.add_argument('--sa-starting-temp', type=float,
                         dest='simulated_annealing_params.sa_starting_temp',
                         help='SA starting temperature')
    sa_group.add_argument('--sa-nsteps', type=int,
                         dest='simulated_annealing_params.sa_nsteps',
                         help='SA number of steps')
    sa_group.add_argument('--greedy-algorithm', action='store_true',
                         dest='simulated_annealing_params.greedy_algorithm_bool',
                         help='Use greedy algorithm')
    sa_group.add_argument('--ga-nsteps', type=int,
                         dest='simulated_annealing_params.ga_nsteps',
                         help='GA number of steps')
    sa_group.add_argument('--sa-step-size', type=float,
                         dest='simulated_annealing_params.sa_step_size',
                         help='SA step size')
    sa_group.add_argument('--ga-step-size', type=float,
                         dest='simulated_annealing_params.ga_step_size',
                         help='GA step size')
    sa_group.add_argument('--nrestarts', type=int,
                         dest='simulated_annealing_params.nrestarts',
                         help='Number of restarts')
    sa_group.add_argument('--ntotalruns', type=int,
                         dest='simulated_annealing_params.ntotalruns',
                         help='Total number of runs')
    sa_group.add_argument('--bonds', action='store_true',
                         dest='simulated_annealing_params.bonds_bool', help='Use bonds')
    sa_group.add_argument('--angles', action='store_true',
                         dest='simulated_annealing_params.angles_bool', help='Use angles')
    sa_group.add_argument('--torsions', action='store_true',
                         dest='simulated_annealing_params.torsions_bool',
                         help='Use torsions')
    sa_group.add_argument('--tuning-ratio-target', type=float,
                         dest='simulated_annealing_params.tuning_ratio_target',
                         help='Tuning ratio target')
    sa_group.add_argument('--c-tuning-initial', type=float,
                         dest='simulated_annealing_params.c_tuning_initial',
                         help='Initial C tuning')
    sa_group.add_argument('--non-h-modes-only', action='store_true',
                         dest='simulated_annealing_params.non_h_modes_only_bool',
                         help='Only include non-hydrogen modes')
    sa_group.add_argument('--hf-energy', action='store_true',
                         dest='simulated_annealing_params.hf_energy_bool',
                         help='Run PySCF HF energy')
    
    return parser

# Parse arguments
parser = create_parser()
args = parser.parse_args()

# Create dict of overrides (only non-None values, and True for store_true actions)
overrides = {}
for action in parser._actions:
    if hasattr(action, 'dest') and action.dest != 'config':
        value = getattr(args, action.dest, None)
        # For store_true actions, only override if flag was provided (value is True)
        # For other actions, override if value is not None
        if value is not None:
            # Check if this is a store_true action by checking the action type
            is_store_true = type(action).__name__ == '_StoreTrueAction'
            if is_store_true:
                # Only override if flag was provided (value is True)
                if value is True:
                    overrides[action.dest] = value
            else:
                # For other actions, override if value is not None
                overrides[action.dest] = value

# create class objects
m = mol.Xyz()
w = wrap.Wrapper()
p = read_input.Input_to_params(args.config, overrides=overrides)

# Call the params function and add them to p object
p = w.run_xyz_openff_mm_params(p, p.start_xyz_file)

# Call the run function
w.run(p)

print("Total time: %3.2f s" % float(default_timer() - start))
