"""
Pytest configuration and fixtures
"""
import pytest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_xyz_file():
    """Create a temporary XYZ file for testing"""
    xyz_content = """2
Test molecule
H    0.0    0.0    0.0
O    0.0    0.0    0.96
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write(xyz_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def sample_xyz_data():
    """Sample XYZ data as numpy arrays"""
    atoms = np.array(['H', 'O'])
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    return atoms, xyz


@pytest.fixture
def sample_xyz_coords():
    """Sample XYZ coordinates for testing"""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])


@pytest.fixture
def sample_qvector():
    """Sample q-vector for scattering calculations"""
    return np.linspace(0.1, 10.0, 50)


@pytest.fixture
def sample_toml_file():
    """Create a temporary TOML file for testing"""
    toml_content = """mode = "test"

[run_params]
run_id = "test_run"
molecule = "test"
results_dir = "test_results"

[files]
forcefield_file = "forcefields/test.offxml"
start_xyz_file = "xyz/test.xyz"
start_sdf_file = "sdf/test.sdf"
reference_xyz_file = "xyz/test_ref.xyz"
target_file = "xyz/test_target.xyz"

[options]
run_pyscf_modes_bool = false
pyscf_basis = "6-31g"
verbose_bool = false
write_dat_file_bool = false

[sampling]
sampling_bool = false
boltzmann_temperature = 300.0

[scattering_params]
inelastic_bool = true
pcd_mode_bool = false
excitation_factor = 1.0

[scattering_params.q]
qmin = 0.1
qmax = 10.0
qlen = 50

[scattering_params.ewald]
ewald_mode_bool = false

[scattering_params.ewald.th]
tmin = 0.0
tmax = 1.0
tlen = 21

[scattering_params.ewald.ph]
pmin = 0.0
pmax = 2.0
plen = 21

[scattering_params.noise]
noise_value = 0.0
noise_data_file = "noise/noise.dat"

[simulated_annealing_params]
sa_starting_temp = 1.0
sa_nsteps = 1000
greedy_algorithm_bool = false
ga_nsteps = 5000
sa_step_size = 0.01
ga_step_size = 0.01
nrestarts = 1
ntotalruns = 1
bonds_bool = true
angles_bool = true
torsions_bool = true
tuning_ratio_target = 1.0
c_tuning_initial = 1.0
non_h_modes_only_bool = false
hydrogen_mode_damping_factor = 0.2
hf_energy_bool = false

[molecule_params.test]
natoms = 2
nmodes = 6
hydrogen_mode_range = [4, 6]
sa_mode_range = [0, 6]
ga_mode_range = [0, 6]
bond_ignore_array = []
angle_ignore_array = []
torsion_ignore_array = []
rmsd_indices = [0, 1]
bond_indices = [0, 1]
angle_indices = []
dihedral_indices = []
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
