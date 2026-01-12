"""
Tests for modules/read_input.py
"""
import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.read_input import Input_to_params


class TestInputToParams:
    """Test Input_to_params class"""
    
    def test_read_valid_toml(self, sample_toml_file):
        """Test reading a valid TOML file"""
        p = Input_to_params(sample_toml_file)
        
        assert p.mode == "test"
        assert p.run_id == "test_run"
        assert p.molecule == "test"
        assert p.inelastic is True
        assert p.qmin == 0.1
        assert p.qmax == 10.0
        assert p.qlen == 50
        assert p.sa_starting_temp == 1.0
        assert p.sa_nsteps == 1000
    
    def test_read_with_overrides(self, sample_toml_file):
        """Test reading with parameter overrides"""
        overrides = {
            "run_params.run_id": "override_run",
            "simulated_annealing_params.sa_nsteps": 5000
        }
        
        p = Input_to_params(sample_toml_file, overrides=overrides)
        
        assert p.run_id == "override_run"
        assert p.sa_nsteps == 5000
        # Other parameters should remain unchanged
        assert p.mode == "test"
        assert p.molecule == "test"
    
    def test_mode_validation(self):
        """Test mode validation"""
        # Create TOML with invalid mode
        toml_content = '''mode = "invalid"
[run_params]
run_id = "test"
molecule = "test"
results_dir = "test"

[files]
forcefield_file = "test.offxml"
start_xyz_file = "test.xyz"
start_sdf_file = "test.sdf"
reference_xyz_file = "test_ref.xyz"
target_file = "test_target.xyz"

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
noise_data_file = "noise.dat"

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
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_file = f.name
        
        try:
            with pytest.raises(SystemExit):
                Input_to_params(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_mode_case_insensitive(self):
        """Test that mode is case insensitive"""
        toml_content = '''mode = "TEST"
[run_params]
run_id = "test"
molecule = "test"
results_dir = "test"

[files]
forcefield_file = "test.offxml"
start_xyz_file = "test.xyz"
start_sdf_file = "test.sdf"
reference_xyz_file = "test_ref.xyz"
target_file = "test_target.xyz"

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
noise_data_file = "noise.dat"

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
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_file = f.name
        
        try:
            p = Input_to_params(temp_file)
            assert p.mode.lower() == "test"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_qvector_generation(self, sample_toml_file):
        """Test qvector is generated correctly"""
        p = Input_to_params(sample_toml_file)
        
        assert hasattr(p, 'qvector')
        assert len(p.qvector) == p.qlen
        assert abs(p.qvector[0] - p.qmin) < 1e-10
        assert abs(p.qvector[-1] - p.qmax) < 1e-10
    
    def test_molecule_params_loading(self):
        """Test molecule-specific parameters are loaded"""
        toml_content = '''mode = "test"
[run_params]
run_id = "test"
molecule = "chd"
results_dir = "test"

[files]
forcefield_file = "test.offxml"
start_xyz_file = "test.xyz"
start_sdf_file = "test.sdf"
reference_xyz_file = "test_ref.xyz"
target_file = "test_target.xyz"

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
noise_data_file = "noise.dat"

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

[molecule_params.chd]
natoms = 14
nmodes = 36
hydrogen_mode_range = [28, 36]
sa_mode_range = [0, 36]
ga_mode_range = [0, 36]
bond_ignore_array = [[0, 5]]
angle_ignore_array = []
torsion_ignore_array = []
rmsd_indices = [0, 1, 2, 3, 4, 5]
bond_indices = [0, 5]
angle_indices = [0, 3, 5]
dihedral_indices = [0, 1, 4, 5]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_file = f.name
        
        try:
            p = Input_to_params(temp_file)
            assert hasattr(p, 'natoms')
            assert p.natoms == 14
            assert p.nmodes == 36
            assert len(p.hydrogen_mode_range) == 2
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
