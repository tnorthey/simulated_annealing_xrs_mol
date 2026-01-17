"""Performance tests for simulated annealing"""
import pytest
import numpy as np
from timeit import default_timer

import modules.sa as sa_module


@pytest.fixture
def small_test_data():
    """Create minimal test data for performance testing"""
    natoms = 3
    nmodes = 3
    qlen = 20
    tlen = 5
    plen = 5
    
    # Starting coordinates
    starting_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0]
    ], dtype=np.float64)
    
    # Displacements (modes x atoms x 3)
    displacements = np.random.randn(nmodes, natoms, 3) * 0.1
    
    # Mode indices
    mode_indices = np.array([0, 1, 2], dtype=np.int64)
    
    # Step sizes
    step_size_array = np.ones(nmodes, dtype=np.float64) * 0.01
    
    # Q vector
    qvector = np.linspace(0.1, 5.0, qlen)
    
    # Theta and phi (for Ewald mode)
    th = np.linspace(0.0, np.pi, tlen)
    ph = np.linspace(0.0, 2 * np.pi, plen)
    
    # Target and reference IAM
    target_function = np.random.rand(qlen) * 100
    reference_iam = np.random.rand(qlen) * 100
    
    # Compton and atomic arrays
    compton = np.random.rand(qlen) * 10
    atomic_total = np.random.rand(qlen) * 50
    
    # Atomic factors and pre-molecular products (pairs x qlen)
    atomic_factor_array = np.random.rand(natoms, qlen) * 5
    npairs = natoms * (natoms - 1) // 2
    pre_molecular = np.zeros((npairs, qlen), dtype=np.float64)
    k = 0
    for ii in range(natoms - 1):
        for jj in range(ii + 1, natoms):
            pre_molecular[k, :] = atomic_factor_array[ii, :] * atomic_factor_array[jj, :]
            k += 1
    
    # Bond/angle/torsion parameters (minimal)
    bond_param_array = np.array([
        [0, 1, 1.0, 1.0],  # bond 0-1: r0=1.0, k=1.0
        [1, 2, 1.0, 1.0],  # bond 1-2
    ], dtype=np.float64)
    
    angle_param_array = np.array([
        [0, 1, 2, np.pi/3, 1.0],  # angle 0-1-2: theta0=60°, k=1.0
    ], dtype=np.float64)
    
    torsion_param_array = np.array([], dtype=np.float64).reshape(0, 6)
    
    return {
        'starting_xyz': starting_xyz,
        'displacements': displacements,
        'mode_indices': mode_indices,
        'step_size_array': step_size_array,
        'target_function': target_function,
        'reference_iam': reference_iam,
        'qvector': qvector,
        'th': th,
        'ph': ph,
        'compton': compton,
        'atomic_total': atomic_total,
        'pre_molecular': pre_molecular,
        'atomic_factor_array': atomic_factor_array,
        'bond_param_array': bond_param_array,
        'angle_param_array': angle_param_array,
        'torsion_param_array': torsion_param_array,
    }


class TestPerformance:
    """Performance tests for simulated annealing"""
    
    @pytest.mark.slow
    def test_single_run_performance(self, small_test_data):
        """Test performance of a single annealing run"""
        data = small_test_data
        sa = sa_module.Annealing()
        nsteps = 1000
        
        # Initialize predicted_start as array (not int 0)
        predicted_start = np.zeros(len(data['qvector']))
        
        # Warm up (first call compiles njit function)
        _ = sa.simulated_annealing_modes_ho(
            data['starting_xyz'],
            data['displacements'],
            data['mode_indices'],
            data['target_function'],
            data['reference_iam'],
            data['qvector'],
            data['th'],
            data['ph'],
            data['compton'],
            data['atomic_total'],
            data['pre_molecular'],
            data['atomic_factor_array'],
            data['step_size_array'],
            data['bond_param_array'],
            data['angle_param_array'],
            data['torsion_param_array'],
            starting_temp=0.2,
            nsteps=100,  # Small warmup
            inelastic=True,
            pcd_mode=False,
            ewald_mode=False,
            use_pre_molecular=True,
            bonds_bool=True,
            angles_bool=True,
            torsions_bool=False,
            predicted_start=predicted_start,
        )
        
        # Actual timing
        start = default_timer()
        result = sa.simulated_annealing_modes_ho(
            data['starting_xyz'],
            data['displacements'],
            data['mode_indices'],
            data['target_function'],
            data['reference_iam'],
            data['qvector'],
            data['th'],
            data['ph'],
            data['compton'],
            data['atomic_total'],
            data['pre_molecular'],
            data['atomic_factor_array'],
            data['step_size_array'],
            data['bond_param_array'],
            data['angle_param_array'],
            data['torsion_param_array'],
            starting_temp=0.2,
            nsteps=nsteps,
            inelastic=True,
            pcd_mode=False,
            ewald_mode=False,
            use_pre_molecular=True,
            bonds_bool=True,
            angles_bool=True,
            torsions_bool=False,
            predicted_start=predicted_start,
        )
        elapsed = default_timer() - start
        
        print(f"\nSingle run ({nsteps} steps): {elapsed:.4f} s")
        print(f"Time per step: {elapsed/nsteps*1000:.4f} ms")
        
        # Assert reasonable performance (< 5 seconds for 1000 steps on small data)
        assert elapsed < 5.0, f"Single run took {elapsed:.4f} s, expected < 5.0 s"
        
        # Verify result structure
        assert len(result) == 5
        f_best, f_xray_best, predicted_best, xyz_best, c_tuning_adjusted = result
        assert isinstance(f_best, float)
        assert isinstance(f_xray_best, float)
        assert xyz_best.shape == data['starting_xyz'].shape
    
    @pytest.mark.slow
    def test_multiple_restarts_performance(self, small_test_data):
        """Test performance of multiple restarts (simulating wrap.py restart loop)"""
        data = small_test_data
        sa = sa_module.Annealing()
        nrestarts = 3
        sa_nsteps = 500
        ga_nsteps = 500
        
        # Initialize best values (simulating wrap.py)
        xyz_best = data['starting_xyz'].copy()
        f_best, f_xray_best = 1e10, 1e10
        predicted_best = np.zeros(len(data['qvector']))
        c_tuning = 1.0
        
        # Initialize predicted_start
        predicted_start = np.zeros(len(data['qvector']))
        
        # Warm up
        _ = sa.simulated_annealing_modes_ho(
            data['starting_xyz'],
            data['displacements'],
            data['mode_indices'],
            data['target_function'],
            data['reference_iam'],
            data['qvector'],
            data['th'],
            data['ph'],
            data['compton'],
            data['atomic_total'],
            data['pre_molecular'],
            data['atomic_factor_array'],
            data['step_size_array'],
            data['bond_param_array'],
            data['angle_param_array'],
            data['torsion_param_array'],
            starting_temp=0.2,
            nsteps=100,
            inelastic=True,
            pcd_mode=False,
            ewald_mode=False,
            use_pre_molecular=True,
            bonds_bool=True,
            angles_bool=True,
            torsions_bool=False,
            predicted_start=predicted_start,
        )
        
        # Actual timing - simulate restart loop
        start = default_timer()
        for i in range(nrestarts + 1):
            xyz_start = xyz_best.copy()
            f_start = f_best
            f_xray_start = f_xray_best
            predicted_start = predicted_best.copy()
            
            if i < nrestarts:
                nsteps = sa_nsteps
                starting_temp = 0.2
                mode_indices = data['mode_indices']
            else:
                # Skip greedy algorithm for this test
                continue
            
            (
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
                c_tuning_adjusted,
            ) = sa.simulated_annealing_modes_ho(
                xyz_start,
                data['displacements'],
                mode_indices,
                data['target_function'],
                data['reference_iam'],
                data['qvector'],
                data['th'],
                data['ph'],
                data['compton'],
                data['atomic_total'],
                data['pre_molecular'],
            data['atomic_factor_array'],
                data['step_size_array'],
                data['bond_param_array'],
                data['angle_param_array'],
                data['torsion_param_array'],
                starting_temp=starting_temp,
                nsteps=nsteps,
                inelastic=True,
                pcd_mode=False,
                ewald_mode=False,
            use_pre_molecular=True,
                bonds_bool=True,
                angles_bool=True,
                torsions_bool=False,
                f_start=f_start,
                f_xray_start=f_xray_start,
                predicted_start=predicted_start,
                c_tuning_initial=c_tuning,
            )
            c_tuning = c_tuning_adjusted
        
        elapsed = default_timer() - start
        total_steps = nrestarts * sa_nsteps
        
        print(f"\nMultiple restarts ({nrestarts} restarts, {sa_nsteps} steps each): {elapsed:.4f} s")
        print(f"Total steps: {total_steps}")
        print(f"Time per step: {elapsed/total_steps*1000:.4f} ms")
        print(f"Time per restart: {elapsed/nrestarts:.4f} s")
        
        # Verify result structure
        assert isinstance(f_best, float)
        assert isinstance(f_xray_best, float)
        assert xyz_best.shape == data['starting_xyz'].shape
    
    @pytest.mark.slow
    def test_scaling_with_restarts(self, small_test_data):
        """Test how performance scales with number of restarts"""
        data = small_test_data
        sa = sa_module.Annealing()
        sa_nsteps = 200
        ga_nsteps = 200
        
        restart_counts = [1, 2, 4]
        times = []
        
        for nrestarts in restart_counts:
            # Initialize
            xyz_best = data['starting_xyz'].copy()
            f_best, f_xray_best = 1e10, 1e10
            predicted_best = np.zeros(len(data['qvector']))
            c_tuning = 1.0
            
            # Initialize predicted_start
            predicted_start = np.zeros(len(data['qvector']))
            
            # Warm up on first iteration
            if nrestarts == restart_counts[0]:
                _ = sa.simulated_annealing_modes_ho(
                    data['starting_xyz'],
                    data['displacements'],
                    data['mode_indices'],
                    data['target_function'],
                    data['reference_iam'],
                    data['qvector'],
                    data['th'],
                    data['ph'],
                    data['compton'],
                    data['atomic_total'],
                    data['pre_molecular'],
                    data['atomic_factor_array'],
                    data['step_size_array'],
                    data['bond_param_array'],
                    data['angle_param_array'],
                    data['torsion_param_array'],
                    starting_temp=0.2,
                    nsteps=100,
                    inelastic=True,
                    pcd_mode=False,
                    ewald_mode=False,
                    use_pre_molecular=True,
                    bonds_bool=True,
                    angles_bool=True,
                    torsions_bool=False,
                    predicted_start=predicted_start,
                )
            
            # Time it
            start = default_timer()
            for i in range(nrestarts):
                xyz_start = xyz_best.copy()
                f_start = f_best
                f_xray_start = f_xray_best
                predicted_start = predicted_best.copy()
                
                (
                    f_best,
                    f_xray_best,
                    predicted_best,
                    xyz_best,
                    c_tuning_adjusted,
                ) = sa.simulated_annealing_modes_ho(
                    xyz_start,
                    data['displacements'],
                    data['mode_indices'],
                    data['target_function'],
                    data['reference_iam'],
                    data['qvector'],
                    data['th'],
                    data['ph'],
                    data['compton'],
                    data['atomic_total'],
                    data['pre_molecular'],
                    data['atomic_factor_array'],
                    data['step_size_array'],
                    data['bond_param_array'],
                    data['angle_param_array'],
                    data['torsion_param_array'],
                    starting_temp=0.2,
                    nsteps=sa_nsteps,
                    inelastic=True,
                    pcd_mode=False,
                    ewald_mode=False,
                    use_pre_molecular=True,
                    bonds_bool=True,
                    angles_bool=True,
                    torsions_bool=False,
                    f_start=f_start,
                    f_xray_start=f_xray_start,
                    predicted_start=predicted_start,
                    c_tuning_initial=c_tuning,
                )
                c_tuning = c_tuning_adjusted
            elapsed = default_timer() - start
            times.append(elapsed)
            
            print(f"  {nrestarts} restarts: {elapsed:.4f} s ({elapsed/nrestarts:.4f} s per restart)")
        
        # Check that scaling is roughly linear (with some tolerance for overhead)
        if len(times) >= 2:
            ratio_2_1 = times[1] / times[0] if times[0] > 0 else 0
            ratio_4_2 = times[2] / times[1] if times[1] > 0 else 0
            
            print(f"\nScaling ratios: 2/1={ratio_2_1:.2f}, 4/2={ratio_4_2:.2f}")
            # Should be roughly linear (ratios around 2.0, with some tolerance)
            assert 1.5 < ratio_2_1 < 2.5, f"Scaling not linear: ratio={ratio_2_1:.2f}"
            assert 1.5 < ratio_4_2 < 2.5, f"Scaling not linear: ratio={ratio_4_2:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
