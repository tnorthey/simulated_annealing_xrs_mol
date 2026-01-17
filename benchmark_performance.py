#!/usr/bin/env python3
"""
Performance benchmark script for simulated annealing.

This script benchmarks the performance of simulated annealing runs,
including single runs and multiple restarts (simulating the restart loop in wrap.py).

Usage:
    python3 benchmark_performance.py [--nrestarts N] [--nsteps N] [--iterations N]
"""

import argparse
import numpy as np
from timeit import default_timer
import sys

# Import the annealing module
import modules.sa as sa_module


def create_test_data(natoms=3, nmodes=3, qlen=20):
    """Create test data for benchmarking"""
    tlen = 5
    plen = 5
    
    # Starting coordinates
    starting_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0]
    ] * (natoms // 3 + 1), dtype=np.float64)[:natoms]
    
    # Displacements (modes x atoms x 3)
    displacements = np.random.randn(nmodes, natoms, 3) * 0.1
    
    # Mode indices
    mode_indices = np.arange(min(nmodes, len(displacements)), dtype=np.int64)
    
    # Step sizes
    step_size_array = np.ones(len(mode_indices), dtype=np.float64) * 0.01
    
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
    if natoms >= 2:
        bonds = [[i, i+1, 1.0, 1.0] for i in range(natoms-1)]
        bond_param_array = np.array(bonds, dtype=np.float64)
    else:
        bond_param_array = np.array([], dtype=np.float64).reshape(0, 4)
    
    angle_param_array = np.array([], dtype=np.float64).reshape(0, 5)
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


def benchmark_single_run(data, nsteps=1000, warmup=True, use_pre_molecular=False):
    """Benchmark a single annealing run"""
    sa = sa_module.Annealing()
    
    # Initialize predicted_start as array
    predicted_start = np.zeros(len(data['qvector']))
    
    if warmup:
        # Warm up (compilation)
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
            use_pre_molecular=use_pre_molecular,
            bonds_bool=True,
            angles_bool=False,
            torsions_bool=False,
            predicted_start=predicted_start,
        )
    
    # Actual benchmark
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
        use_pre_molecular=use_pre_molecular,
        bonds_bool=True,
        angles_bool=False,
        torsions_bool=False,
        predicted_start=predicted_start,
    )
    elapsed = default_timer() - start
    
    return elapsed, result


def benchmark_restart_loop(
    data, nrestarts=3, sa_nsteps=500, ga_nsteps=500, warmup=True, use_pre_molecular=False
):
    """Benchmark restart loop (simulating wrap.py behavior)"""
    sa = sa_module.Annealing()
    
    # Initialize best values (simulating wrap.py)
    xyz_best = data['starting_xyz'].copy()
    f_best, f_xray_best = 1e10, 1e10
    predicted_best = np.zeros(len(data['qvector']))
    c_tuning = 1.0
    
    # Initialize predicted_start as array
    predicted_start = np.zeros(len(data['qvector']))
    
    if warmup:
        # Warm up (compilation)
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
            use_pre_molecular=use_pre_molecular,
            bonds_bool=True,
            angles_bool=False,
            torsions_bool=False,
            predicted_start=predicted_start,
        )
    
    # Actual benchmark - simulate restart loop
    start_time = default_timer()
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
            # Skip greedy algorithm for benchmark
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
            use_pre_molecular=use_pre_molecular,
            bonds_bool=True,
            angles_bool=False,
            torsions_bool=False,
            f_start=f_start,
            f_xray_start=f_xray_start,
            predicted_start=predicted_start,
            c_tuning_initial=c_tuning,
        )
        c_tuning = c_tuning_adjusted
    
    elapsed = default_timer() - start_time
    
    return elapsed, (f_best, f_xray_best, predicted_best, xyz_best, c_tuning_adjusted)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark performance of simulated annealing"
    )
    parser.add_argument(
        '--nrestarts', type=int, default=3,
        help='Number of restarts for restart loop benchmark (default: 3)'
    )
    parser.add_argument(
        '--nsteps', type=int, default=500,
        help='Number of steps per restart (default: 500)'
    )
    parser.add_argument(
        '--iterations', type=int, default=5,
        help='Number of iterations for averaging (default: 5)'
    )
    parser.add_argument(
        '--natoms', type=int, default=3,
        help='Number of atoms in test molecule (default: 3)'
    )
    parser.add_argument(
        '--qlen', type=int, default=20,
        help='Length of q vector (default: 20)'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--pre-molecular', action='store_true',
        help='Use pre_molecular pair products (legacy behavior)'
    )
    mode_group.add_argument(
        '--on-the-fly', action='store_true',
        help='Compute f_i f_j inside the annealing loop (default)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducible benchmarks (default: None)'
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    print("=" * 70)
    print("Performance Benchmark: Simulated Annealing")
    print("=" * 70)
    use_pre_molecular = args.pre_molecular
    print(f"\nConfiguration:")
    print(f"  Atoms: {args.natoms}")
    print(f"  Q vector length: {args.qlen}")
    print(f"  Restarts: {args.nrestarts}")
    print(f"  Steps per restart: {args.nsteps}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Mode: {'pre_molecular' if use_pre_molecular else 'on-the-fly'}")
    print()
    
    # Create test data
    print("Creating test data...")
    data = create_test_data(natoms=args.natoms, qlen=args.qlen)
    
    # Benchmark single run
    print("\n" + "=" * 70)
    print("Benchmark 1: Single Annealing Run")
    print("=" * 70)
    single_times = []
    for i in range(args.iterations):
        elapsed, _ = benchmark_single_run(
            data,
            nsteps=args.nsteps,
            warmup=(i == 0),
            use_pre_molecular=use_pre_molecular,
        )
        single_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} s ({elapsed/args.nsteps*1000:.4f} ms/step)")
    
    avg_single = np.mean(single_times)
    std_single = np.std(single_times)
    print(f"\n  Average: {avg_single:.4f} s ± {std_single:.4f} s")
    print(f"  Average per step: {avg_single/args.nsteps*1000:.4f} ms")
    
    # Benchmark restart loop
    print("\n" + "=" * 70)
    print("Benchmark 2: Restart Loop (simulating wrap.py)")
    print("=" * 70)
    restart_times = []
    for i in range(args.iterations):
        elapsed, _ = benchmark_restart_loop(
            data,
            nrestarts=args.nrestarts,
            sa_nsteps=args.nsteps,
            ga_nsteps=args.nsteps,
            warmup=(i == 0),
            use_pre_molecular=use_pre_molecular,
        )
        restart_times.append(elapsed)
        total_steps = args.nrestarts * args.nsteps
        print(f"  Run {i+1}: {elapsed:.4f} s ({elapsed/total_steps*1000:.4f} ms/step, {elapsed/args.nrestarts:.4f} s/restart)")
    
    avg_restart = np.mean(restart_times)
    std_restart = np.std(restart_times)
    total_steps = args.nrestarts * args.nsteps
    print(f"\n  Average: {avg_restart:.4f} s ± {std_restart:.4f} s")
    print(f"  Average per step: {avg_restart/total_steps*1000:.4f} ms")
    print(f"  Average per restart: {avg_restart/args.nrestarts:.4f} s")
    
    # Scaling analysis
    print("\n" + "=" * 70)
    print("Scaling Analysis")
    print("=" * 70)
    expected_time = avg_single * args.nrestarts
    overhead = avg_restart - expected_time
    overhead_pct = (overhead / expected_time * 100) if expected_time > 0 else 0
    print(f"  Expected time ({args.nrestarts} × single run): {expected_time:.4f} s")
    print(f"  Actual time (restart loop): {avg_restart:.4f} s")
    print(f"  Overhead: {overhead:.4f} s ({overhead_pct:.2f}%)")
    
    # Scaling with different restart counts
    print("\n  Scaling with number of restarts:")
    scaling_restarts = [1, 2, 4, 8]
    scaling_times = []
    for nr in scaling_restarts:
        elapsed, _ = benchmark_restart_loop(
            data,
            nrestarts=nr,
            sa_nsteps=args.nsteps,
            ga_nsteps=args.nsteps,
            warmup=(nr == scaling_restarts[0]),
            use_pre_molecular=use_pre_molecular,
        )
        scaling_times.append(elapsed)
        print(f"    {nr} restarts: {elapsed:.4f} s ({elapsed/nr:.4f} s/restart)")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Single run ({args.nsteps} steps): {avg_single:.4f} s")
    print(f"Restart loop ({args.nrestarts} restarts × {args.nsteps} steps): {avg_restart:.4f} s")
    print(f"Overhead per restart: {overhead/args.nrestarts:.4f} s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
