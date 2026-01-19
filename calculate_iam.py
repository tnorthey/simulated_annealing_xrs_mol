#!/usr/bin/env python3
"""
Standalone script to calculate IAM scattering signal from XYZ file(s).

This script reads an XYZ file (or XYZ trajectory) and calculates the IAM
(Independent Atom Model) scattering signal. It supports:
- Single XYZ files or XYZ trajectories
- PCD mode (percentage difference from reference structure)
- Inelastic scattering (Compton scattering)
- Ewald sphere mode (3D scattering)

Usage:
    python3 calculate_iam.py input.xyz output.dat
    python3 calculate_iam.py input.xyz output.dat --reference reference.xyz --pcd
    python3 calculate_iam.py input.xyz output.dat --ewald
    python3 calculate_iam.py input.xyz output.dat --elastic
"""

import argparse
import numpy as np
import os
import sys

# Import modules
import modules.mol as mol
import modules.x as xray


def read_xyz_trajectory(filename):
    """
    Read XYZ file(s) - handles both single structure and trajectory.
    
    Returns:
        list of tuples: [(natoms, comment, atomlist, xyzmatrix), ...]
    """
    structures = []
    m = mol.Xyz()
    
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            # Read number of atoms
            try:
                natoms = int(line.strip())
            except (ValueError, AttributeError):
                break
            
            # Read comment line
            comment = f.readline().strip()
            
            # Read coordinates
            atomlist = []
            xyzmatrix = []
            for i in range(natoms):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 4:
                    break
                atomlist.append(parts[0])
                xyzmatrix.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            if len(atomlist) == natoms:
                structures.append((
                    natoms,
                    comment,
                    np.array(atomlist),
                    np.array(xyzmatrix)
                ))
            else:
                print(f"Warning: Incomplete structure found, skipping...")
                break
    
    if not structures:
        raise ValueError(f"No valid structures found in {filename}")
    
    return structures


def calculate_iam_for_structure(xyz, atomic_numbers, qvector, xray_obj, 
                                 ion=False,
                                 inelastic=True, ewald_mode=False,
                                 th=None, ph=None, compton_array=None):
    """
    Calculate IAM signal for a single structure.
    
    Returns:
        iam: IAM signal (1D or 3D depending on ewald_mode)
        atomic: Atomic contribution
        molecular: Molecular contribution  
        compton: Compton contribution (if inelastic)
    """
    if ewald_mode:
        if th is None or ph is None:
            raise ValueError("th and ph must be provided for Ewald mode")
        
        (iam, atomic, molecular, compton, pre_molecular,
         iam_total_rotavg, atomic_rotavg, molecular_rotavg, compton_rotavg
        ) = xray_obj.iam_calc_ewald(
            atomic_numbers, xyz, qvector, th, ph,
            ion=ion,
            inelastic=inelastic, compton_array=compton_array
        )
        # Return rotational average for output
        return iam_total_rotavg, atomic_rotavg, molecular_rotavg, compton_rotavg
    else:
        iam, atomic, molecular, compton, pre_molecular = xray_obj.iam_calc(
            atomic_numbers, xyz, qvector,
            ion=ion,
            electron_mode=False, inelastic=inelastic, compton_array=compton_array
        )
        return iam, atomic, molecular, compton


def calculate_pcd(iam, reference_iam):
    """
    Calculate PCD (Percentage Change Difference) signal.
    
    PCD = 100 * (IAM / reference_IAM - 1)
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pcd = 100 * (iam / reference_iam - 1)
        pcd = np.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description='Calculate IAM scattering signal from XYZ file(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('input_xyz', type=str,
                       help='Input XYZ file (single structure or trajectory)')
    parser.add_argument('output_dat', type=str,
                       help='Output DAT file with IAM signal')
    
    # Q-vector parameters
    parser.add_argument('--qmin', type=float, default=0.1,
                       help='Minimum q value (default: 0.1)')
    parser.add_argument('--qmax', type=float, default=8.0,
                       help='Maximum q value (default: 8.0)')
    parser.add_argument('--qlen', type=int, default=100,
                       help='Number of q points (default: 100)')
    
    # PCD mode
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference XYZ file for PCD calculation')
    parser.add_argument('--pcd', action='store_true',
                       help='Calculate PCD (percentage difference) instead of IAM')
    
    # Scattering options
    #
    # To avoid user error / ambiguity, require explicitly choosing elastic vs inelastic.
    scatter_mode = parser.add_mutually_exclusive_group(required=True)
    scatter_mode.add_argument(
        "--inelastic",
        dest="inelastic",
        action="store_true",
        help="Include inelastic (Compton) scattering",
    )
    scatter_mode.add_argument(
        "--elastic",
        dest="inelastic",
        action="store_false",
        help="Elastic-only scattering (disable Compton)",
    )
    parser.add_argument('--ion', action='store_true',
                       help='Include ion correction factors (dd/ee) in the atomic form factor')
    parser.add_argument('--ewald', action='store_true',
                       help='Use Ewald sphere mode (3D scattering)')
    
    # Ewald parameters (only used if --ewald is set)
    parser.add_argument('--tmin', type=float, default=0.0,
                       help='Minimum theta (in units of pi, default: 0.0)')
    parser.add_argument('--tmax', type=float, default=1.0,
                       help='Maximum theta (in units of pi, default: 1.0)')
    parser.add_argument('--tlen', type=int, default=21,
                       help='Number of theta points (default: 21)')
    parser.add_argument('--pmin', type=float, default=0.0,
                       help='Minimum phi (in units of pi, default: 0.0)')
    parser.add_argument('--pmax', type=float, default=2.0,
                       help='Maximum phi (in units of pi, default: 2.0)')
    parser.add_argument('--plen', type=int, default=21,
                       help='Number of phi points (default: 21)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pcd and args.reference is None:
        parser.error("--pcd requires --reference to be specified")
    
    if args.ewald and (args.tmin >= args.tmax or args.pmin >= args.pmax):
        parser.error("For Ewald mode: tmin < tmax and pmin < pmax required")
    
    if args.qmin >= args.qmax:
        parser.error("qmin must be less than qmax")
    
    if args.qmin < 0:
        parser.error("qmin must be non-negative")
    
    # Check input file exists
    if not os.path.exists(args.input_xyz):
        print(f"Error: Input file not found: {args.input_xyz}")
        sys.exit(1)
    
    # Initialize objects
    m = mol.Xyz()
    x = xray.Xray()
    
    # Read input structures
    print(f"Reading structures from {args.input_xyz}...")
    structures = read_xyz_trajectory(args.input_xyz)
    print(f"Found {len(structures)} structure(s)")
    
    # Get atomic numbers from first structure
    natoms, comment, atomlist, xyz = structures[0]
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    
    # Check all structures have same number of atoms
    for i, (n, _, al, _) in enumerate(structures):
        if n != natoms or len(al) != len(atomlist):
            print(f"Error: Structure {i+1} has different number of atoms")
            sys.exit(1)
        # Check atom types match
        if not np.array_equal(al, atomlist):
            print(f"Warning: Structure {i+1} has different atom types")
    
    # Setup q-vector
    qvector = np.linspace(args.qmin, args.qmax, args.qlen, endpoint=True)
    
    # Setup Ewald parameters if needed
    th = None
    ph = None
    if args.ewald:
        th = np.pi * np.linspace(args.tmin, args.tmax, args.tlen, endpoint=True)
        ph = np.pi * np.linspace(args.pmin, args.pmax, args.plen, endpoint=False)
        print(f"Ewald mode: q={len(qvector)}, theta={len(th)}, phi={len(ph)}")
    
    # Calculate Compton scattering if inelastic
    compton_array = None
    if args.inelastic:
        print("Calculating Compton scattering contributions...")
        try:
            compton_array = x.compton_spline(atomic_numbers, qvector)
        except FileNotFoundError:
            print("Warning: Compton data file not found, continuing without inelastic scattering")
            args.inelastic = False
    
    # Calculate reference IAM if PCD mode
    reference_iam = None
    if args.pcd:
        print(f"Reading reference structure from {args.reference}...")
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            sys.exit(1)
        
        ref_structures = read_xyz_trajectory(args.reference)
        if len(ref_structures) > 1:
            print("Warning: Reference file contains multiple structures, using first one")
        _, _, ref_atomlist, ref_xyz = ref_structures[0]
        
        # Check atom types match
        if not np.array_equal(ref_atomlist, atomlist):
            print("Error: Reference structure has different atom types")
            sys.exit(1)
        
        ref_atomic_numbers = [m.periodic_table(symbol) for symbol in ref_atomlist]
        print("Calculating reference IAM signal...")
        reference_iam, _, _, _ = calculate_iam_for_structure(
            ref_xyz, ref_atomic_numbers, qvector, x,
            ion=args.ion,
            inelastic=args.inelastic, ewald_mode=args.ewald,
            th=th, ph=ph, compton_array=compton_array
        )
    
    # Calculate IAM for all structures
    print("Calculating IAM signals...")
    all_iam_signals = []
    
    for i, (natoms, comment, atomlist, xyz) in enumerate(structures):
        if len(structures) > 1:
            print(f"  Processing structure {i+1}/{len(structures)}...")
        
        iam, atomic, molecular, compton = calculate_iam_for_structure(
            xyz, atomic_numbers, qvector, x,
            ion=args.ion,
            inelastic=args.inelastic, ewald_mode=args.ewald,
            th=th, ph=ph, compton_array=compton_array
        )
        
        # Apply PCD if requested
        if args.pcd:
            iam = calculate_pcd(iam, reference_iam)
        
        all_iam_signals.append(iam)
    
    # Write output
    print(f"Writing output to {args.output_dat}...")
    
    if len(all_iam_signals) == 1:
        # Single structure: write q, IAM columns
        output_data = np.column_stack((qvector, all_iam_signals[0]))
        np.savetxt(args.output_dat, output_data,
                  fmt='%.6e', header='q (A^-1)    IAM signal')
    else:
        # Trajectory: write q, IAM_1, IAM_2, ... columns
        output_data = np.column_stack([qvector] + all_iam_signals)
        header = 'q (A^-1)    ' + '    '.join([f'IAM_{i+1}' for i in range(len(all_iam_signals))])
        np.savetxt(args.output_dat, output_data,
                  fmt='%.6e', header=header)
        print(f"  Wrote {len(all_iam_signals)} IAM signals to output file")
    
    print("Done!")


if __name__ == '__main__':
    main()
