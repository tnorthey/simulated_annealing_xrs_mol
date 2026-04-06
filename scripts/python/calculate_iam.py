#!/usr/bin/env python3
"""
Standalone script to calculate IAM scattering signal from XYZ file(s).

This script reads an XYZ file (or XYZ trajectory) and calculates the IAM
(Independent Atom Model) scattering signal. It supports:
- Single XYZ files or XYZ trajectories
- PCD mode (percentage difference from reference XYZ or reference DAT intensity)
- Inelastic scattering (Compton scattering)
- Ewald sphere mode (3D scattering)

Usage:
    python3 calculate_iam.py input.xyz output.dat
    python3 calculate_iam.py input.xyz output.dat --reference reference.xyz --pcd
    python3 calculate_iam.py input.xyz output.dat --elastic --pcd --reference ref.xyz --reference-dat ref_I.dat --ab-initio-scattering ab_initio.dat [--ab-initio-correction-mode elastic|total]
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
from modules.wrap import _read_scattering_dat, _safe_ab_initio_correction_ratio


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
    parser.add_argument('--reference-dat', type=str, default=None,
                       dest='reference_dat',
                       help='Reference DAT (q, I_ref) for PCD denominator. With --ab-initio-scattering and --pcd, required: PCD = 100*(I_corr/I_ref-1) with I_corr=c×(atomic+molecular).')
    parser.add_argument('--ab-initio-scattering', type=str, default=None,
                       dest='ab_initio_scattering',
                       help='DAT with col1=q, col2=ab initio total I(q) at --reference (requires --reference). With --pcd, also pass --reference-dat for the PCD baseline I_ref(q).')
    parser.add_argument(
        '--ab-initio-correction-mode',
        type=str,
        choices=['elastic', 'total'],
        default='elastic',
        dest='ab_initio_correction_mode',
        help='With --ab-initio-scattering: elastic => c=I_abi/I_elastic_IAM(ref), output c×(atomic+molecular); '
             'total => c=I_abi/I_total_IAM(ref), output c×full IAM',
    )
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
    if args.pcd:
        if args.ab_initio_scattering:
            if not args.reference:
                parser.error(
                    "--ab-initio-scattering with --pcd requires --reference REF.xyz "
                    "(for c(q) = I_ab_initio / I_elastic_IAM(ref))"
                )
            if not args.reference_dat:
                parser.error(
                    "--ab-initio-scattering with --pcd requires --reference-dat REF.dat "
                    "(PCD compares c(q)×I_elastic to this I_ref(q))"
                )
        else:
            has_xyz = args.reference is not None
            has_dat = args.reference_dat is not None
            if has_xyz == has_dat:
                parser.error("--pcd requires exactly one of --reference or --reference-dat")
    
    if args.ewald and args.ab_initio_scattering:
        parser.error(
            "ab initio scattering correction is only supported for isotropic (non-Ewald) q; "
            "omit --ewald or omit --ab-initio-scattering"
        )
    if args.ab_initio_scattering and not args.reference:
        parser.error("--ab-initio-scattering requires --reference REF.xyz")
    
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
    
    # q-dependent correction factors (ones if not requested)
    if args.ab_initio_scattering:
        print(f"Computing correction factor from ab initio scattering: {args.ab_initio_scattering}...")
        if not os.path.exists(args.ab_initio_scattering):
            print(f"Error: File not found: {args.ab_initio_scattering}")
            sys.exit(1)
        q_abi, I_abi, abi_has_q = _read_scattering_dat(args.ab_initio_scattering)
        if not abi_has_q:
            print(
                "Error: ab initio file must have two columns (q and intensity)."
            )
            sys.exit(1)
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            sys.exit(1)
        ref_structures = read_xyz_trajectory(args.reference)
        _, _, ref_atomlist, ref_xyz = ref_structures[0]
        if not np.array_equal(ref_atomlist, atomlist):
            print("Error: --reference structure has different atom types than input")
            sys.exit(1)
        ref_atomic_numbers = [m.periodic_table(symbol) for symbol in ref_atomlist]
        out_dir = os.path.dirname(os.path.abspath(args.output_dat))
        if not out_dir:
            out_dir = os.getcwd()
        if args.ab_initio_correction_mode == "elastic":
            iam_ref_abi, _, _, _ = calculate_iam_for_structure(
                ref_xyz, ref_atomic_numbers, q_abi, x,
                ion=args.ion,
                inelastic=False, ewald_mode=False,
                th=None, ph=None, compton_array=None,
            )
            ref_iam_path = os.path.join(out_dir, "reference_iam_scattering.dat")
            np.savetxt(ref_iam_path, np.column_stack((q_abi, iam_ref_abi)))
            print(
                f"Wrote elastic IAM(ref) on ab initio q-grid to {ref_iam_path} "
                "(denominator for c = I_ab_initio_total / I_elastic_IAM)"
            )
        else:
            compton_abi = None
            if args.inelastic:
                compton_abi = x.compton_spline(atomic_numbers, q_abi)
            iam_ref_abi, _, _, _ = calculate_iam_for_structure(
                ref_xyz, ref_atomic_numbers, q_abi, x,
                ion=args.ion,
                inelastic=args.inelastic, ewald_mode=False,
                th=None, ph=None, compton_array=compton_abi,
            )
            ref_iam_path = os.path.join(out_dir, "reference_iam_total_scattering.dat")
            np.savetxt(ref_iam_path, np.column_stack((q_abi, iam_ref_abi)))
            print(
                f"Wrote total IAM(ref) on ab initio q-grid to {ref_iam_path} "
                "(denominator for c = I_ab_initio_total / I_total_IAM)"
            )
        corr_abi = _safe_ab_initio_correction_ratio(I_abi, iam_ref_abi)
        if q_abi.size != qvector.size or not np.allclose(q_abi, qvector):
            correction_factor_q = np.interp(
                qvector, q_abi, corr_abi, left=corr_abi[0], right=corr_abi[-1]
            )
            print(
                f"Interpolated ab-initio correction ratio from {len(q_abi)} points "
                f"to {len(qvector)} q-points"
            )
        else:
            correction_factor_q = np.asarray(corr_abi, dtype=np.float64)
            print(f"Ab initio q-grid matches configured qvector ({len(q_abi)} points)")
    else:
        correction_factor_q = np.ones(qvector.size, dtype=np.float64)
    
    # Calculate reference IAM if PCD mode
    reference_iam = None
    if args.pcd:
        if args.reference_dat:
            if args.ab_initio_scattering:
                _cmp = (
                    "c(q)×I_elastic"
                    if args.ab_initio_correction_mode == "elastic"
                    else "c(q)×I_total"
                )
                print(
                    f"Loading PCD baseline I_ref(q) from DAT (compared to {_cmp}): "
                    f"{args.reference_dat}"
                )
            else:
                print(f"Loading reference IAM from DAT file: {args.reference_dat}")
            if not os.path.exists(args.reference_dat):
                print(f"Error: Reference DAT file not found: {args.reference_dat}")
                sys.exit(1)
            ref_q, ref_iam, ref_has_explicit_q = _read_scattering_dat(args.reference_dat)
            if not ref_has_explicit_q:
                if ref_iam.size != qvector.size:
                    print(
                        f"Error: Single-column reference DAT has {ref_iam.size} points "
                        f"but qvector has {qvector.size} points. They must match."
                    )
                    sys.exit(1)
                reference_iam = ref_iam
            elif ref_q.size != qvector.size or not np.allclose(ref_q, qvector):
                reference_iam = np.interp(
                    qvector, ref_q, ref_iam, left=ref_iam[0], right=ref_iam[-1]
                )
                print(
                    f"Interpolated reference IAM from {len(ref_q)} points "
                    f"to {len(qvector)} points"
                )
            else:
                reference_iam = ref_iam
                print(f"Reference DAT q-grid matches qvector ({len(ref_q)} points)")
        else:
            print(f"Reading reference structure from {args.reference}...")
            if not os.path.exists(args.reference):
                print(f"Error: Reference file not found: {args.reference}")
                sys.exit(1)
            
            ref_structures = read_xyz_trajectory(args.reference)
            if len(ref_structures) > 1:
                print("Warning: Reference file contains multiple structures, using first one")
            _, _, ref_atomlist, ref_xyz = ref_structures[0]
            
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
        
        if args.ab_initio_scattering:
            if args.ab_initio_correction_mode == "elastic":
                iam = correction_factor_q * (atomic + molecular)
            else:
                iam = correction_factor_q * iam
        else:
            iam = iam * correction_factor_q
        
        if args.pcd:
            iam = calculate_pcd(iam, reference_iam)
        
        all_iam_signals.append(iam)
    
    # Write output
    print(f"Writing output to {args.output_dat}...")
    
    if len(all_iam_signals) == 1:
        # Single structure: write q, IAM/PCD columns
        output_data = np.column_stack((qvector, all_iam_signals[0]))
        ylabel = 'PCD (%)' if args.pcd else 'IAM signal'
        np.savetxt(args.output_dat, output_data,
                  fmt='%.6e', header=f'q (A^-1)    {ylabel}')
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
