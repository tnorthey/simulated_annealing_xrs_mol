#!/usr/bin/env python3
"""
Standalone script to analyze molecular geometry from XYZ file(s).

This script reads an XYZ file (or XYZ trajectory) and calculates:
- Dihedral angles (4 atoms: i-j-k-l)
- Angles (3 atoms: i-j-k)
- Bond lengths (2 atoms: i-j)

Usage:
    # Calculate bond length between atoms 0 and 1
    python3 analyze_geometry.py input.xyz --bond 0 1
    
    # Calculate angle between atoms 0, 1, 2
    python3 analyze_geometry.py input.xyz --angle 0 1 2
    
    # Calculate dihedral angle between atoms 0, 1, 2, 3
    python3 analyze_geometry.py input.xyz --dihedral 0 1 2 3
    
    # Multiple calculations
    python3 analyze_geometry.py input.xyz --bond 0 1 --angle 0 1 2 --dihedral 0 1 2 3
    
    # For trajectory, output to CSV
    python3 analyze_geometry.py trajectory.xyz --bond 0 1 --output results.csv
"""

import argparse
import numpy as np
import os
import sys
import csv

# Import modules
import modules.mol as mol
import modules.analysis as analysis


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


# Use functions from analysis module
calculate_bond_length = analysis.calculate_bond_length
calculate_angle = analysis.calculate_angle
calculate_dihedral = analysis.calculate_dihedral


def validate_indices(indices, natoms, name):
    """Validate that atom indices are within bounds."""
    for idx in indices:
        if idx < 0 or idx >= natoms:
            raise ValueError(
                f"Invalid {name} atom index: {idx}. "
                f"Must be between 0 and {natoms - 1} (0-indexed)."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze molecular geometry from XYZ file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'input_xyz',
        type=str,
        help='Input XYZ file (single structure or trajectory)'
    )
    
    parser.add_argument(
        '--bond',
        type=int,
        nargs=2,
        metavar=('I', 'J'),
        action='append',
        help='Calculate bond length between atoms I and J (0-indexed). Can be specified multiple times.'
    )
    
    parser.add_argument(
        '--angle',
        type=int,
        nargs=3,
        metavar=('I', 'J', 'K'),
        action='append',
        help='Calculate angle I-J-K in degrees (0-indexed). Can be specified multiple times.'
    )
    
    parser.add_argument(
        '--dihedral',
        type=int,
        nargs=4,
        metavar=('I', 'J', 'K', 'L'),
        action='append',
        help='Calculate dihedral angle I-J-K-L in degrees (0-indexed). Can be specified multiple times.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file (CSV format). If not specified, prints to stdout.'
    )
    
    parser.add_argument(
        '--units',
        type=str,
        choices=['angstrom', 'bohr'],
        default='angstrom',
        help='Units for bond lengths (default: angstrom)'
    )
    
    args = parser.parse_args()
    
    # Check that at least one calculation type is specified
    if not args.bond and not args.angle and not args.dihedral:
        parser.error("At least one of --bond, --angle, or --dihedral must be specified")
    
    # Check input file exists
    if not os.path.exists(args.input_xyz):
        print(f"Error: Input file '{args.input_xyz}' not found.")
        sys.exit(1)
    
    # Read structures
    try:
        structures = read_xyz_trajectory(args.input_xyz)
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        sys.exit(1)
    
    nstructures = len(structures)
    print(f"Read {nstructures} structure(s) from {args.input_xyz}")
    
    # Validate indices for first structure
    natoms = structures[0][0]
    if args.bond:
        for bond in args.bond:
            validate_indices(bond, natoms, "bond")
    if args.angle:
        for angle in args.angle:
            validate_indices(angle, natoms, "angle")
    if args.dihedral:
        for dihedral in args.dihedral:
            validate_indices(dihedral, natoms, "dihedral")
    
    # Prepare output
    output_file = open(args.output, 'w') if args.output else sys.stdout
    
    # Prepare CSV writer
    writer = None
    if args.output or nstructures > 1:
        # Use CSV format for multiple structures or when output file is specified
        writer = csv.writer(output_file)
        
        # No header - just write data rows
    
    # Convert units if needed
    unit_factor = 1.0
    unit_label = 'Å'
    if args.units == 'bohr':
        unit_factor = 1.8897259886  # Å to Bohr
        unit_label = 'Bohr'
    
    # Process each structure
    for frame_idx, (natoms, comment, atomlist, xyz) in enumerate(structures):
        results = []
        
        # Calculate bonds
        if args.bond:
            for bond in args.bond:
                i, j = bond
                length = calculate_bond_length(xyz, i, j) * unit_factor
                results.append(length)
        
        # Calculate angles
        if args.angle:
            for angle in args.angle:
                i, j, k = angle
                angle_val = calculate_angle(xyz, i, j, k)
                results.append(angle_val)
        
        # Calculate dihedrals
        if args.dihedral:
            for dihedral in args.dihedral:
                i, j, k, l = dihedral
                dihedral_val = calculate_dihedral(xyz, i, j, k, l)
                results.append(dihedral_val)
        
        # Output results
        if writer:
            # CSV format (only calculation results, no frame/comment)
            writer.writerow(results)
        else:
            # Single structure, human-readable format
            print(f"\nStructure {frame_idx + 1}: {comment}")
            result_idx = 0
            
            if args.bond:
                print("Bond lengths:")
                for bond in args.bond:
                    i, j = bond
                    length = results[result_idx]
                    print(f"  Bond {i}-{j}: {length:.6f} {unit_label}")
                    result_idx += 1
            
            if args.angle:
                print("Angles:")
                for angle in args.angle:
                    i, j, k = angle
                    angle_val = results[result_idx]
                    print(f"  Angle {i}-{j}-{k}: {angle_val:.6f}°")
                    result_idx += 1
            
            if args.dihedral:
                print("Dihedral angles:")
                for dihedral in args.dihedral:
                    i, j, k, l = dihedral
                    dihedral_val = results[result_idx]
                    print(f"  Dihedral {i}-{j}-{k}-{l}: {dihedral_val:.6f}°")
                    result_idx += 1
    
    if args.output:
        output_file.close()
        print(f"\nResults written to {args.output}")
    elif nstructures > 1:
        print("\nResults printed above (use --output to save to file)")


if __name__ == '__main__':
    main()
