from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit
from openmm import HarmonicBondForce
from openmm import HarmonicAngleForce
from openmm.openmm import PeriodicTorsionForce
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import rdMolTransforms
from openbabel import openbabel, pybel
from openff.toolkit.utils.toolkits import ToolkitRegistry, BuiltInToolkitWrapper
import numpy as np


class Openff_retreive_mm_params:
    def __init__(self):
        pass

    def read_xyz(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        num_atoms = int(lines[0])
        atom_lines = lines[2 : 2 + num_atoms]
        symbols = []
        coords = []
        for line in atom_lines:
            parts = line.strip().split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return symbols, np.array(coords)

    def openbabel_xyz2sdf(self, xyz_input_file, sdf_output_file):
        """Uses OpenBabel to guess bonds and create an SDF file from an XYZ file"""
        mol = next(pybel.readfile("xyz", xyz_input_file))  # Read xyz file
        # Step 2: Add hydrogens and generate 3D structure
        # mol.addh()
        # mol.make3D()  # Optionally improve geometry
        # Step 3: (Optional) optimize geometry with UFF
        # mol.localopt(forcefield="uff")
        # Step 4: Write to SDF — this will now include correct bond orders
        mol.write("sdf", sdf_output_file, overwrite=True)

    def rdkit_xyz2sdf(self, xyz_input_file, sdf_output_file):
        """Uses RDKit to guess bonds and create an SDF file from an XYZ file"""
        # Step 1: Read XYZ file manually
        symbols, coords = self.read_xyz(xyz_input_file)
        # Step 2: Build RDKit molecule
        mol = RWMol()
        atom_indices = []
        for symbol in symbols:
            atom = Chem.Atom(symbol)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)

        # Step 3: Add bonds based on distance (very simple guess)
        def guess_bonds(mol, coords, cutoff=1.8):
            """Guess bonds only by distance cutoff"""
            n = len(coords)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < cutoff:
                        mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

        def add_bonds_smart(mol, coords, scale=1.2):
            pt = rdchem.GetPeriodicTable()
            n = len(coords)
            for i in range(n):
                ri = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
                for j in range(i + 1, n):
                    rj = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
                    max_bond = (ri + rj) * scale
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist <= max_bond:
                        mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

        # Then call:
        # guess_bonds(mol, coords)
        add_bonds_smart(mol, coords)
        # Step 4: Add 3D coordinates
        conf = Chem.Conformer(len(symbols))
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(*pos))
        mol.AddConformer(conf)
        mol = mol.GetMol()  # Finalize edits
        Chem.SanitizeMol(mol)
        # Step 6: Save to SDF
        writer = Chem.SDWriter(sdf_output_file)
        writer.write(mol)
        writer.close()
        return

    def create_topology_from_sdf(
        self, sdf_file, ff_file="openff_unconstrained-2.0.0.offxml", debug_bool=False
    ):
        """creates an OpenMM system (topology, forcefield) from the sdf_file"""
        # 0: use toolkit registry
        toolkit_registry = ToolkitRegistry()
        toolkit_registry.register_toolkit(BuiltInToolkitWrapper())
        # Step 1: Load the SDF with RDKit WITHOUT removing hydrogens
        rdkit_mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
        # Step 2: Convert to OpenFF Molecule, preserving explicit atoms
        off_mol = Molecule.from_rdkit(
            rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
        # Other step: avoid an error with partial charge assignment later (which isn't needed anyway)

        off_mol.assign_partial_charges("formal_charge", toolkit_registry=toolkit_registry)
        # Step 3: Build the Topology
        topology = Topology.from_molecules(off_mol)
        if debug_bool:
            # Step 4: Check atom counts
            print("RDKit atoms:", rdkit_mol.GetNumAtoms())
            print("OpenFF molecule atoms:", off_mol.n_atoms)
            print("OpenFF topology atoms:", topology.n_atoms)
        # Now add the forcfield
        ff = ForceField(ff_file)
        # ff = ForceField("openff-2.1.0.offxml")  # this one restrains C-H bonds and doesn't give me the bond strengths
        # Create an OpenMM system
        openmm_system = ff.create_openmm_system(topology)
        print(
            "OpenMM system created with", openmm_system.getNumParticles(), "particles."
        )
        return topology, openmm_system

    def retreive_bonds_k_values(self, topology, openmm_system):
        """gets the bond-lengths and bond-strengths from the OpenMM system (topology, forcefield)"""
        # Get the HarmonicBondForce from the OpenMM system
        bond_force = next(
            f for f in openmm_system.getForces() if isinstance(f, HarmonicBondForce)
        )
        # Get the list of atoms for type info
        atoms = list(topology.atoms)
        nbonds = bond_force.getNumBonds()
        k_kcal_per_ang2_array = np.zeros(nbonds)
        length_angstrom_array = np.zeros(nbonds)
        atom1_idx_array = np.zeros(nbonds)
        atom2_idx_array = np.zeros(nbonds)
        # Loop through the bonds in the OpenMM system
        for bond_index in range(nbonds):
            atom1_idx, atom2_idx, length, k = bond_force.getBondParameters(bond_index)
            atom1_idx_array[bond_index] = atom1_idx
            atom2_idx_array[bond_index] = atom2_idx
            length_angstrom = length.value_in_unit(unit.angstrom)
            length_angstrom_array[bond_index] = length_angstrom
            k_kcal_per_ang2 = k.value_in_unit(
                unit.kilocalories_per_mole / unit.angstrom**2
            )
            k_kcal_per_ang2_array[bond_index] = k_kcal_per_ang2
            # print(f"Bond {bond_index}: {atom1.symbol}-{atom2.symbol} "
            #      f"({atom1_idx}-{atom2_idx})")
            # print(f"  Length: {length_angstrom:.3f} Å")
            # print(f"  Force constant: {k_kcal_per_ang2} kcal/(mol Å^2)")
        return (
            atom1_idx_array,
            atom2_idx_array,
            length_angstrom_array,
            k_kcal_per_ang2_array,
        )

    def retreive_angles_k_values(self, topology, openmm_system):
        """Gets the angle equilibrium values and force constants from the OpenMM system (topology, forcefield)."""
        # Get the HarmonicAngleForce from the system
        angle_force = next(
            f for f in openmm_system.getForces() if isinstance(f, HarmonicAngleForce)
        )
        natoms = topology.n_atoms  # Not strictly necessary but useful if validating
        nangles = angle_force.getNumAngles()
        # Preallocate arrays
        atom1_idx_array = np.zeros(nangles, dtype=int)
        atom2_idx_array = np.zeros(nangles, dtype=int)
        atom3_idx_array = np.zeros(nangles, dtype=int)
        angle_rad_array = np.zeros(nangles)
        k_kcal_per_rad2_array = np.zeros(nangles)
        # Loop through the angles
        for i in range(nangles):
            a1, a2, a3, theta, k = angle_force.getAngleParameters(i)
            atom1_idx_array[i] = a1
            atom2_idx_array[i] = a2
            atom3_idx_array[i] = a3
            angle_rad_array[i] = theta.value_in_unit(unit.radian)
            k_kcal_per_rad2_array[i] = k.value_in_unit(
                unit.kilocalories_per_mole / unit.radian**2
            )
        return (
            atom1_idx_array,
            atom2_idx_array,
            atom3_idx_array,
            angle_rad_array,
            k_kcal_per_rad2_array,
        )


    def extract_periodic_torsions(self, system):
        """Extract torsion parameters from an OpenMM PeriodicTorsionForce.
    
        Returns:
            atom1_idx_array (np.ndarray): Atom 1 indices
            atom2_idx_array (np.ndarray): Atom 2 indices
            atom3_idx_array (np.ndarray): Atom 3 indices
            atom4_idx_array (np.ndarray): Atom 4 indices
            angle_rad_array (np.ndarray): Phase angle (delta) in radians
            k_kcal_per_rad2_array (np.ndarray): Torsion force constants in kcal/mol
        """
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if isinstance(force, PeriodicTorsionForce):
                torsion_force = force
                break
        else:
            raise ValueError("No PeriodicTorsionForce found in the system.")
    
        atom1_list = []
        atom2_list = []
        atom3_list = []
        atom4_list = []
        angle_list = []
        k_list = []
    
        for idx in range(torsion_force.getNumTorsions()):
            i, j, k, l, periodicity, phase, k_torsion = torsion_force.getTorsionParameters(idx)
    
            atom1_list.append(i)
            atom2_list.append(j)
            atom3_list.append(k)
            atom4_list.append(l)
    
            # Store phase in radians and k in kcal/mol
            angle_list.append(phase.value_in_unit(unit.radian))
            k_list.append(k_torsion.value_in_unit(unit.kilocalories_per_mole))
    
        return (
            np.array(atom1_list, dtype=int),
            np.array(atom2_list, dtype=int),
            np.array(atom3_list, dtype=int),
            np.array(atom4_list, dtype=int),
            np.array(angle_list),
            np.array(k_list)
        )

    def torsion_angle(self, a, b, c, d):
        b1 = b - a
        b2 = c - b
        b3 = d - c
    
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
    
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
    
        m1 = np.cross(n1, b2/np.linalg.norm(b2))
    
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
    
        return np.arctan2(y, x)  # radians


    def update_torsion_deltas(self, torsion_param_array, xyz):
        """
        torsion_param_array: np.array with columns:
            atom1_idx, atom2_idx, atom3_idx, atom4_idx, delta (rad), k (kcal/rad²)
        xyz: xyz coordinates, shape (N, 3)
        """
    
        for row in torsion_param_array:
            i, j, k, l = map(int, row[:4])
    
            # OpenMM's internal torsion computation
            angle = self.torsion_angle(
                xyz[i, :], xyz[j, :], xyz[k, :], xyz[l, :]
            )

            row[4] = angle  # overwrite delta with xyz torsions
    
        return torsion_param_array

