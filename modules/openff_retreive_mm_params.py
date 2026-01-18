from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils.exceptions import RadicalsNotSupportedError
from openmm import unit
from openmm import HarmonicBondForce
from openmm import HarmonicAngleForce
from openmm.openmm import PeriodicTorsionForce
from openmm import app
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

    def fix_radicals(self, rdkit_mol):
        """Fix radicals in RDKit molecule by setting radical electrons to 0"""
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(rdkit_mol)
        # Iterate through all atoms and set radical electrons to 0
        for atom in mol_copy.GetAtoms():
            num_radicals = atom.GetNumRadicalElectrons()
            if num_radicals > 0:
                atom.SetNumRadicalElectrons(0)
                # If removing radicals causes valence issues, try to fix by adjusting bond orders
                # This is a heuristic: if an atom has too many bonds after removing radicals,
                # we might need to adjust bond orders
        # Try to sanitize the molecule after fixing radicals
        try:
            Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
        except:
            try:
                # Try without radical checking
                Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
            except:
                # If sanitization fails completely, just return the molecule with radicals removed
                pass
        # Double-check: ensure all radicals are actually removed
        for atom in mol_copy.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
        return mol_copy
    
    def fix_molecule_issues(self, rdkit_mol):
        """Aggressively fix molecule issues: radicals, bond orders, and valence problems"""
        # First, use the dedicated fix_radicals method
        mol_copy = self.fix_radicals(rdkit_mol)
        
        # Step 1: Try to fix bond orders using RDKit's Kekulize
        try:
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        except:
            pass
        
        # Step 2: Try to fix valence issues by adjusting bond orders
        # If an atom has too many bonds, try to convert single bonds to aromatic or adjust
        try:
            # Try sanitizing with AllChem which is more comprehensive
            AllChem.SanitizeMol(mol_copy)
        except:
            # If full sanitization fails, try partial
            try:
                Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            except:
                try:
                    Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
                except:
                    pass
        
        # Step 3: Try to set aromatic flags properly
        try:
            Chem.SetAromaticity(mol_copy)
        except:
            pass
        
        # Step 4: Final radical check and removal
        for atom in mol_copy.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
        
        return mol_copy

    def simplify_bond_orders(self, rdkit_mol):
        """
        Simplify bond orders in RDKit molecule to make it more compatible with OpenFF.
        Converts double/triple/aromatic bonds to single bonds as a fallback strategy.
        This allows using C-C parameters when C=C parameters aren't available.
        
        Args:
            rdkit_mol: RDKit molecule
            
        Returns:
            Simplified RDKit molecule with all bonds set to single
        """
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(rdkit_mol)
        
        # Convert all bonds to single bonds
        for bond in mol_copy.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type != Chem.rdchem.BondType.SINGLE:
                # Convert double, triple, aromatic to single
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        
        # Clear aromatic flags since we've converted aromatic bonds to single
        try:
            Chem.SetAromaticity(mol_copy, clearAromaticFlags=True)
        except:
            pass
        
        return mol_copy

    def extract_params_from_geometry(self, xyz_file, xyz_coords=None):
        """
        Extract MM parameters directly from geometry without OpenFF.
        Uses starting geometry as equilibrium values and generic force constants.
        This is a last-resort fallback that works for any molecule structure.
        
        Args:
            xyz_file: Path to XYZ file (used to get atom symbols if xyz_coords not provided)
            xyz_coords: Optional numpy array of coordinates (shape: N, 3)
        
        Returns:
            bond_param_array: [atom1_idx, atom2_idx, r0 (Å), k (kcal/mol/Å²)]
            angle_param_array: [atom1_idx, atom2_idx, atom3_idx, theta0 (rad), k (kcal/mol/rad²)]
            torsion_param_array: [atom1_idx, atom2_idx, atom3_idx, atom4_idx, delta (rad), k (kcal/mol)]
        """
        # Read XYZ if coordinates not provided
        if xyz_coords is None:
            symbols, coords = self.read_xyz(xyz_file)
            xyz_coords = coords
        else:
            # Still need symbols for bond detection
            symbols, _ = self.read_xyz(xyz_file)
        
        natoms = len(xyz_coords)
        
        # Build RDKit molecule to identify connectivity
        mol = RWMol()
        for symbol in symbols:
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)
        
        # Add bonds based on distance
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
        
        add_bonds_smart(mol, xyz_coords)
        mol = mol.GetMol()
        
        # Extract bonds
        bond_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Calculate current bond length
            r0 = np.linalg.norm(xyz_coords[i] - xyz_coords[j])
            # Generic force constant (typical C-C bond ~500 kcal/mol/Å²)
            # Scale based on atom types
            atom_i = mol.GetAtomWithIdx(i).GetAtomicNum()
            atom_j = mol.GetAtomWithIdx(j).GetAtomicNum()
            # Simple heuristic: heavier atoms = stronger bonds
            avg_atomic_num = (atom_i + atom_j) / 2.0
            k_bond = 400.0 + 50.0 * avg_atomic_num  # kcal/mol/Å²
            bond_list.append([i, j, r0, k_bond])
        
        bond_param_array = np.array(bond_list)
        
        # Extract angles
        angle_list = []
        for i in range(natoms):
            atom_i = mol.GetAtomWithIdx(i)
            neighbors = [n.GetIdx() for n in atom_i.GetNeighbors()]
            # For each pair of neighbors, create an angle
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    n1 = neighbors[j]
                    n2 = neighbors[k]
                    # Calculate current angle
                    vec1 = xyz_coords[n1] - xyz_coords[i]
                    vec2 = xyz_coords[n2] - xyz_coords[i]
                    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta0 = np.arccos(cos_theta)  # radians
                    # Generic angle force constant (~50-100 kcal/mol/rad²)
                    k_angle = 60.0  # kcal/mol/rad²
                    angle_list.append([n1, i, n2, theta0, k_angle])
        
        angle_param_array = np.array(angle_list) if angle_list else np.empty((0, 5))
        
        # Extract torsions (dihedrals)
        torsion_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Get neighbors of i (excluding j)
            neighbors_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
            # Get neighbors of j (excluding i)
            neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
            # Create torsions for each combination
            for k in neighbors_i:
                for l in neighbors_j:
                    # Calculate current dihedral angle
                    p0 = xyz_coords[k]
                    p1 = xyz_coords[i]
                    p2 = xyz_coords[j]
                    p3 = xyz_coords[l]
                    b0 = -(p1 - p0)
                    b1 = p2 - p1
                    b2 = p3 - p2
                    b1_norm = b1 / np.linalg.norm(b1)
                    v = b0 - np.dot(b0, b1_norm) * b1_norm
                    w = b2 - np.dot(b2, b1_norm) * b1_norm
                    x = np.dot(v, w)
                    y = np.dot(np.cross(b1_norm, v), w)
                    delta = np.arctan2(y, x)  # radians
                    # Generic torsion force constant (~1-5 kcal/mol)
                    k_torsion = 2.0  # kcal/mol
                    torsion_list.append([k, i, j, l, delta, k_torsion])
        
        torsion_param_array = np.array(torsion_list) if torsion_list else np.empty((0, 6))
        
        return bond_param_array, angle_param_array, torsion_param_array

    def create_topology_from_sdf_robust(self, sdf_file, ff_file, debug_bool=False):
        """
        Robust version of create_topology_from_sdf that applies the same fixes
        as the XYZ robust method.
        """
        # Load SDF with RDKit
        rdkit_mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
        if rdkit_mol is None:
            raise ValueError(f"Failed to read SDF file: {sdf_file}")
        
        # Apply the same fixes as robust XYZ method
        # Fix radicals
        for atom in rdkit_mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
        
        # Apply comprehensive fixes
        rdkit_mol = self.fix_molecule_issues(rdkit_mol)
        
        # Try to convert to OpenFF
        off_mol = None
        bond_simplification_used = False
        
        try:
            off_mol = Molecule.from_rdkit(
                rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        except Exception as e:
            print(f"Warning: OpenFF conversion failed, trying simplified bond orders...")
            rdkit_mol = self.simplify_bond_orders(rdkit_mol)
            for atom in rdkit_mol.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    atom.SetNumRadicalElectrons(0)
            try:
                off_mol = Molecule.from_rdkit(
                    rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
                )
                bond_simplification_used = True
                print("Successfully created OpenFF molecule with simplified bond orders from SDF.")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to create OpenFF molecule from SDF even with simplified bonds: {e2}"
                ) from e2
        
        # Create topology and system
        topology = Topology.from_molecules(off_mol)
        toolkit_registry = ToolkitRegistry()
        toolkit_registry.register_toolkit(BuiltInToolkitWrapper())
        off_mol.assign_partial_charges("formal_charge", toolkit_registry=toolkit_registry)
        
        ff = ForceField(ff_file)
        openmm_system = ff.create_openmm_system(topology)
        
        if bond_simplification_used:
            print("Note: Using simplified bond orders (approximate parameters) from SDF.")
        
        return topology, openmm_system

    def create_topology_from_xyz_robust(
        self, xyz_file, ff_file="openff_unconstrained-2.0.0.offxml", debug_bool=False
    ):
        """
        More robust method: Create OpenMM system directly from XYZ using OpenFF Toolkit.
        This bypasses SDF file creation and avoids RadicalsNotSupportedError by fixing
        radicals proactively in RDKit before conversion.
        
        Args:
            xyz_file: Path to XYZ file
            ff_file: Path to OpenFF force field file (.offxml)
            debug_bool: Enable debug output
        
        Returns:
            topology: OpenMM Topology
            openmm_system: OpenMM System
        """
        # Step 1: Read XYZ file
        symbols, coords = self.read_xyz(xyz_file)
        
        # Step 2: Build RDKit molecule (more tolerant of radicals than OpenFF)
        mol = RWMol()
        for symbol in symbols:
            atom = Chem.Atom(symbol)
            idx = mol.AddAtom(atom)
        
        # Step 3: Add bonds based on distance (using existing smart method)
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
        
        add_bonds_smart(mol, coords)
        
        # Step 4: Add 3D coordinates
        conf = Chem.Conformer(len(symbols))
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(*pos))
        mol.AddConformer(conf)
        mol = mol.GetMol()  # Finalize edits
        
        # Step 5: Try to sanitize (but continue even if it fails)
        try:
            Chem.SanitizeMol(mol)
        except:
            # Even if sanitization fails, we can still proceed
            print("Warning: RDKit sanitization failed, but proceeding anyway...")
        
        # Step 6: Fix radicals proactively (set to 0) - this is the key to avoiding errors
        radicals_found = False
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
                radicals_found = True
        
        if radicals_found:
            print("Warning: Radicals detected and fixed in RDKit molecule before OpenFF conversion.")
        
        # Step 7: Apply comprehensive molecule fixing
        mol = self.fix_molecule_issues(mol)
        
        # Step 8: Double-check all radicals are removed
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
        
        # Step 9: Convert RDKit molecule directly to OpenFF (no PDB needed)
        # This works because we've fixed radicals proactively
        # Try with original bond orders first
        off_mol = None
        last_error = None
        bond_simplification_used = False
        
        try:
            off_mol = Molecule.from_rdkit(
                mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        except (RadicalsNotSupportedError, Exception) as e:
            last_error = e
            # If conversion fails, try simplifying bond orders (C=C -> C-C, aromatic -> single)
            print(f"Warning: OpenFF conversion failed with original bond orders: {type(e).__name__}")
            print("Attempting to simplify bond orders (e.g., C=C -> C-C) to use similar parameters...")
            
            # Simplify bond orders
            mol_simplified = self.simplify_bond_orders(mol)
            bond_simplification_used = True
            
            # Fix radicals again after bond simplification
            for atom in mol_simplified.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    atom.SetNumRadicalElectrons(0)
            
            try:
                off_mol = Molecule.from_rdkit(
                    mol_simplified, allow_undefined_stereo=True, hydrogens_are_explicit=True
                )
                print("Successfully created OpenFF molecule with simplified bond orders.")
                print("Note: Using approximate parameters (e.g., C-C instead of C=C).")
            except Exception as e2:
                last_error = e2
                # If simplified version also fails, provide helpful error message
                num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol_simplified.GetAtoms())
                error_msg = (
                    f"Failed to create OpenFF molecule from XYZ file: {xyz_file}\n"
                    f"Both original and simplified bond orders failed.\n\n"
                    f"Diagnostics:\n"
                    f"  - Number of atoms: {mol_simplified.GetNumAtoms()}\n"
                    f"  - Number of bonds: {mol_simplified.GetNumBonds()}\n"
                    f"  - Total radical electrons: {num_radicals}\n"
                    f"  - Last error: {type(last_error).__name__}: {last_error}\n\n"
                    f"Suggestions:\n"
                    f"  - Check the input XYZ file for chemically sensible connectivity.\n"
                    f"  - Ensure bond orders are correct in the starting geometry.\n"
                    f"  - Try using the SDF method instead (fallback will be attempted).\n"
                    f"  - Manually fix the molecule structure if needed."
                )
                raise RuntimeError(error_msg) from e2
        
        if off_mol is None:
            # This shouldn't happen, but just in case
            raise RuntimeError("Failed to create OpenFF molecule: unknown error")
        
        # Step 10: Create OpenFF topology and assign charges
        topology = Topology.from_molecules(off_mol)
        toolkit_registry = ToolkitRegistry()
        toolkit_registry.register_toolkit(BuiltInToolkitWrapper())
        off_mol.assign_partial_charges("formal_charge", toolkit_registry=toolkit_registry)
        
        # Step 11: Create OpenMM system with OpenFF force field
        try:
            ff = ForceField(ff_file)
            openmm_system = ff.create_openmm_system(topology)
        except Exception as e:
            # If system creation fails, it might be due to missing parameters
            # Try with simplified bond orders if we haven't already
            if not bond_simplification_used:
                print(f"Warning: OpenMM system creation failed: {type(e).__name__}")
                print("Retrying with simplified bond orders...")
                
                # Simplify bonds and try again
                mol_simplified = self.simplify_bond_orders(mol)
                for atom in mol_simplified.GetAtoms():
                    if atom.GetNumRadicalElectrons() > 0:
                        atom.SetNumRadicalElectrons(0)
                
                try:
                    off_mol_simplified = Molecule.from_rdkit(
                        mol_simplified, allow_undefined_stereo=True, hydrogens_are_explicit=True
                    )
                    topology = Topology.from_molecules(off_mol_simplified)
                    toolkit_registry = ToolkitRegistry()
                    toolkit_registry.register_toolkit(BuiltInToolkitWrapper())
                    off_mol_simplified.assign_partial_charges("formal_charge", toolkit_registry=toolkit_registry)
                    openmm_system = ff.create_openmm_system(topology)
                    bond_simplification_used = True
                    off_mol = off_mol_simplified  # Update reference for debug output
                    print("Successfully created OpenMM system with simplified bond orders.")
                    print("Note: Using approximate parameters (e.g., C-C instead of C=C).")
                except Exception as e2:
                    error_msg = (
                        f"Failed to create OpenMM system from XYZ file: {xyz_file}\n"
                        f"Error during system creation: {type(e2).__name__}: {e2}\n\n"
                        f"Suggestions:\n"
                        f"  - Check the input XYZ file for correctness.\n"
                        f"  - Try using the SDF method instead (fallback will be attempted)."
                    )
                    raise RuntimeError(error_msg) from e2
            else:
                # Already tried simplified bonds, raise error
                error_msg = (
                    f"Failed to create OpenMM system from XYZ file: {xyz_file}\n"
                    f"Error: {type(e).__name__}: {e}\n\n"
                    f"Even simplified bond orders failed. Suggestions:\n"
                    f"  - Check the input XYZ file for correctness.\n"
                    f"  - Try using the SDF method instead (fallback will be attempted)."
                )
                raise RuntimeError(error_msg) from e
        
        if debug_bool:
            print(f"RDKit atoms: {mol.GetNumAtoms()}")
            print(f"OpenFF molecule atoms: {off_mol.n_atoms}")
            print(f"OpenMM topology atoms: {topology.n_atoms}")
            print(f"OpenMM system particles: {openmm_system.getNumParticles()}")
            if bond_simplification_used:
                print("Bond simplification was used (approximate parameters).")
        
        if bond_simplification_used:
            print(f"OpenMM system created with {openmm_system.getNumParticles()} particles using {ff_file}.")
            print("(Using simplified bond orders - approximate parameters)")
        else:
            print(f"OpenMM system created with {openmm_system.getNumParticles()} particles using {ff_file}.")
        
        return topology, openmm_system

    def create_topology_from_sdf(
        self, sdf_file, ff_file="openff_unconstrained-2.0.0.offxml", debug_bool=False, 
        fix_radicals_always=True, save_fixed_sdf=False, perturbation_step_size=0.01
    ):
        """creates an OpenMM system (topology, forcefield) from the sdf_file
        
        Args:
            sdf_file: Path to SDF file
            ff_file: Path to force field file
            debug_bool: Enable debug output
            fix_radicals_always: If True, always fix radicals proactively (recommended)
            save_fixed_sdf: If True, save the fixed SDF back to disk (overwrites original)
        """
        # 0: use toolkit registry
        toolkit_registry = ToolkitRegistry()
        toolkit_registry.register_toolkit(BuiltInToolkitWrapper())
        # Step 1: Load the SDF with RDKit WITHOUT removing hydrogens
        rdkit_mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
        
        # Step 1.5: Proactively fix molecule issues if requested (recommended)
        # This prevents errors and ensures the molecule is compatible with OpenFF
        molecule_fixed = False
        if fix_radicals_always:
            # Check if radicals or other issues exist
            has_radicals = any(atom.GetNumRadicalElectrons() > 0 for atom in rdkit_mol.GetAtoms())
            if has_radicals:
                print("Warning: Radicals detected in molecule. Attempting comprehensive fix...")
                # Fix radicals first
                rdkit_mol = self.fix_radicals(rdkit_mol)
                # Then apply comprehensive fixes
                rdkit_mol = self.fix_molecule_issues(rdkit_mol)
                # Double-check radicals are gone
                rdkit_mol = self.fix_radicals(rdkit_mol)
                
                # Verify no radicals remain
                remaining_radicals = sum(atom.GetNumRadicalElectrons() for atom in rdkit_mol.GetAtoms())
                if remaining_radicals > 0:
                    print(f"Warning: {remaining_radicals} radical electrons still present. Forcing removal...")
                    # Force remove all radicals
                    for atom in rdkit_mol.GetAtoms():
                        atom.SetNumRadicalElectrons(0)
                
                molecule_fixed = True
                print("Molecule issues fixed. Proceeding with OpenFF molecule creation.")
                # Optionally save the fixed SDF back to disk
                if save_fixed_sdf:
                    writer = Chem.SDWriter(sdf_file)
                    writer.write(rdkit_mol)
                    writer.close()
                    print(f"Fixed SDF saved back to {sdf_file}")
        
        # Step 2: Convert to OpenFF Molecule, preserving explicit atoms
        # Try multiple approaches if the first fails
        off_mol = None
        last_error = None
        
        # Approach 1: Standard conversion
        try:
            off_mol = Molecule.from_rdkit(
                rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        except RadicalsNotSupportedError as e:
            last_error = e
            if not molecule_fixed:
                # Try comprehensive fix
                print("Warning: Radicals detected. Attempting comprehensive molecule fix...")
                rdkit_mol = self.fix_molecule_issues(rdkit_mol)
                molecule_fixed = True
                # Retry after fixing
                try:
                    off_mol = Molecule.from_rdkit(
                        rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
                    )
                    print("Successfully fixed molecule issues and created OpenFF molecule.")
                except Exception as e2:
                    last_error = e2
                    # Try one more time with just radical fixing
                    print("Warning: Comprehensive fix failed. Trying simple radical fix...")
                    rdkit_mol = self.fix_radicals(rdkit_mol)
                    try:
                        off_mol = Molecule.from_rdkit(
                            rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
                        )
                        print("Successfully fixed radicals and created OpenFF molecule.")
                    except Exception as e3:
                        last_error = e3
        except Exception as e:
            # Catch any other errors
            last_error = e
            if not molecule_fixed:
                # Try fixing
                print(f"Warning: Error creating OpenFF molecule ({type(e).__name__}). Attempting fix...")
                rdkit_mol = self.fix_molecule_issues(rdkit_mol)
                try:
        off_mol = Molecule.from_rdkit(
            rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
                    print("Successfully fixed molecule and created OpenFF molecule.")
                except Exception as e2:
                    last_error = e2
        
        # If we still don't have a molecule, raise an error - user must fix the file manually
        if off_mol is None:
            # Provide detailed diagnostics
            num_atoms = rdkit_mol.GetNumAtoms()
            num_bonds = rdkit_mol.GetNumBonds()
            radicals = sum(atom.GetNumRadicalElectrons() for atom in rdkit_mol.GetAtoms())
            
            error_msg = (
                f"Failed to create OpenFF molecule from SDF file: {sdf_file}\n"
                f"All automatic fixing attempts failed. Please manually fix the SDF/XYZ file.\n\n"
                f"Diagnostics:\n"
                f"  - Number of atoms: {num_atoms}\n"
                f"  - Number of bonds: {num_bonds}\n"
                f"  - Total radical electrons: {radicals}\n"
                f"  - Last error: {last_error}\n\n"
                f"Possible issues:\n"
                f"  1. Incorrect bond orders or valence in the starting XYZ/SDF file.\n"
                f"  2. Aromaticity issues that RDKit/OpenFF cannot resolve automatically.\n"
                f"  3. Unsupported radical species.\n\n"
                f"Suggestions:\n"
                f"  - Try deleting the SDF file '{sdf_file}' and running again to recreate it from the XYZ file.\n"
                f"  - Manually inspect '{sdf_file}' for correctness.\n"
                f"  - Try generating the SDF with a different tool or method.\n"
                f"  - Ensure the input XYZ file has chemically sensible connectivity.\n"
                f"  - Consider using a different starting geometry if available.\n"
                f"  - Use a molecular editor to fix bond orders and remove radicals manually."
            )
            raise RuntimeError(error_msg) from last_error
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

