"""
Tests for modules/mol.py
"""
import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.mol import Xyz


class TestPeriodicTable:
    """Test periodic_table function"""
    
    def setup_method(self):
        self.xyz = Xyz()
    
    def test_common_elements(self):
        """Test common elements"""
        assert self.xyz.periodic_table("H") == 1
        assert self.xyz.periodic_table("C") == 6
        assert self.xyz.periodic_table("N") == 7
        assert self.xyz.periodic_table("O") == 8
    
    def test_heavy_elements(self):
        """Test heavier elements"""
        assert self.xyz.periodic_table("Fe") == 26
        assert self.xyz.periodic_table("Cu") == 29
        assert self.xyz.periodic_table("Zn") == 30
    
    def test_invalid_element(self):
        """Test invalid element returns None"""
        assert self.xyz.periodic_table("Xx") is None
        assert self.xyz.periodic_table("") is None


class TestReadWriteXYZ:
    """Test XYZ file reading and writing"""
    
    def setup_method(self):
        self.xyz = Xyz()
    
    def test_read_xyz(self, sample_xyz_file):
        """Test reading XYZ file"""
        natoms, comment, atoms, coords = self.xyz.read_xyz(sample_xyz_file)
        
        assert natoms == 2
        assert comment.strip() == "Test molecule"
        assert len(atoms) == 2
        assert atoms[0] == "H"
        assert atoms[1] == "O"
        assert coords.shape == (2, 3)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords[1], [0.0, 0.0, 0.96])
    
    def test_write_xyz(self, sample_xyz_data):
        """Test writing XYZ file"""
        atoms, coords = sample_xyz_data
        comment = "Test write"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            self.xyz.write_xyz(temp_file, comment, atoms, coords)
            
            # Read it back
            natoms, read_comment, read_atoms, read_coords = self.xyz.read_xyz(temp_file)
            
            assert natoms == 2
            assert read_comment.strip() == comment
            assert len(read_atoms) == 2
            assert read_atoms[0] == "H"
            assert read_atoms[1] == "O"
            np.testing.assert_array_almost_equal(read_coords, coords)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestDistancesArray:
    """Test distances_array function"""
    
    def setup_method(self):
        self.xyz = Xyz()
    
    def test_distances_simple(self):
        """Test distance calculation for simple case"""
        # Two points 1 unit apart on x-axis
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dist = self.xyz.distances_array(coords)
        
        assert dist.shape == (2, 2)
        assert dist[0, 0] == 0.0
        assert dist[1, 1] == 0.0
        assert abs(dist[0, 1] - 1.0) < 1e-10
        assert abs(dist[1, 0] - 1.0) < 1e-10  # Symmetric
    
    def test_distances_square(self):
        """Test distances for square configuration"""
        # Square in xy-plane
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        dist = self.xyz.distances_array(coords)
        
        assert dist.shape == (4, 4)
        # Diagonal should be sqrt(2)
        np.testing.assert_almost_equal(dist[0, 2], np.sqrt(2), decimal=6)
        np.testing.assert_almost_equal(dist[1, 3], np.sqrt(2), decimal=6)
        # Symmetry
        np.testing.assert_almost_equal(dist[0, 2], dist[2, 0], decimal=10)


class TestRMSD:
    """Test RMSD calculations"""
    
    def setup_method(self):
        self.xyz = Xyz()
    
    def test_rmsd_atoms_identical(self):
        """Test RMSD for identical structures"""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        indices = [0, 1]
        
        rmsd = self.xyz.rmsd_atoms(coords1, coords2, indices)
        assert abs(rmsd) < 1e-10
    
    def test_rmsd_atoms_shifted(self):
        """Test RMSD for shifted structures"""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # Shifted by 1
        indices = [0, 1]
        
        rmsd = self.xyz.rmsd_atoms(coords1, coords2, indices)
        np.testing.assert_almost_equal(rmsd, 1.0, decimal=6)
    
    def test_rmsd_kabsch_identical(self):
        """Test Kabsch RMSD for identical structures"""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        indices = [0, 1, 2]
        
        rmsd, rot = self.xyz.rmsd_kabsch(coords1, coords2, indices)
        assert rmsd < 1e-6


class TestMAPD:
    """Test MAPD calculations"""
    
    def setup_method(self):
        self.xyz = Xyz()
    
    def test_mapd_function_identical(self):
        """Test MAPD for identical structures"""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        indices = [0, 1, 2]
        
        mapd = self.xyz.mapd_function(coords1, coords2, indices)
        assert mapd < 1e-6
    
    def test_mapd_distances_identical(self):
        """Test MAPD for identical distance arrays"""
        rij = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        rij_ = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        
        mapd = self.xyz.mapd_distances(rij, rij_, bond_print=False)
        assert mapd < 1e-6


# Note: Dihedral and angle calculation tests have been moved to tests/test_analysis.py
