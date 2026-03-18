"""
Tests for modules/wrap.py
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip tests if wrap module dependencies are not available
try:
    from modules.wrap import Wrapper, _read_scattering_dat
    WRAP_AVAILABLE = True
except ImportError:
    WRAP_AVAILABLE = False


@pytest.mark.skipif(not WRAP_AVAILABLE, reason="wrap module dependencies not available")
class TestWrapper:
    """Test Wrapper class"""
    
    def setup_method(self):
        self.w = Wrapper()
    
    def test_init(self):
        """Test Wrapper initialization"""
        w = Wrapper()
        assert w is not None
    
    @pytest.mark.skipif(
        not os.path.exists("xyz/start.xyz"),
        reason="Test XYZ file not found"
    )
    def test_run_xyz_openff_mm_params_structure(self):
        """Test that run_xyz_openff_mm_params returns proper structure"""
        # This is a complex test that would require actual forcefield files
        # Just test the structure exists
        w = Wrapper()
        assert hasattr(w, 'run_xyz_openff_mm_params')
        assert callable(w.run_xyz_openff_mm_params)
    
    def test_run_method_exists(self):
        """Test that run method exists"""
        w = Wrapper()
        assert hasattr(w, 'run')
        assert callable(w.run)

    def test_read_scattering_dat_two_columns(self, tmp_path):
        """Two-column dat files should be parsed as q and I(q)."""
        dat_path = tmp_path / "two_col.dat"
        q = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        iq = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        np.savetxt(dat_path, np.column_stack((q, iq)))

        q_read, iq_read = _read_scattering_dat(str(dat_path))

        assert np.allclose(q_read, q)
        assert np.allclose(iq_read, iq)

    def test_read_scattering_dat_single_column(self, tmp_path):
        """Single-column dat files should be treated as I(q)-only."""
        dat_path = tmp_path / "one_col.dat"
        iq = np.array([5.0, 6.0, 7.0], dtype=np.float64)
        np.savetxt(dat_path, iq)

        q_read, iq_read = _read_scattering_dat(str(dat_path))

        assert np.allclose(q_read, np.arange(iq.size, dtype=np.float64))
        assert np.allclose(iq_read, iq)
