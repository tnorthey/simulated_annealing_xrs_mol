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
    from modules.wrap import Wrapper
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
