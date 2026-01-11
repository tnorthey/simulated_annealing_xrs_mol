import numpy as np
from pyscf import gto, scf, hessian
from pyscf.hessian import thermo

######
class Pyscf_wrapper:
    """PySCF functions"""

    def __init__(self):
        pass

    def xyz_calc_modes(self, xyz_filename, save_to_npy=False, basis='sto-3g'):
        # === Build PySCF molecule ===
        mol = gto.Mole()
        mol.atom = xyz_filename
        mol.build()
        mol.basis = basis  # Use "def2-SVP" or similar for better accuracy
        mol.unit = "Angstrom"
        mol.build()
        
        # === Run SCF calculation ===
        mf = scf.RHF(mol)
        mf.kernel()
        
        # === Compute Hessian and run frequency analysis ===
        hess = hessian.RHF(mf).kernel()
        results = thermo.harmonic_analysis(mol, hess)
        frequencies_cm1 = results["freq_wavenumber"]
        mode_vectors = results["norm_mode"]  # normal modes
        
        if save_to_npy:
           # Save to .npy files
           np.save("nm/modes.npy", mode_vectors)
           np.save("nm/frequencies_cm1.npy", frequencies_cm1)
        
        return mode_vectors, frequencies_cm1

