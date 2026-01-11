import numpy as np
from scipy.constants import Boltzmann, speed_of_light

######
class Sample:
    """methods to sample molecular structures"""

    def __init__(self):
        pass

    def generate_boltzmann_displacement(self, modes, freqs_cm1, T=300):
        """
        Sample a classical thermal displacement from normal modes (Å).

        Parameters:
            freqs_cm1: (N_modes,) Vibrational frequencies in cm⁻¹
            modes: (N_modes, N_atoms, 3) Normal modes in Å
            T: Temperature in K

        Returns:
            displacement: (N_atoms, 3) displacement in Å
        """
        # Constants in atomic units
        kB_au = 3.1668114e-6  # Hartree / K
        cm1_to_hartree = 4.556335e-6

        # Convert frequencies to Hartree
        energy_hartree = freqs_cm1 * cm1_to_hartree  # shape (N_modes,)
        
        # Classical standard deviations (Bohr)
        std_bohr = np.sqrt(kB_au * T) / energy_hartree  # shape (N_modes,)
        bohr_to_ang = 0.529177210903
        std_ang = std_bohr * bohr_to_ang

        # Sample random coefficients
        coeffs = np.random.normal(0.0, std_ang)  # shape (N_modes,)

        # Weighted sum over modes
        displacement = np.tensordot(coeffs, modes, axes=(0, 0))  # shape (N_atoms, 3)

        return displacement


### End Sample class
