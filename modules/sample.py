import numpy as np

######
class Sample:
    """methods to sample molecular structures"""

    def __init__(self):
        pass

    # Atomic masses (standard atomic weights, in atomic mass units).
    # Only used for Boltzmann sampling; keep local to avoid new dependencies.
    _ATOMIC_MASS_AMU = {
        1: 1.00784,   # H
        2: 4.002602,  # He
        3: 6.94,      # Li
        4: 9.0121831, # Be
        5: 10.81,     # B
        6: 12.011,    # C
        7: 14.007,    # N
        8: 15.999,    # O
        9: 18.998403163, # F
        10: 20.1797,  # Ne
        11: 22.98976928, # Na
        12: 24.305,   # Mg
        13: 26.9815385, # Al
        14: 28.085,   # Si
        15: 30.973761998, # P
        16: 32.06,    # S
        17: 35.45,    # Cl
        18: 39.948,   # Ar
        19: 39.0983,  # K
        20: 40.078,   # Ca
        35: 79.904,   # Br
        53: 126.90447 # I
    }

    @classmethod
    def _atomic_masses_kg(cls, atomic_numbers):
        """Map atomic numbers to masses in kg (fallback: mass number ~= Z)."""
        amu_to_kg = 1.66053906660e-27
        masses_amu = np.array(
            [cls._ATOMIC_MASS_AMU.get(int(z), float(int(z))) for z in atomic_numbers],
            dtype=np.float64,
        )
        return masses_amu * amu_to_kg

    def generate_boltzmann_displacement(self, modes, freqs_cm1, T=300.0, atomic_numbers=None):
        """
        Sample a classical thermal displacement from normal modes (Å).

        Important: to get physically reasonable amplitudes, the sampling must include
        a mode effective mass. Without masses, low-frequency modes produce unrealistically
        large (multi-Å) displacements.

        Parameters:
            freqs_cm1: (N_modes,) Vibrational frequencies in cm⁻¹
            modes: (N_modes, N_atoms, 3) Normal mode direction vectors (typically dimensionless)
            T: Temperature in K
            atomic_numbers: (N_atoms,) Atomic numbers (needed for correct scaling)

        Returns:
            displacement: (N_atoms, 3) displacement in Å
        """
        if atomic_numbers is None:
            raise ValueError(
                "generate_boltzmann_displacement requires atomic_numbers to avoid "
                "unphysically large displacements (missing mode effective masses)."
            )

        modes = np.asarray(modes, dtype=np.float64)
        freqs_cm1 = np.asarray(freqs_cm1, dtype=np.float64)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)

        if modes.ndim != 3:
            raise ValueError(f"modes must be (N_modes, N_atoms, 3); got shape {modes.shape}")
        if freqs_cm1.ndim != 1 or freqs_cm1.shape[0] != modes.shape[0]:
            raise ValueError(
                f"freqs_cm1 must be (N_modes,) matching modes; got {freqs_cm1.shape} vs {modes.shape[0]}"
            )
        if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != modes.shape[1]:
            raise ValueError(
                f"atomic_numbers must be (N_atoms,) matching modes; got {atomic_numbers.shape} vs {modes.shape[1]}"
            )

        # Physical constants (SI)
        kB = 1.380649e-23  # J/K
        c_cm_s = 2.99792458e10  # speed of light in cm/s
        angstrom_per_meter = 1e10

        # Angular frequencies in rad/s
        omega = 2.0 * np.pi * c_cm_s * freqs_cm1  # (N_modes,)

        # Normalize mode direction vectors (avoid relying on upstream normalization).
        flat = modes.reshape(modes.shape[0], -1)
        norms = np.linalg.norm(flat, axis=1)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        modes_unit = modes / safe_norms[:, None, None]

        # Mode effective mass μ_i = Σ_a m_a * |e_{i,a}|^2  (if e is dimensionless direction vector)
        masses_kg = self._atomic_masses_kg(atomic_numbers)  # (N_atoms,)
        per_atom_weight = np.sum(modes_unit * modes_unit, axis=2)  # (N_modes, N_atoms)
        mu = np.sum(per_atom_weight * masses_kg[None, :], axis=1)  # (N_modes,)

        # Classical coordinate std for each mode: σ_q = sqrt(kB*T / (μ * ω^2))
        denom = mu * omega * omega
        # Avoid divide-by-zero (shouldn't happen for real vibrational modes).
        std_q_m = np.sqrt(np.where(denom > 0.0, (kB * float(T)) / denom, 0.0))  # meters

        coeffs_m = np.random.normal(0.0, std_q_m, size=std_q_m.shape[0])  # meters
        displacement_m = np.tensordot(coeffs_m, modes_unit, axes=(0, 0))  # (N_atoms, 3) meters
        displacement_ang = displacement_m * angstrom_per_meter
        return displacement_ang


### End Sample class
