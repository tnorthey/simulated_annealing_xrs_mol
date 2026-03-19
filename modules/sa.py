from numba import njit
import numpy as np
from numpy.typing import NDArray
from timeit import default_timer

from modules.compute_backend import get_backend, to_numpy


#############################
class Annealing:
    """Simulated annealing functions"""

    def __init__(self):
        # Populated by GPU batched runs (gpu_chains > 1).
        self.last_chain_results = None
        # Cache for static host->device array conversions.
        self._gpu_array_cache = {}

    def simulated_annealing_modes_ho(
        self,
        starting_xyz: NDArray,
        displacements: NDArray,
        mode_indices: NDArray,
        target_function: NDArray,
        reference_iam: NDArray,
        qvector: NDArray,
        th: NDArray,
        ph: NDArray,
        compton: NDArray,
        atomic_total: NDArray,
        pre_molecular: NDArray,
        step_size_array: NDArray,
        bond_param_array: NDArray,
        angle_param_array: NDArray,
        torsion_param_array: NDArray,
        starting_temp=0.2,
        nsteps=10000,
        inelastic=True,
        pcd_mode=False,
        ewald_mode=False,
        bonds_bool=True,
        angles_bool=True,
        torsions_bool=True,
        f_start=1e10,
        f_xray_start=1e10,
        predicted_start=0,
        tuning_ratio_target=1,
        c_tuning_initial=1,
        verbose: bool = False,
        backend: str = "cpu",
        gpu_emulation: bool = False,
        gpu_chains: int = 1,
        keep_on_device: bool = False,
    ):
        """simulated annealing minimisation to target_function"""
        # Clear stale results from prior invocations.
        self.last_chain_results = None
        ######## READ BOND/ANGLE PARAMS #######
        # Cache derived arrays on self so their id()s stay stable across
        # calls, allowing _as_backend_array to reuse GPU copies.
        _param_key = (
            id(bond_param_array), bond_param_array.shape,
            id(angle_param_array), angle_param_array.shape,
            id(torsion_param_array), torsion_param_array.shape,
        )
        if getattr(self, '_param_cache_key', None) == _param_key:
            (
                bond_atom1_idx_arr, bond_atom2_idx_arr, r0_arr, k_arr,
                angle_atom1_idx_arr, angle_atom2_idx_arr, angle_atom3_idx_arr,
                theta0_arr, k_theta_arr,
                torsion_atom1_idx_arr, torsion_atom2_idx_arr,
                torsion_atom3_idx_arr, torsion_atom4_idx_arr,
                delta0_arr, k_delta_arr,
            ) = self._param_arrays
            nbonds, nangles, ntorsions = (
                len(r0_arr), len(theta0_arr), len(delta0_arr),
            )
        else:
            # Bonds
            bond_atom1_idx_arr = bond_param_array[:, 0].astype(int)
            bond_atom2_idx_arr = bond_param_array[:, 1].astype(int)
            r0_arr = bond_param_array[:, 2]
            k_arr = bond_param_array[:, 3]
            nbonds = len(r0_arr)
            # Angles
            angle_atom1_idx_arr = angle_param_array[:, 0].astype(int)
            angle_atom2_idx_arr = angle_param_array[:, 1].astype(int)
            angle_atom3_idx_arr = angle_param_array[:, 2].astype(int)
            theta0_arr = angle_param_array[:, 3]
            k_theta_arr = angle_param_array[:, 4]
            nangles = len(theta0_arr)
            # Torsions
            torsion_atom1_idx_arr = torsion_param_array[:, 0].astype(int)
            torsion_atom2_idx_arr = torsion_param_array[:, 1].astype(int)
            torsion_atom3_idx_arr = torsion_param_array[:, 2].astype(int)
            torsion_atom4_idx_arr = torsion_param_array[:, 3].astype(int)
            delta0_arr = torsion_param_array[:, 4]
            k_delta_arr = torsion_param_array[:, 5]
            ntorsions = len(delta0_arr)
            self._param_cache_key = _param_key
            self._param_arrays = (
                bond_atom1_idx_arr, bond_atom2_idx_arr, r0_arr, k_arr,
                angle_atom1_idx_arr, angle_atom2_idx_arr, angle_atom3_idx_arr,
                theta0_arr, k_theta_arr,
                torsion_atom1_idx_arr, torsion_atom2_idx_arr,
                torsion_atom3_idx_arr, torsion_atom4_idx_arr,
                delta0_arr, k_delta_arr,
            )
        ##=#=#=# DEFINITIONS #=#=#=##
        natoms = starting_xyz.shape[0]  # number of atoms
        c_tuning = c_tuning_initial  # initialise C_tuning
        gpu_chains = max(1, int(gpu_chains))

        def _as_backend_array(xp, arr, dtype=None):
            """Convert to backend array, reusing cached static transfers when possible."""
            xp_ndarray = getattr(xp, "ndarray", None)
            if xp_ndarray is not None and isinstance(arr, xp_ndarray):
                if dtype is None:
                    return arr
                return arr.astype(dtype, copy=False)
            key = (id(xp), id(arr), str(dtype), getattr(arr, "shape", None))
            cached = self._gpu_array_cache.get(key)
            if cached is not None and cached.shape == getattr(arr, "shape", None):
                return cached
            out = xp.asarray(arr, dtype=dtype)
            self._gpu_array_cache[key] = out
            return out
        # nmodes = displacements.shape[0]  # number of displacement vectors
        # nmode_indices = len(mode_indices)
        # print((nmodes, nmode_indices))
        # modes = list(range(nmodes))  # all modes
        ## q-vector, atomic, and pre-molecular IAM contributions ##
        # print(qvector)
        qmin, qmax, qlen = qvector[0], qvector[-1], len(qvector)
        tmin, tmax, tlen = th[0], th[-1], len(th)
        pmin, pmax, plen = ph[0], ph[-1], len(ph)
        ##=#=#=# END DEFINITIONS #=#=#=#
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
        # Cache the shifted target_function and abs so the same numpy
        # objects persist across calls → _as_backend_array cache hits.
        _tf_key = (id(target_function), inelastic, pcd_mode,
                   id(atomic_total), id(compton), id(reference_iam))
        _tf_cached = getattr(self, '_tf_cache_key', None) == _tf_key
        if _tf_cached:
            abs_target_function = self._abs_target_function
        else:
            abs_target_function = np.abs(target_function)
        ### define qx, qy, qz for Ewald mode (CPU path)
        if ewald_mode:
            r_grid, th_grid, ph_grid = np.meshgrid(qvector, th, ph, indexing="ij")
            # Convert spherical coordinates to Cartesian coordinates
            qx = r_grid * np.sin(th_grid) * np.cos(ph_grid)
            qy = r_grid * np.sin(th_grid) * np.sin(ph_grid)
            qz = r_grid * np.cos(th_grid)
        # Shift target function to remove constant IAM terms outside the SA loop
        if _tf_cached:
            target_function = self._target_function_modified
        else:
            if inelastic:
                iam_offset = atomic_total + compton
            else:
                iam_offset = atomic_total
            if pcd_mode:
                pcd_offset = 100.0 * (iam_offset / reference_iam - 1.0)
                target_function = target_function - pcd_offset
            else:
                target_function = target_function - iam_offset
            self._tf_cache_key = _tf_key
            self._abs_target_function = abs_target_function
            self._target_function_modified = target_function
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
        # Ensure predicted_start has the right shape/type (avoid int sentinel inside njit)
        if isinstance(predicted_start, (int, float)) and predicted_start == 0:
            if ewald_mode:
                predicted_start = np.zeros((qlen, tlen, plen), dtype=np.float64)
            else:
                predicted_start = np.zeros(qlen, dtype=np.float64)

        @njit(nogil=True, fastmath=False)  # numba decorator to compile to machine code
        def run_annealing(nsteps):

            ##=#=#=# INITIATE LOOP VARIABLES #=#=#=#=#
            xyz = starting_xyz.copy()
            xyz_trial = np.empty_like(xyz)
            xyz_best = xyz.copy()
            f_best, f_xray_best = f_start, f_xray_start  # initialise on restart
            predicted_best = predicted_start.copy()  # initialise on restart
            f = 1e9  # high initial value so 1st step will be accepted
            c = 0  # count accepted steps
            mdisp = displacements
            n_mode_indices = len(mode_indices)
            (
                total_bonding_contrib,
                total_angular_contrib,
                total_torsional_contrib,
                total_xray_contrib,
            ) = (0, 0, 0, 0)
            ##=#=#=# END INITIATE LOOP VARIABLES #=#=#
            # Pre-compute inverse nsteps and constants to avoid division in loop
            inv_nsteps = 1.0 / nsteps
            HALF = 0.5
            # Reuse buffers to avoid per-step allocations
            summed_displacement = np.zeros((natoms, 3), dtype=np.float64)
            if not ewald_mode:
                molecular = np.zeros(qlen, dtype=np.float64)
                iam = np.empty(qlen, dtype=np.float64)
                predicted_function_ = np.empty(qlen, dtype=np.float64)

            for i in range(nsteps):

                ##=#=#=#=# TEMPERATURE #=#=#=#=#=#=#=#=##
                tmp = 1.0 - i * inv_nsteps  # this is prop. to how far the molecule moves
                temp = starting_temp * tmp  # this is the probability of going uphill
                ##=#=#=# END TEMPERATURE #=#=#=#=#=#=#=##

                ##=#=#=# DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##
                # this is faster in numba than the vectorised version...
                # numba likes loops (not vectors apparently)
                summed_displacement[:, :] = 0.0
                for mi in range(n_mode_indices):
                    n = mode_indices[mi]
                    scale = step_size_array[n] * tmp * (2.0 * np.random.random() - 1.0)
                    # manual fused multiply-add into buffer
                    for a in range(natoms):
                        summed_displacement[a, 0] += mdisp[n, a, 0] * scale
                        summed_displacement[a, 1] += mdisp[n, a, 1] * scale
                        summed_displacement[a, 2] += mdisp[n, a, 2] * scale
                for a in range(natoms):
                    xyz_trial[a, 0] = xyz[a, 0] + summed_displacement[a, 0]
                    xyz_trial[a, 1] = xyz[a, 1] + summed_displacement[a, 1]
                    xyz_trial[a, 2] = xyz[a, 2] + summed_displacement[a, 2]
                ##=#=#=# END DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##

                ##=#=#=# TEMPERATURE ACCEPTANCE CRITERIA #=#=#=##
                if temp > np.random.random():
                    c += 1  # count acceptances
                    xyz, xyz_trial = xyz_trial, xyz  # accept trial (swap buffers)
                    continue  # go to next step in for loop
                ##=#=#=# END TEMPERATURE ACCEPTANCE CRITERIA #=#=#=##

                ##=#=#=# IAM CALCULATION #=#=#=##
                if (
                    ewald_mode
                ):  # x-ray signal in Ewald sphere, q = (q_radial, q_theta, q_phi)
                    # molecular
                    molecular = np.zeros((qlen, tlen, plen))  # total molecular factor
                    k = 0  # begin counter
                    for n in range(natoms - 1):
                        for m in range(n + 1, natoms):  # j > i
                            fnm = pre_molecular[k, :, :, :]
                            k += 1  # count iterations
                            xnm = xyz[n, 0] - xyz[m, 0]
                            ynm = xyz[n, 1] - xyz[m, 1]
                            znm = xyz[n, 2] - xyz[m, 2]
                            molecular += (
                                2 * fnm * np.cos((qx * xnm + qy * ynm + qz * znm))
                            )
                    ### end ewald_mode
                else:  # assumed to be isotropic 1D signal
                    molecular[:] = 0.0  # total molecular factor
                    k = 0
                    for ii in range(natoms):
                        for jj in range(ii + 1, natoms):  # j > i
                            # Manual distance calculation (faster than LA.norm in numba)
                            dx = xyz_trial[ii, 0] - xyz_trial[jj, 0]
                            dy = xyz_trial[ii, 1] - xyz_trial[jj, 1]
                            dz = xyz_trial[ii, 2] - xyz_trial[jj, 2]
                            r = np.sqrt(dx * dx + dy * dy + dz * dz)
                            for qi in range(qlen):
                                qd = qvector[qi] * r
                                molecular[qi] += (
                                    2.0
                                    * pre_molecular[k, qi]
                                    * np.sin(qd)
                                    / qd
                                )
                            k += 1
                    for qi in range(qlen):
                        iam[qi] = molecular[qi]
                ##=#=#=# END IAM CALCULATION #=#=#=##

                ##=#=#=# PCD & DSIGNAL CALCULATIONS #=#=#=##
                if pcd_mode:
                    inv_qlen = 1.0 / qlen
                    sse = 0.0
                    for qi in range(qlen):
                        # Objective compares *molecular-only* contribution because
                        # `target_function` was pre-shifted outside the SA loop.
                        pred_mol = 100.0 * (iam[qi] / reference_iam[qi])
                        diff = pred_mol - target_function[qi]
                        sse += diff * diff
                        # For output, store the full PCD curve (incl. constant atomic/compton term)
                        if inelastic:
                            offset = atomic_total[qi] + compton[qi]
                        else:
                            offset = atomic_total[qi]
                        predicted_function_[qi] = pred_mol + 100.0 * (
                            offset / reference_iam[qi] - 1.0
                        )
                    xray_contrib = sse * inv_qlen
                else:
                    ### x-ray part of objective function
                    ### TO DO: depends on ewald_mode ...
                    if ewald_mode:
                        n = qlen * tlen * plen
                    else:
                        n = qlen
                    # Pre-compute inverse n to avoid division
                    inv_n = 1.0 / n
                    sse = 0.0
                    for qi in range(qlen):
                        # Objective compares *molecular-only* contribution because
                        # `target_function` was pre-shifted outside the SA loop.
                        pred_mol = iam[qi]
                        diff = pred_mol - target_function[qi]
                        sse += (diff * diff) / abs_target_function[qi]
                        # For output, store the full IAM curve (incl. constant atomic/compton term)
                        if inelastic:
                            predicted_function_[qi] = pred_mol + atomic_total[qi] + compton[qi]
                        else:
                            predicted_function_[qi] = pred_mol + atomic_total[qi]
                    xray_contrib = sse * inv_n

                ### harmonic oscillator part of f
                # somehow this is faster in numba than the vectorised version
                # New method: read from bond_param_array (OpenMM params)
                bonding_contrib = 0.0
                if bonds_bool:
                    for i_bond in range(nbonds):
                        # Manual distance calculation (faster than LA.norm in numba)
                        idx1 = bond_atom1_idx_arr[i_bond]
                        idx2 = bond_atom2_idx_arr[i_bond]
                        dx = xyz_trial[idx1, 0] - xyz_trial[idx2, 0]
                        dy = xyz_trial[idx1, 1] - xyz_trial[idx2, 1]
                        dz = xyz_trial[idx1, 2] - xyz_trial[idx2, 2]
                        r = np.sqrt(dx * dx + dy * dy + dz * dz)
                        bonding_contrib += k_arr[i_bond] * HALF * (r - r0_arr[i_bond]) ** 2

                angular_contrib = 0.0
                if angles_bool:
                    for i_ang in range(nangles):
                        """Return angle ABC (at B) given three positions A, B, C."""
                        """Faster to not call outside function. And works better with numba."""
                        # Direct indexing to avoid repeated slicing
                        idx1 = angle_atom1_idx_arr[i_ang]
                        idx2 = angle_atom2_idx_arr[i_ang]
                        idx3 = angle_atom3_idx_arr[i_ang]
                        BA_x = xyz_trial[idx1, 0] - xyz_trial[idx2, 0]
                        BA_y = xyz_trial[idx1, 1] - xyz_trial[idx2, 1]
                        BA_z = xyz_trial[idx1, 2] - xyz_trial[idx2, 2]
                        BC_x = xyz_trial[idx3, 0] - xyz_trial[idx2, 0]
                        BC_y = xyz_trial[idx3, 1] - xyz_trial[idx2, 1]
                        BC_z = xyz_trial[idx3, 2] - xyz_trial[idx2, 2]
                        norm_BA = np.sqrt(BA_x * BA_x + BA_y * BA_y + BA_z * BA_z)
                        norm_BC = np.sqrt(BC_x * BC_x + BC_y * BC_y + BC_z * BC_z)
                        inv_norm_BA_BC = 1.0 / (norm_BA * norm_BC)
                        cos_theta = (BA_x * BC_x + BA_y * BC_y + BA_z * BC_z) * inv_norm_BA_BC
                        cos_theta = min(
                            1.0, max(-1.0, cos_theta)
                        )  # stops cosine being out of range [-1, 1] due to slight numerical flucuations
                        theta = np.arccos(cos_theta)

                        angular_contrib += (
                            k_theta_arr[i_ang] * HALF * (theta - theta0_arr[i_ang]) ** 2
                        )

                torsion_contrib = 0.0
                if torsions_bool:
                    for i_tors in range(ntorsions):
                        # calculate the dihedrals
                        idx1 = torsion_atom1_idx_arr[i_tors]
                        idx2 = torsion_atom2_idx_arr[i_tors]
                        idx3 = torsion_atom3_idx_arr[i_tors]
                        idx4 = torsion_atom4_idx_arr[i_tors]
                        # Direct indexing instead of slicing
                        p0 = xyz_trial[idx1, :]
                        p1 = xyz_trial[idx2, :]
                        p2 = xyz_trial[idx3, :]
                        p3 = xyz_trial[idx4, :]

                        b0 = -1.0 * (p1 - p0)
                        b1 = p2 - p1
                        b2 = p3 - p2

                        # normalize b1 so that it does not influence magnitude of vector
                        # projections that come next
                        # Manual norm calculation (faster than np.linalg.norm in numba)
                        b1_norm = np.sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2])
                        b1 /= b1_norm

                        # vector projections
                        # v = projection of b0 onto plane perpendicular to b1
                        #   = b0 minus component that aligns with b1
                        # w = projection of b2 onto plane perpendicular to b1
                        #   = b2 minus component that aligns with b1
                        v = b0 - np.dot(b0, b1) * b1
                        w = b2 - np.dot(b2, b1) * b1

                        # angle between v and w in a plane is the torsion angle
                        # v and w may not be normalized but that's fine since tan is y/x
                        x = np.dot(v, w)
                        y = np.dot(np.cross(b1, v), w)
                        torsion = np.arctan2(y, x)

                        torsion_contrib += k_delta_arr[i_tors] * (
                            1 + np.cos(torsion - delta0_arr[i_tors])
                        )

                ### combine x-ray and bonding, angular contributions
                f_ = xray_contrib + c_tuning * (
                    bonding_contrib + angular_contrib + torsion_contrib
                )
                ##=#=#=# END PCD & DSIGNAL CALCULATIONS #=#=#=##

                ##=#=#=# ACCEPTANCE CRITERIA #=#=#=##
                # Use multiplication instead of division (faster)
                if f_ < f * 0.999:
                    c += 1  # count acceptances
                    f = f_
                    xyz, xyz_trial = xyz_trial, xyz  # accept trial (swap buffers)
                    if f < f_best:
                        # store values corresponding to f_best
                        f_best = f
                        xyz_best[:, :] = xyz[:, :]
                        predicted_best = predicted_function_.copy()
                        f_xray_best = xray_contrib
                    total_bonding_contrib += c_tuning * bonding_contrib
                    total_angular_contrib += c_tuning * angular_contrib
                    total_torsional_contrib += c_tuning * torsion_contrib
                    total_xray_contrib += xray_contrib
                ##=#=#=# END ACCEPTANCE CRITERIA #=#=#=##
            # print ratio of contributions to f
            priors_contrib = (
                total_bonding_contrib + total_angular_contrib + total_torsional_contrib
            )
            total_contrib = total_xray_contrib + priors_contrib
            if total_contrib > 0 and priors_contrib > 0:
                xray_ratio = total_xray_contrib / total_contrib
                bonding_ratio = total_bonding_contrib / total_contrib
                angular_ratio = total_angular_contrib / total_contrib
                torsional_ratio = total_torsional_contrib / total_contrib
                # readjust c_tuning
                c_tuning_adjusted = (
                    (1 - tuning_ratio_target)
                    * c_tuning
                    / (1 - total_xray_contrib / total_contrib)
                )
            else:
                (
                    xray_ratio,
                    bonding_ratio,
                    angular_ratio,
                    torsional_ratio,
                    c_tuning_adjusted,
                ) = (0, 0, 0, 0, 0)
            return (
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
                xray_ratio,
                bonding_ratio,
                angular_ratio,
                torsional_ratio,
                c,
                c_tuning_adjusted,
            )

        ### END run_annealing() function ###

        def run_annealing_gpu_batched(nsteps, xp, n_chains):
            """
            Run multiple independent SA chains in parallel on GPU and
            return the best chain outcome.
            """
            prep_start = default_timer()
            # Backend arrays with chain dimension
            xyz = xp.repeat(
                xp.asarray(starting_xyz, dtype=xp.float64)[xp.newaxis, :, :],
                n_chains,
                axis=0,
            )
            xyz_trial = xp.empty_like(xyz)
            xyz_best = xyz.copy()

            predicted_start_xp = xp.asarray(predicted_start, dtype=xp.float64)
            if predicted_start_xp.ndim == 1:
                predicted_best = xp.repeat(
                    predicted_start_xp[xp.newaxis, :], n_chains, axis=0
                )
            elif predicted_start_xp.ndim == 2 and predicted_start_xp.shape[0] == n_chains:
                predicted_best = predicted_start_xp.copy()
            else:
                predicted_best = xp.repeat(
                    predicted_start_xp.reshape(1, -1), n_chains, axis=0
                )

            f_best = xp.full(n_chains, float(f_start), dtype=xp.float64)
            f_xray_best = xp.full(n_chains, float(f_xray_start), dtype=xp.float64)
            f = xp.full(n_chains, 1e9, dtype=xp.float64)
            c = 0

            n_mode_indices = len(mode_indices)
            mdisp = _as_backend_array(xp, displacements, xp.float64)
            step_size_xp = _as_backend_array(xp, step_size_array, xp.float64)
            qvector_xp = _as_backend_array(xp, qvector, xp.float64)
            target_function_xp = _as_backend_array(xp, target_function, xp.float64)
            reference_iam_xp = _as_backend_array(xp, reference_iam, xp.float64)
            abs_target_function_xp = _as_backend_array(xp, abs_target_function, xp.float64)
            atomic_total_xp = _as_backend_array(xp, atomic_total, xp.float64)
            compton_xp = _as_backend_array(xp, compton, xp.float64)
            pre_molecular_xp = _as_backend_array(xp, pre_molecular, xp.float64)

            # Constraints arrays – keep on GPU for fancy indexing;
            # torsion indices stay numpy for scalar loop access.
            bond_atom1_idx = _as_backend_array(xp, bond_atom1_idx_arr, xp.int64)
            bond_atom2_idx = _as_backend_array(xp, bond_atom2_idx_arr, xp.int64)
            angle_atom1_idx = _as_backend_array(xp, angle_atom1_idx_arr, xp.int64)
            angle_atom2_idx = _as_backend_array(xp, angle_atom2_idx_arr, xp.int64)
            angle_atom3_idx = _as_backend_array(xp, angle_atom3_idx_arr, xp.int64)
            torsion_atom1_idx = torsion_atom1_idx_arr
            torsion_atom2_idx = torsion_atom2_idx_arr
            torsion_atom3_idx = torsion_atom3_idx_arr
            torsion_atom4_idx = torsion_atom4_idx_arr
            r0_xp = _as_backend_array(xp, r0_arr, xp.float64)
            k_xp = _as_backend_array(xp, k_arr, xp.float64)
            theta0_xp = _as_backend_array(xp, theta0_arr, xp.float64)
            k_theta_xp = _as_backend_array(xp, k_theta_arr, xp.float64)
            delta0_xp = _as_backend_array(xp, delta0_arr, xp.float64)
            k_delta_xp = _as_backend_array(xp, k_delta_arr, xp.float64)

            inv_nsteps = 1.0 / nsteps
            inv_qlen = 1.0 / qlen
            HALF = 0.5

            # Batched path currently supports isotropic mode only.
            if ewald_mode:
                raise NotImplementedError(
                    "gpu_chains > 1 is not implemented for ewald_mode yet."
                )

            # Pre-compute pair indices for vectorized IAM
            pair_i_np, pair_j_np = np.triu_indices(natoms, k=1)
            pair_i_xp = xp.asarray(pair_i_np)
            pair_j_xp = xp.asarray(pair_j_np)

            # Pre-extract mode data for vectorized displacement (single matmul)
            mode_indices_xp = xp.asarray(np.asarray(mode_indices))
            mode_disps_xp = mdisp[mode_indices_xp]
            mode_disps_flat = mode_disps_xp.reshape(n_mode_indices, -1)
            mode_steps_xp = step_size_xp[mode_indices_xp]

            # Torsion indices on GPU for vectorized dihedral calc
            if torsions_bool and ntorsions > 0:
                torsion_a1_xp = _as_backend_array(xp, torsion_atom1_idx_arr, xp.int64)
                torsion_a2_xp = _as_backend_array(xp, torsion_atom2_idx_arr, xp.int64)
                torsion_a3_xp = _as_backend_array(xp, torsion_atom3_idx_arr, xp.int64)
                torsion_a4_xp = _as_backend_array(xp, torsion_atom4_idx_arr, xp.int64)

            # GPU accumulators – no per-step GPU→CPU sync
            c_xp = xp.zeros(n_chains, dtype=xp.int64)
            total_bonding_xp = xp.zeros(n_chains, dtype=xp.float64)
            total_angular_xp = xp.zeros(n_chains, dtype=xp.float64)
            total_torsional_xp = xp.zeros(n_chains, dtype=xp.float64)
            total_xray_xp = xp.zeros(n_chains, dtype=xp.float64)

            if pcd_mode:
                if inelastic:
                    _pcd_offset = atomic_total_xp + compton_xp
                else:
                    _pcd_offset = atomic_total_xp
                _pcd_offset_correction = 100.0 * (
                    _pcd_offset[xp.newaxis, :] / reference_iam_xp[xp.newaxis, :] - 1.0
                )
            else:
                if inelastic:
                    _iam_offset = (
                        atomic_total_xp[xp.newaxis, :] + compton_xp[xp.newaxis, :]
                    )
                else:
                    _iam_offset = atomic_total_xp[xp.newaxis, :]

            prep_time_s = default_timer() - prep_start
            loop_start = default_timer()
            for i in range(nsteps):
                tmp = 1.0 - i * inv_nsteps
                temp = starting_temp * tmp

                # Vectorised displacement (one matmul replaces mode loop)
                rand = 2.0 * xp.random.random((n_chains, n_mode_indices)) - 1.0
                scales = mode_steps_xp[xp.newaxis, :] * tmp * rand
                summed_displacement = (scales @ mode_disps_flat).reshape(
                    n_chains, natoms, 3
                )
                xyz_trial = xyz + summed_displacement

                # Temperature acceptance – no GPU→CPU sync
                temp_accept = temp > xp.random.random(n_chains)
                xyz = xp.where(
                    temp_accept[:, xp.newaxis, xp.newaxis], xyz_trial, xyz
                )
                c_xp += temp_accept.astype(xp.int64)

                # Vectorised IAM (all atom pairs in one batch)
                dx = xyz_trial[:, pair_i_xp, 0] - xyz_trial[:, pair_j_xp, 0]
                dy = xyz_trial[:, pair_i_xp, 1] - xyz_trial[:, pair_j_xp, 1]
                dz = xyz_trial[:, pair_i_xp, 2] - xyz_trial[:, pair_j_xp, 2]
                r_pairs = xp.sqrt(dx * dx + dy * dy + dz * dz)
                qd = r_pairs[:, :, xp.newaxis] * qvector_xp[xp.newaxis, xp.newaxis, :]
                iam = xp.sum(
                    2.0 * pre_molecular_xp[xp.newaxis, :, :] * xp.sin(qd) / qd,
                    axis=1,
                )

                # Objective function (PCD or IAM)
                if pcd_mode:
                    pred_mol = 100.0 * (iam / reference_iam_xp[xp.newaxis, :])
                    diff = pred_mol - target_function_xp[xp.newaxis, :]
                    sse = xp.sum(diff * diff, axis=1)
                    predicted_function_ = pred_mol + _pcd_offset_correction
                    xray_contrib = sse * inv_qlen
                else:
                    diff = iam - target_function_xp[xp.newaxis, :]
                    sse = xp.sum(
                        (diff * diff) / abs_target_function_xp[xp.newaxis, :], axis=1
                    )
                    predicted_function_ = iam + _iam_offset
                    xray_contrib = sse * inv_qlen

                bonding_contrib = xp.zeros(n_chains, dtype=xp.float64)
                if bonds_bool and nbonds > 0:
                    vec = (
                        xyz_trial[:, bond_atom1_idx, :]
                        - xyz_trial[:, bond_atom2_idx, :]
                    )
                    r_b = xp.sqrt(xp.sum(vec * vec, axis=2))
                    bonding_contrib = xp.sum(
                        k_xp[xp.newaxis, :] * HALF * (r_b - r0_xp[xp.newaxis, :]) ** 2,
                        axis=1,
                    )

                angular_contrib = xp.zeros(n_chains, dtype=xp.float64)
                if angles_bool and nangles > 0:
                    ba = (
                        xyz_trial[:, angle_atom1_idx, :]
                        - xyz_trial[:, angle_atom2_idx, :]
                    )
                    bc = (
                        xyz_trial[:, angle_atom3_idx, :]
                        - xyz_trial[:, angle_atom2_idx, :]
                    )
                    norm_ba = xp.sqrt(xp.sum(ba * ba, axis=2))
                    norm_bc = xp.sqrt(xp.sum(bc * bc, axis=2))
                    cos_theta = xp.sum(ba * bc, axis=2) / (norm_ba * norm_bc)
                    cos_theta = xp.clip(cos_theta, -1.0, 1.0)
                    theta = xp.arccos(cos_theta)
                    angular_contrib = xp.sum(
                        k_theta_xp[xp.newaxis, :]
                        * HALF
                        * (theta - theta0_xp[xp.newaxis, :]) ** 2,
                        axis=1,
                    )

                # Vectorised torsion (all dihedrals in one batch)
                torsion_contrib = xp.zeros(n_chains, dtype=xp.float64)
                if torsions_bool and ntorsions > 0:
                    t_p0 = xyz_trial[:, torsion_a1_xp, :]
                    t_p1 = xyz_trial[:, torsion_a2_xp, :]
                    t_p2 = xyz_trial[:, torsion_a3_xp, :]
                    t_p3 = xyz_trial[:, torsion_a4_xp, :]
                    b0 = t_p0 - t_p1
                    b1 = t_p2 - t_p1
                    b2 = t_p3 - t_p2
                    b1_norm = xp.sqrt(xp.sum(b1 * b1, axis=2, keepdims=True))
                    b1 = b1 / b1_norm
                    v = b0 - xp.sum(b0 * b1, axis=2, keepdims=True) * b1
                    w = b2 - xp.sum(b2 * b1, axis=2, keepdims=True) * b1
                    x_t = xp.sum(v * w, axis=2)
                    y_t = xp.sum(xp.cross(b1, v) * w, axis=2)
                    torsion_angles = xp.arctan2(y_t, x_t)
                    torsion_contrib = xp.sum(
                        k_delta_xp[xp.newaxis, :]
                        * (1 + xp.cos(torsion_angles - delta0_xp[xp.newaxis, :])),
                        axis=1,
                    )

                f_ = xray_contrib + c_tuning * (
                    bonding_contrib + angular_contrib + torsion_contrib
                )
                improve = xp.logical_and(~temp_accept, f_ < f * 0.999)
                improve_f = improve.astype(xp.float64)
                c_xp += improve.astype(xp.int64)
                f = xp.where(improve, f_, f)
                xyz = xp.where(
                    improve[:, xp.newaxis, xp.newaxis], xyz_trial, xyz
                )

                is_new_best = xp.logical_and(improve, f < f_best)
                f_best = xp.where(is_new_best, f, f_best)
                f_xray_best = xp.where(is_new_best, xray_contrib, f_xray_best)
                xyz_best = xp.where(
                    is_new_best[:, xp.newaxis, xp.newaxis], xyz, xyz_best
                )
                predicted_best = xp.where(
                    is_new_best[:, xp.newaxis],
                    predicted_function_,
                    predicted_best,
                )

                total_bonding_xp += c_tuning * bonding_contrib * improve_f
                total_angular_xp += c_tuning * angular_contrib * improve_f
                total_torsional_xp += c_tuning * torsion_contrib * improve_f
                total_xray_xp += xray_contrib * improve_f

            # Reduce GPU accumulators to host scalars (single sync)
            c = int(to_numpy(xp.sum(c_xp), xp))
            total_bonding_contrib = float(to_numpy(xp.sum(total_bonding_xp), xp))
            total_angular_contrib = float(to_numpy(xp.sum(total_angular_xp), xp))
            total_torsional_contrib = float(to_numpy(xp.sum(total_torsional_xp), xp))
            total_xray_contrib = float(to_numpy(xp.sum(total_xray_xp), xp))

            priors_contrib = (
                total_bonding_contrib + total_angular_contrib + total_torsional_contrib
            )
            total_contrib = total_xray_contrib + priors_contrib
            if total_contrib > 0 and priors_contrib > 0:
                xray_ratio = total_xray_contrib / total_contrib
                bonding_ratio = total_bonding_contrib / total_contrib
                angular_ratio = total_angular_contrib / total_contrib
                torsional_ratio = total_torsional_contrib / total_contrib
                c_tuning_adjusted = (
                    (1 - tuning_ratio_target)
                    * c_tuning
                    / (1 - total_xray_contrib / total_contrib)
                )
            else:
                (
                    xray_ratio,
                    bonding_ratio,
                    angular_ratio,
                    torsional_ratio,
                    c_tuning_adjusted,
                ) = (0, 0, 0, 0, 0)

            best_chain_idx = int(to_numpy(xp.argmin(f_best), xp))
            f_best_scalar = float(to_numpy(f_best[best_chain_idx], xp))
            f_xray_best_scalar = float(to_numpy(f_xray_best[best_chain_idx], xp))
            predicted_best_chain = predicted_best[best_chain_idx]
            xyz_best_chain = xyz_best[best_chain_idx]
            loop_time_s = default_timer() - loop_start

            return (
                f_best_scalar,
                f_xray_best_scalar,
                predicted_best_chain,
                xyz_best_chain,
                xray_ratio,
                bonding_ratio,
                angular_ratio,
                torsional_ratio,
                c,
                c_tuning_adjusted,
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
                best_chain_idx,
                prep_time_s,
                loop_time_s,
            )

        ### Call the run_annealing() function...
        start = default_timer()
        backend_name = str(backend).lower().strip()
        if backend_name == "cpu":
            (
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
                xray_ratio,
                bonding_ratio,
                angular_ratio,
                torsional_ratio,
                c,
                c_tuning_adjusted,
            ) = run_annealing(nsteps)
        else:
            backend_info = get_backend(backend_name, emulate=gpu_emulation)
            gpu_d2h_time_s = 0.0
            if gpu_chains > 1:
                print(
                    f"[GPU] Multi-chain mode enabled: launching {gpu_chains} chains "
                    f"(nsteps={nsteps}, backend={backend_name})"
                )
                
            (
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
                xray_ratio,
                bonding_ratio,
                angular_ratio,
                torsional_ratio,
                c,
                c_tuning_adjusted,
                f_best_all,
                f_xray_best_all,
                predicted_best_all,
                xyz_best_all,
                best_chain_idx,
                prep_time_s,
                loop_time_s,
            ) = run_annealing_gpu_batched(nsteps, backend_info.xp, gpu_chains)
            if gpu_chains > 1:
                if keep_on_device:
                    self.last_chain_results = {
                        "f_best_all": f_best_all,
                        "f_xray_best_all": f_xray_best_all,
                        "predicted_best_all": predicted_best_all,
                        "xyz_best_all": xyz_best_all,
                        "best_chain_idx": int(best_chain_idx),
                    }
                else:
                    self.last_chain_results = {
                        "f_best_all": to_numpy(f_best_all, backend_info.xp),
                        "f_xray_best_all": to_numpy(f_xray_best_all, backend_info.xp),
                        "predicted_best_all": to_numpy(
                            predicted_best_all, backend_info.xp
                        ),
                        "xyz_best_all": to_numpy(xyz_best_all, backend_info.xp),
                        "best_chain_idx": int(best_chain_idx),
                    }
                print(f"[GPU] All {gpu_chains} chains finished.")
            if not keep_on_device:
                d2h_start = default_timer()
                predicted_best = to_numpy(predicted_best, backend_info.xp)
                xyz_best = to_numpy(xyz_best, backend_info.xp)
                gpu_d2h_time_s = default_timer() - d2h_start
            gpu_total_time_s = float(default_timer() - start)
            print(
                "[GPU-TIMING] prep(H2D+init): "
                f"{prep_time_s:6.3f}s | compute(loop): {loop_time_s:6.3f}s | "
                f"copy-back(D2H): {gpu_d2h_time_s:6.3f}s | total: {gpu_total_time_s:6.3f}s"
            )
            if keep_on_device:
                print(
                    "[GPU-TIMING] D2H copy deferred (persistent GPU mode keeps arrays on device)."
                )
        if verbose:
            print("run_annealing() time: %3.2f s" % float(default_timer() - start))
            print("xray contrib ratio: %f" % xray_ratio)
            print("bonding contrib ratio: %f" % bonding_ratio)
            print("angular contrib ratio: %f" % angular_ratio)
            print("torsional contrib ratio: %f" % torsional_ratio)
            print("Accepted / Total steps: %i/%i" % (c, nsteps))
        # end function
        return (
            f_best,
            f_xray_best,
            predicted_best,
            xyz_best,
            c_tuning_adjusted,
        )

