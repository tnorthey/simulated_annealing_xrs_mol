from numba import njit
import numpy as np
from numpy.typing import NDArray
from timeit import default_timer

from modules.compute_backend import get_backend, to_numpy


#############################
class Annealing:
    """Simulated annealing functions"""

    def __init__(self):
        pass

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
    ):
        """simulated annealing minimisation to target_function"""
        ######## READ BOND/ANGLE PARAMS #######
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
        ##=#=#=# DEFINITIONS #=#=#=##
        natoms = starting_xyz.shape[0]  # number of atoms
        c_tuning = c_tuning_initial  # initialise C_tuning
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
        # Pre-compute abs(target_function) to avoid repeated abs() calls in loop
        abs_target_function = np.abs(target_function)
        ### define qx, qy, qz for Ewald mode (CPU path)
        if ewald_mode:
            r_grid, th_grid, ph_grid = np.meshgrid(qvector, th, ph, indexing="ij")
            # Convert spherical coordinates to Cartesian coordinates
            qx = r_grid * np.sin(th_grid) * np.cos(ph_grid)
            qy = r_grid * np.sin(th_grid) * np.sin(ph_grid)
            qz = r_grid * np.cos(th_grid)
        # Shift target function to remove constant IAM terms outside the SA loop
        if inelastic:
            iam_offset = atomic_total + compton
        else:
            iam_offset = atomic_total
        if pcd_mode:
            pcd_offset = 100.0 * (iam_offset / reference_iam - 1.0)
            target_function = target_function - pcd_offset
        else:
            target_function = target_function - iam_offset
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

        def run_annealing_gpu(nsteps, xp):
            def _to_scalar(value):
                return float(to_numpy(value, xp))

            # Backend arrays
            xyz = xp.asarray(starting_xyz, dtype=xp.float64).copy()
            xyz_trial = xp.empty_like(xyz)
            xyz_best = xyz.copy()
            predicted_best = xp.asarray(predicted_start, dtype=xp.float64).copy()
            f_best, f_xray_best = f_start, f_xray_start
            f = 1e9  # high initial value so 1st step will be accepted
            c = 0
            n_mode_indices = len(mode_indices)
            mdisp = xp.asarray(displacements, dtype=xp.float64)
            step_size_xp = xp.asarray(step_size_array, dtype=xp.float64)
            qvector_xp = xp.asarray(qvector, dtype=xp.float64)
            target_function_xp = xp.asarray(target_function, dtype=xp.float64)
            reference_iam_xp = xp.asarray(reference_iam, dtype=xp.float64)
            abs_target_function_xp = xp.asarray(abs_target_function, dtype=xp.float64)
            atomic_total_xp = xp.asarray(atomic_total, dtype=xp.float64)
            compton_xp = xp.asarray(compton, dtype=xp.float64)
            pre_molecular_xp = xp.asarray(pre_molecular, dtype=xp.float64)

            # Constraints arrays
            bond_atom1_idx = bond_atom1_idx_arr
            bond_atom2_idx = bond_atom2_idx_arr
            angle_atom1_idx = angle_atom1_idx_arr
            angle_atom2_idx = angle_atom2_idx_arr
            angle_atom3_idx = angle_atom3_idx_arr
            torsion_atom1_idx = torsion_atom1_idx_arr
            torsion_atom2_idx = torsion_atom2_idx_arr
            torsion_atom3_idx = torsion_atom3_idx_arr
            torsion_atom4_idx = torsion_atom4_idx_arr
            r0_xp = xp.asarray(r0_arr, dtype=xp.float64)
            k_xp = xp.asarray(k_arr, dtype=xp.float64)
            theta0_xp = xp.asarray(theta0_arr, dtype=xp.float64)
            k_theta_xp = xp.asarray(k_theta_arr, dtype=xp.float64)
            delta0_xp = xp.asarray(delta0_arr, dtype=xp.float64)
            k_delta_xp = xp.asarray(k_delta_arr, dtype=xp.float64)

            total_bonding_contrib = 0.0
            total_angular_contrib = 0.0
            total_torsional_contrib = 0.0
            total_xray_contrib = 0.0

            inv_nsteps = 1.0 / nsteps
            inv_qlen = 1.0 / qlen
            HALF = 0.5

            if ewald_mode:
                r_grid, th_grid, ph_grid = xp.meshgrid(
                    qvector_xp, xp.asarray(th), xp.asarray(ph), indexing="ij"
                )
                qx = r_grid * xp.sin(th_grid) * xp.cos(ph_grid)
                qy = r_grid * xp.sin(th_grid) * xp.sin(ph_grid)
                qz = r_grid * xp.cos(th_grid)

            for i in range(nsteps):
                tmp = 1.0 - i * inv_nsteps
                temp = starting_temp * tmp

                # Displace xyz along displacement vectors
                summed_displacement = xp.zeros_like(xyz)
                for mi in range(n_mode_indices):
                    n = mode_indices[mi]
                    scale = step_size_xp[n] * tmp * (2.0 * xp.random.random() - 1.0)
                    summed_displacement += mdisp[n] * scale
                xyz_trial = xyz + summed_displacement

                if temp > _to_scalar(xp.random.random()):
                    c += 1
                    xyz, xyz_trial = xyz_trial, xyz
                    continue

                # IAM calculation
                if ewald_mode:
                    molecular = xp.zeros((qlen, tlen, plen), dtype=xp.float64)
                    k = 0
                    for n in range(natoms - 1):
                        for m in range(n + 1, natoms):
                            fnm = pre_molecular_xp[k]
                            k += 1
                            xnm = xyz[n, 0] - xyz[m, 0]
                            ynm = xyz[n, 1] - xyz[m, 1]
                            znm = xyz[n, 2] - xyz[m, 2]
                            molecular += 2 * fnm * xp.cos(
                                (qx * xnm + qy * ynm + qz * znm)
                            )
                    iam = molecular
                else:
                    molecular = xp.zeros(qlen, dtype=xp.float64)
                    k = 0
                    for ii in range(natoms):
                        for jj in range(ii + 1, natoms):
                            dx = xyz_trial[ii, 0] - xyz_trial[jj, 0]
                            dy = xyz_trial[ii, 1] - xyz_trial[jj, 1]
                            dz = xyz_trial[ii, 2] - xyz_trial[jj, 2]
                            r = xp.sqrt(dx * dx + dy * dy + dz * dz)
                            qd = qvector_xp * r
                            molecular += 2.0 * pre_molecular_xp[k] * xp.sin(qd) / qd
                            k += 1
                    iam = molecular

                # Objective function (PCD or IAM)
                if pcd_mode:
                    pred_mol = 100.0 * (iam / reference_iam_xp)
                    diff = pred_mol - target_function_xp
                    sse = xp.sum(diff * diff)
                    if inelastic:
                        offset = atomic_total_xp + compton_xp
                    else:
                        offset = atomic_total_xp
                    predicted_function_ = pred_mol + 100.0 * (
                        offset / reference_iam_xp - 1.0
                    )
                    xray_contrib = _to_scalar(sse) * inv_qlen
                else:
                    if ewald_mode:
                        n = qlen * tlen * plen
                    else:
                        n = qlen
                    inv_n = 1.0 / n
                    diff = iam - target_function_xp
                    sse = xp.sum((diff * diff) / abs_target_function_xp)
                    if inelastic:
                        predicted_function_ = iam + atomic_total_xp + compton_xp
                    else:
                        predicted_function_ = iam + atomic_total_xp
                    xray_contrib = _to_scalar(sse) * inv_n

                bonding_contrib = 0.0
                if bonds_bool and nbonds > 0:
                    vec = xyz_trial[bond_atom1_idx] - xyz_trial[bond_atom2_idx]
                    r = xp.sqrt(xp.sum(vec * vec, axis=1))
                    bonding_contrib = _to_scalar(
                        xp.sum(k_xp * HALF * (r - r0_xp) ** 2)
                    )

                angular_contrib = 0.0
                if angles_bool and nangles > 0:
                    ba = xyz_trial[angle_atom1_idx] - xyz_trial[angle_atom2_idx]
                    bc = xyz_trial[angle_atom3_idx] - xyz_trial[angle_atom2_idx]
                    norm_ba = xp.sqrt(xp.sum(ba * ba, axis=1))
                    norm_bc = xp.sqrt(xp.sum(bc * bc, axis=1))
                    cos_theta = xp.sum(ba * bc, axis=1) / (norm_ba * norm_bc)
                    cos_theta = xp.clip(cos_theta, -1.0, 1.0)
                    theta = xp.arccos(cos_theta)
                    angular_contrib = _to_scalar(
                        xp.sum(k_theta_xp * HALF * (theta - theta0_xp) ** 2)
                    )

                torsion_contrib = 0.0
                if torsions_bool and ntorsions > 0:
                    for i_tors in range(ntorsions):
                        idx1 = torsion_atom1_idx[i_tors]
                        idx2 = torsion_atom2_idx[i_tors]
                        idx3 = torsion_atom3_idx[i_tors]
                        idx4 = torsion_atom4_idx[i_tors]
                        p0 = xyz_trial[idx1, :]
                        p1 = xyz_trial[idx2, :]
                        p2 = xyz_trial[idx3, :]
                        p3 = xyz_trial[idx4, :]

                        b0 = -1.0 * (p1 - p0)
                        b1 = p2 - p1
                        b2 = p3 - p2

                        b1_norm = xp.sqrt(xp.sum(b1 * b1))
                        b1 = b1 / b1_norm

                        v = b0 - xp.dot(b0, b1) * b1
                        w = b2 - xp.dot(b2, b1) * b1

                        x = xp.dot(v, w)
                        y = xp.dot(xp.cross(b1, v), w)
                        torsion = xp.arctan2(y, x)

                        torsion_contrib += _to_scalar(
                            k_delta_xp[i_tors]
                            * (1 + xp.cos(torsion - delta0_xp[i_tors]))
                        )

                f_ = xray_contrib + c_tuning * (
                    bonding_contrib + angular_contrib + torsion_contrib
                )

                if f_ < f * 0.999:
                    c += 1
                    f = f_
                    xyz, xyz_trial = xyz_trial, xyz
                    if f < f_best:
                        f_best = f
                        xyz_best[:, :] = xyz[:, :]
                        predicted_best = predicted_function_.copy()
                        f_xray_best = xray_contrib
                    total_bonding_contrib += c_tuning * bonding_contrib
                    total_angular_contrib += c_tuning * angular_contrib
                    total_torsional_contrib += c_tuning * torsion_contrib
                    total_xray_contrib += xray_contrib

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
            ) = run_annealing_gpu(nsteps, backend_info.xp)
            predicted_best = to_numpy(predicted_best, backend_info.xp)
            xyz_best = to_numpy(xyz_best, backend_info.xp)
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

