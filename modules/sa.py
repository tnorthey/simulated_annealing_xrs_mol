from random import random
from numba import njit
import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray, DTypeLike
import sys
from timeit import default_timer
import cProfile
import pstats

# my modules
import modules.mol as mol
import modules.x as xray

# create class objects
m = mol.Xyz()
x = xray.Xray()


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
        if not inelastic:
            compton = 0
        ##=#=#=# END DEFINITIONS #=#=#=#
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        def angle3(A, B, C):
            """Return angle ABC (at B) given three positions A, B, C."""
            BA = A - B
            BC = C - B
            cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            return angle_rad

        def angle_array(angular_indices):
            """calculate starting angles for angular indices"""
            nangular_indices = len(angular_indices[0])  # number of angular indices
            theta_arr = np.zeros((nangular_indices))
            for i_ang in range(nangular_indices):
                p0 = starting_xyz[angular_indices[0][i_ang], :]
                p1 = starting_xyz[angular_indices[1][i_ang], :]
                p2 = starting_xyz[angular_indices[2][i_ang], :]
                ba = p1 - p0
                bc = p1 - p2
                cosine_theta = np.dot(ba, bc) / (
                    np.linalg.norm(ba) * np.linalg.norm(bc)
                )
                theta_arr[i_ang] = np.arccos(cosine_theta)
            return theta_arr

        # print(np.degrees(theta0_arr))
        # print("HO factors: %4.3f %4.3f" % (bonding_factor[0], bonding_factor[1]))
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
        ### define qx, qy, qz for Ewald mode
        if ewald_mode:
            r_grid, th_grid, ph_grid = np.meshgrid(qvector, th, ph, indexing="ij")
            # Convert spherical coordinates to Cartesian coordinates
            qx = r_grid * np.sin(th_grid) * np.cos(ph_grid)
            qy = r_grid * np.sin(th_grid) * np.sin(ph_grid)
            qz = r_grid * np.cos(th_grid)
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        @njit(nogil=True)  # numba decorator to compile to machine code
        def run_annealing(nsteps):

            ##=#=#=# INITIATE LOOP VARIABLES #=#=#=#=#
            xyz, xyz_best = starting_xyz, starting_xyz
            f_best, f_xray_best = f_start, f_xray_start  # initialise on restart
            predicted_best = predicted_start  # initialise on restart
            f = 1e9  # high initial value so 1st step will be accepted
            c = 0  # count accepted steps
            mdisp = displacements
            (
                total_bonding_contrib,
                total_angular_contrib,
                total_torsional_contrib,
                total_xray_contrib,
            ) = (0, 0, 0, 0)
            ##=#=#=# END INITIATE LOOP VARIABLES #=#=#

            for i in range(nsteps):

                ##=#=#=#=# TEMPERATURE #=#=#=#=#=#=#=#=##
                tmp = 1 - i / nsteps  # this is prop. to how far the molecule moves
                temp = starting_temp * tmp  # this is the probability of going uphill
                ##=#=#=# END TEMPERATURE #=#=#=#=#=#=#=##

                ##=#=#=# DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##
                # this is faster in numba than the vectorised version...
                # numba likes loops (not vectors apparently)
                summed_displacement = np.zeros(mdisp[0, :, :].shape)
                for n in mode_indices:
                    summed_displacement += (
                        mdisp[n, :, :] * step_size_array[n] * tmp * (2 * random() - 1)
                    )
                xyz_ = xyz + summed_displacement  # save a temporary displaced xyz: xyz_
                ##=#=#=# END DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##

                ##=#=#=# TEMPERATURE ACCEPTANCE CRITERIA #=#=#=##
                if temp > random():
                    c += 1  # count acceptances
                    xyz = xyz_  # update xyz to displaced xyz
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
                    molecular = np.zeros(qlen)  # total molecular factor
                    k = 0
                    for ii in range(natoms):
                        for jj in range(ii + 1, natoms):  # j > i
                            qdij = qvector * LA.norm(xyz_[ii, :] - xyz_[jj, :])
                            molecular += 2 * pre_molecular[k, :] * np.sin(qdij) / qdij
                            k += 1
                iam_ = atomic_total + molecular + compton
                ##=#=#=# END IAM CALCULATION #=#=#=##

                ##=#=#=# PCD & DSIGNAL CALCULATIONS #=#=#=##
                if pcd_mode:
                    predicted_function_ = 100 * (iam_ / reference_iam - 1)
                    ### x-ray part of objective function
                    ### TO DO: depends on ewald_mode ...
                    xray_contrib = (
                        np.sum((predicted_function_ - target_function) ** 2) / qlen
                    )
                else:
                    predicted_function_ = iam_
                    ### x-ray part of objective function
                    ### TO DO: depends on ewald_mode ...
                    if ewald_mode:
                        n = qlen * tlen * plen
                    else:
                        n = qlen
                    xray_contrib = (
                        np.sum(
                            (predicted_function_ - target_function) ** 2
                            / np.abs(target_function)
                        )
                        / n
                    )

                ### harmonic oscillator part of f
                # somehow this is faster in numba than the vectorised version
                # New method: read from bond_param_array (OpenMM params)
                bonding_contrib = 0
                if bonds_bool:
                    for i in range(nbonds):
                        r = LA.norm(
                            xyz_[bond_atom1_idx_arr[i], :]
                            - xyz_[bond_atom2_idx_arr[i], :]
                        )
                        bonding_contrib += k_arr[i] * 0.5 * (r - r0_arr[i]) ** 2

                angular_contrib = 0
                if angles_bool:
                    for i in range(nangles):
                        """Return angle ABC (at B) given three positions A, B, C."""
                        """Faster to not call outside function. And works better with numba."""
                        BA = (
                            xyz_[angle_atom1_idx_arr[i], :]
                            - xyz_[angle_atom2_idx_arr[i], :]
                        )
                        BC = (
                            xyz_[angle_atom3_idx_arr[i], :]
                            - xyz_[angle_atom2_idx_arr[i], :]
                        )
                        norm_BA = np.sqrt(np.sum(BA * BA))
                        norm_BC = np.sqrt(np.sum(BC * BC))
                        cos_theta = np.dot(BA, BC) / (norm_BA * norm_BC)
                        cos_theta = min(
                            1.0, max(-1.0, cos_theta)
                        )  # stops cosine being out of range [-1, 1] due to slight numerical flucuations
                        theta = np.arccos(cos_theta)

                        angular_contrib += (
                            k_theta_arr[i] * 0.5 * (theta - theta0_arr[i]) ** 2
                        )

                torsion_contrib = 0
                if torsions_bool:
                    for i in range(ntorsions):
                        # calculate the dihedrals
                        p0 = xyz_[torsion_atom1_idx_arr[i], :]
                        p1 = xyz_[torsion_atom2_idx_arr[i], :]
                        p2 = xyz_[torsion_atom3_idx_arr[i], :]
                        p3 = xyz_[torsion_atom4_idx_arr[i], :]

                        b0 = -1.0 * (p1 - p0)
                        b1 = p2 - p1
                        b2 = p3 - p2

                        # normalize b1 so that it does not influence magnitude of vector
                        # projections that come next
                        b1 /= np.linalg.norm(b1)

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

                        torsion_contrib += k_delta_arr[i] * (
                            1 + np.cos(torsion - delta0_arr[i])
                        )

                ### combine x-ray and bonding, angular contributions
                f_ = xray_contrib + c_tuning * (
                    bonding_contrib + angular_contrib + torsion_contrib
                )
                ##=#=#=# END PCD & DSIGNAL CALCULATIONS #=#=#=##

                ##=#=#=# ACCEPTANCE CRITERIA #=#=#=##
                if f_ / f < 0.999:
                    c += 1  # count acceptances
                    f, xyz = f_, xyz_  # update f and xyz
                    if f < f_best:
                        # store values corresponding to f_best
                        f_best, xyz_best, predicted_best = (
                            f,
                            xyz,
                            predicted_function_,
                        )
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

        ### Call the run_annealing() function...
        start = default_timer()
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
        print("run_annealing() time: %3.2f s" % float(default_timer() - start))
        ###
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

    def read_nm_displacements(self, fname: str, natoms: int) -> NDArray:
        """read_nm_displacements: Reads displacement vector from file=fname e.g. 'normalmodes.txt'
        Inputs: 	natoms (int), total number of atoms
        Outputs:	displacements, array of displacements, size: (nmodes, natoms, 3)"""
        if natoms == 2:
            nmodes = 1
        elif natoms > 2:
            nmodes = 3 * natoms - 6
        else:
            print("ERROR: natoms. Are there < 2 atoms?")
            return False
        with open(fname, "r") as xyzfile:
            tmp = np.loadtxt(fname)
        displacements = np.zeros((nmodes, natoms, 3))
        for i in range(3 * natoms):
            for j in range(nmodes):
                if i % 3 == 0:  # Indices 0,3,6,...
                    dindex = int(i / 3)
                    displacements[j, dindex, 0] = tmp[i, j]  # x coordinates
                elif (i - 1) % 3 == 0:  # Indices 1,4,7,...
                    displacements[j, dindex, 1] = tmp[i, j]  # y coordinates
                elif (i - 2) % 3 == 0:  # Indices 2,5,8,...
                    displacements[j, dindex, 2] = tmp[i, j]  # z coordinates
        return displacements

    def displacements_from_wavenumbers(self, wavenumbers, step_size, exponential=False):
        nmodes = len(wavenumbers)
        displacement_factors = np.zeros(nmodes)
        for i in range(nmodes):  # initial factors are inv. prop. to wavenumber
            if wavenumbers[i] > 0:
                if exponential:
                    displacement_factors[i] = np.exp(wavenumbers[0] / wavenumbers[i])
                else:
                    displacement_factors[i] = wavenumbers[0] / wavenumbers[i]
            else:
                displacement_factors[i] = 0.0
        displacement_factors *= step_size  # adjust max size of displacement step
        return displacement_factors

    def uniform_factors(self, nmodes, displacement_factors):
        """uniformly random displacement step along each mode"""
        # initialise random number generator (with random seed)
        rng = np.random.default_rng()
        factors = np.zeros(nmodes)
        for j in range(nmodes):
            # random factors in range [-a, a]
            a = displacement_factors[j]
            factors[j] = 2 * a * rng.random() - a
        return factors

    def simulate_trajectory(
        self, starting_xyz, displacements, wavenumbers, nsteps, step_size
    ):
        """creates a simulated trajectory by randomly moving along normal modes"""
        natom = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        xyz = starting_xyz  # start at starting xyz
        xyz_traj = np.zeros((natom, 3, nsteps))
        for i in range(nsteps):
            factors = self.uniform_factors(
                nmodes, displacement_factors
            )  # random factors
            xyz = nm.nm_displacer(xyz, displacements, modes, factors)
            xyz_traj[:, :, i] = xyz
        return xyz_traj
