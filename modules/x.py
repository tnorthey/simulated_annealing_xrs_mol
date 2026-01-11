import numpy as np
from scipy import interpolate
from numpy.typing import NDArray, DTypeLike

class Xray:
    def __init__(self):
        ''' initialise x-ray variables '''

        # Initialise IAM coefficient arrays
        self.aa = np.array(
            [
                [0.489918, 0.262003, 0.196767, 0.049879],  # hydrogen
                [0.8734, 0.6309, 0.3112, 0.1780],  # helium
                [1.1282, 0.7508, 0.6175, 0.4653],  # lithium
                [1.5919, 1.1278, 0.5391, 0.7029],  # berylium
                [2.0545, 1.3326, 1.0979, 0.7068],  # boron
                [2.3100, 1.0200, 1.5886, 0.8650],  # carbon
                [12.2126, 3.1322, 2.0125, 1.1663],  # nitrogen
                [3.0485, 2.2868, 1.5463, 0.8670],  # oxygen
                [3.5392, 2.6412, 1.5170, 1.0243],  # fluorine
                [3.9553, 3.1125, 1.4546, 1.1251],  # neon
                [4.7626, 3.1736, 1.2674, 1.1128],  # sodium
                [5.4204, 2.1735, 1.2269, 2.3073],  # magnesium
                [6.4202, 1.9002, 1.5936, 1.9646],  # aluminium
                [6.2915, 3.0353, 1.9891, 1.5410],  # Siv
                [6.4345, 4.1791, 1.7800, 1.4908],  # phosphorus
                [6.9053, 5.2034, 1.4379, 1.5863],  # sulphur
                [11.4604, 7.1964, 6.2556, 1.6455],  # chlorine
            ]
        )

        self.bb = np.array(
            [
                [20.6593, 7.74039, 49.5519, 2.20159],  # hydrogen
                [9.1037, 3.3568, 22.9276, 0.9821],  # helium
                [3.9546, 1.0524, 85.3905, 168.261],  # lithium
                [43.6427, 1.8623, 103.483, 0.5420],  # berylium
                [23.2185, 1.0210, 60.3498, 0.1403],  # boron
                [20.8439, 10.2075, 0.5687, 51.6512],  # carbon
                [0.00570, 9.8933, 28.9975, 0.5826],  # nitrogen
                [13.2771, 5.7011, 0.3239, 32.9089],  # oxygen
                [10.2825, 4.2944, 0.2615, 26.1476],  # fluorine
                [8.4042, 3.4262, 0.2306, 21.7184],  # Ne
                [3.2850, 8.8422, 0.3136, 129.424],  # Na
                [2.8275, 79.2611, 0.3808, 7.1937],  # Mg
                [3.0387, 0.7426, 31.5472, 85.0886],  # Al
                [2.4386, 32.3337, 0.6785, 81.6937],  # Siv
                [1.9067, 27.1570, 0.5260, 68.1645],  # P
                [1.4679, 22.2151, 0.2536, 56.1720],  # S
                [0.0104, 1.1662, 18.5194, 47.7784],  # Cl
            ]
        )

        self.cc = np.array(
            [
                0.001305,  # hydrogen
                0.0064,  # helium
                0.0377,  # lithium
                0.0385,  # berylium
                -0.1932,  # boron
                0.2156,  # carbon
                -11.529,  # nitrogen
                0.2508,  # oxygen
                0.2776,  # fluorine
                0.3515,  # Ne
                0.6760,  # Na
                0.8584,  # Mg
                1.1151,  # Al
                1.1407,  # Si
                1.1149,  # P
                0.8669,  # S
                -9.5574,  # Cl
            ]
        )

    def iam_calc(
        self,
        atomic_numbers,
        xyz,
        qvector,
        electron_mode=False,
        inelastic=False,
        compton_array=np.zeros(0),
    ):
        """calculate IAM molecular scattering curve for atoms, xyz, qvector"""
        natoms = len(atomic_numbers)
        qlen = len(qvector)
        atomic = np.zeros(qlen)  # total atomic factor
        molecular = np.zeros(qlen)  # total molecular factor
        compton = np.zeros(qlen)  # total compton factor
        atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
        if electron_mode:  # electron mode
            zfactor = atomic_numbers
            e_mode_int = -1
        else:  # x-ray mode
            zfactor = np.multiply(0.0, atomic_numbers)
            e_mode_int = 1
        for i in range(natoms):
            tmp = self.atomic_factor(atomic_numbers[i], qvector)
            atomic_factor_array[i, :] = tmp
            atomic += (zfactor[i] - tmp) ** 2
            if inelastic:
                compton += compton_array[i, :]
        nij = int(natoms * (natoms - 1) / 2)
        pre_molecular = np.zeros(
            (nij, qlen)
        )  # pre_molecular array for speed in other functions
        k = 0  # begin counter
        for n in range(natoms - 1):
            for m in range(n + 1, natoms):  # j > i
                fnm = np.multiply(
                    zfactor[n] + e_mode_int * atomic_factor_array[n, :],
                    zfactor[m] + e_mode_int * atomic_factor_array[m, :],
                )
                pre_molecular[k, :] = (
                    fnm  # store in array for speed in other functions later
                )
                k += 1  # count iterations
                molecular += (
                    2
                    * fnm
                    * np.sinc(qvector * np.linalg.norm(xyz[n, :] - xyz[m, :]) / np.pi)
                )
        iam = atomic + molecular
        if inelastic:
            iam += compton
        return iam, atomic, molecular, compton, pre_molecular

    def spherical_rotavg(self, f, th, ph):
        """
        Rotational average in sphericals: I use it with the Ewald sphere
        f must be a 3D array with coordinates (r, theta, phi)
        Note: f must be the full sphere as I divide by the volume of a sphere 4*pi
        """
        # read size of array axes
        qlen, tlen, plen = f.shape[0], len(th), len(ph)
        # first sum over phi,
        f_rotavg_phi = np.sum(f, axis=2)
        # multiply by the sin(th) term,
        for j in range(tlen):
            f_rotavg_phi[:, j] *= np.sin(th[j])
        (dth := th[1] - th[0]) if tlen > 1 else (dth := 1)
        (dph := ph[1] - ph[0]) if plen > 1 else (dph := 1)
        f_rotavg = np.sum(f_rotavg_phi, axis=1) * dth * dph / (4 * np.pi)
        return f_rotavg

    def iam_calc_ewald(
        self,
        atomic_numbers,
        xyz,
        qvector,
        th,
        ph,
        inelastic=False,
        compton_array=np.zeros(0),
    ):
        """
        calculate IAM function in the Ewald sphere
        """
        natoms = len(atomic_numbers)
        qlen, tlen, plen = len(qvector), len(th), len(ph)
        qmin, tmin, pmin = min(qvector), min(th), min(ph)
        qmax, tmax, pmax = max(qvector), max(th), max(ph)
        # define coordinates on meshgrid
        r_grid, th_grid, ph_grid = np.meshgrid(qvector, th, ph, indexing="ij")
        # Convert spherical coordinates to Cartesian coordinates
        qx = r_grid * np.sin(th_grid) * np.cos(ph_grid)
        qy = r_grid * np.sin(th_grid) * np.sin(ph_grid)
        qz = r_grid * np.cos(th_grid)
        # inelastic effects
        compton = np.zeros((qlen, tlen, plen))  # total compton factor
        # atomic
        atomic = np.zeros((qlen, tlen, plen))  # total atomic factor
        atomic_factor_array = np.zeros(
            (natoms, qlen, tlen, plen)
        )  # array of atomic factors
        for n in range(natoms):
            for i in range(plen):
                for j in range(tlen):
                    atomic_factor_array[n, :, j, i] = self.atomic_factor(
                        atomic_numbers[n], qvector
                    )
                    if inelastic:
                        compton[:, j, i] += compton_array[n, :]
            atomic += np.power(atomic_factor_array[n, :, :, :], 2)
        # molecular
        molecular = np.zeros((qlen, tlen, plen))  # total molecular factor
        nij = int(natoms * (natoms - 1) / 2)
        pre_molecular = np.zeros(
            (nij, qlen, tlen, plen)
        )  # pre_molecular array for speed in other functions
        k = 0  # begin counter
        for n in range(natoms - 1):
            for m in range(n + 1, natoms):  # j > i
                fnm = np.multiply(
                    atomic_factor_array[n, :, :, :], atomic_factor_array[m, :, :, :]
                )
                pre_molecular[k, :, :, :] = (
                    fnm  # store in array for speed in other functions later
                )
                k += 1  # count iterations
                xnm = xyz[n, 0] - xyz[m, 0]
                ynm = xyz[n, 1] - xyz[m, 1]
                znm = xyz[n, 2] - xyz[m, 2]
                molecular += 2 * fnm * np.cos((qx * xnm + qy * ynm + qz * znm))
        iam_total = atomic + molecular + compton
        atomic_rotavg = np.sum(atomic, axis=(1, 2)) / (tlen * plen)
        compton_rotavg = np.sum(compton, axis=(1, 2)) / (tlen * plen)
        # molecular rotatational average includes area element sin(th)dth*dph
        molecular_rotavg = self.spherical_rotavg(molecular, th, ph)
        # note: the Ewald sphere rotational average tends towards the exact IAM solution
        iam_total_rotavg = atomic_rotavg + molecular_rotavg + compton_rotavg
        return (
            iam_total,
            atomic,
            molecular,
            compton,
            pre_molecular,
            iam_total_rotavg,
            atomic_rotavg,
            molecular_rotavg,
            compton_rotavg,
        )

    def atomic_factor(self, atom_number, qvector):
        """returns atomic x-ray scattering factor for atom_number, and qvector"""
        aa, bb, cc = self.aa, self.bb, self.cc
        if isinstance(qvector, float):
            qvector = np.array([qvector])
        qlen = len(qvector)
        atomfactor = np.zeros(qlen)
        for j in range(qlen):
            for i in range(4):
                atomfactor[j] += aa[atom_number - 1, i] * np.exp(
                    -bb[atom_number - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                )
        atomfactor += cc[atom_number - 1]
        return atomfactor

    def compton_spline(self, atomic_numbers, qvector):
        """spline the compton factors to correct qvector, outputs array (atoms, qvector)"""
        natom = len(atomic_numbers)
        compton_array = np.zeros(
            (natom, len(qvector))
        )  # inelastic component for each atom
        tmp = np.load("data_/Compton_Scattering_Intensities.npz")  # compton factors
        q_compton, arr = tmp["q_compton"], tmp["compton"]
        for i in range(natom):
            tck = interpolate.splrep(q_compton, arr[atomic_numbers[i] - 1, :], s=0)
            compton_array[i, :] = interpolate.splev(qvector, tck, der=0)
        return compton_array

    def iam_calc_2d(self, atomic_numbers, xyz, qvector):
        """
        calculate IAM molecular scattering curve for atoms, xyz, qvector
        q on a 2d grid in radial scattering angle theta [0, pi] and azimuthal phi [0, 2*pi]
        """
        natom = len(atomic_numbers)
        qlen = len(qvector)
        # print('q')
        # print(qvector)
        qmin = qvector[0]
        qmax = qvector[-1]
        theta_min = 2 * np.arcsin(qmin / qmax)
        # print("theta_min")
        # print(theta_min)
        theta = np.linspace(theta_min, 1 * np.pi, qlen, endpoint=True)
        phi = np.linspace(0, 2 * np.pi, qlen, endpoint=True)
        print(theta)
        print(phi)
        # qx etc. must be a 2D grid...
        # hmmmm, this works but there might be a better way.
        qx = np.zeros((qlen, qlen))
        qy = np.zeros((qlen, qlen))
        qz = np.zeros((qlen, qlen))
        # why is au2ang involved? might be wrong
        # au2ang = 0.52918
        # k0 = au2ang * qmax / 2
        k0 = qmax / 2
        for i in range(qlen):
            for j in range(qlen):
                qx[i, j] = (
                    -2
                    * k0
                    * np.sin(theta[i] / 2)
                    * np.cos(theta[i] / 2)
                    * np.cos(phi[j])
                )
                qy[i, j] = (
                    -2
                    * k0
                    * np.sin(theta[i] / 2)
                    * np.cos(theta[i] / 2)
                    * np.sin(phi[j])
                )
                qz[i, j] = 2 * k0 * np.sin(theta[i] / 2) * np.sin(theta[i] / 2)
                # qx[i, j] = -k0 * np.sin(theta[i]) * np.cos(phi[j])
                # qy[i, j] = -k0 * np.sin(theta[i]) * np.sin(phi[j])
                # qz[i, j] = k0 * (1 - np.cos(theta[i]))
        atomic = np.zeros((qlen, qlen))  # total atomic factor
        molecular = np.zeros((qlen, qlen))  # total molecular factor
        atomic_factor_array = np.zeros((natom, qlen, qlen))  # array of atomic factors
        # check qmin
        # q_check = (qx ** 2 + qy ** 2 + qz ** 2) ** 0.5
        # print("qmin (check)")
        # print(q_check[0, 0])
        # atomic
        for i in range(natom):
            for j in range(qlen):
                atomic_factor_array[i, :, j] = self.atomic_factor(
                    atomic_numbers[i], qvector
                )
            atomic += np.power(atomic_factor_array[i, :, :], 2)
        # molecular
        for i in range(natom):
            for j in range(i + 1, natom):  # j > i
                fij = np.multiply(
                    atomic_factor_array[i, :, :], atomic_factor_array[j, :, :]
                )
                xij = xyz[i, 0] - xyz[j, 0]
                yij = xyz[i, 1] - xyz[j, 1]
                zij = xyz[i, 2] - xyz[j, 2]
                molecular += fij * np.cos(qx * xij + qy * yij + qz * zij)
        molecular *= 2
        iam_total = atomic + molecular
        rotavg = np.sum(iam_total, axis=1) / qlen  # phi average.. I think is correct
        return iam_total, atomic, molecular, atomic_factor_array, rotavg, qx, qy, qz

    ### other functions ... that may be called by the Gradient descent.
    ###

    def jq_atomic_factors_calc(self, atomic_numbers, qvector):
        """Calculate the atomic term of IAM x-ray scattering, J(q)
        and the array of atomic factors, [f(q)]"""
        natoms = len(atomic_numbers)
        qlen = len(qvector)
        atomic_factor_arr = np.zeros((natoms, qlen))  # array of atomic factors
        jq = np.zeros(qlen)  # total atomic factor
        for i in range(natoms):
            tmp = self.atomic_factor(atomic_numbers[i], qvector)
            atomic_factor_arr[i, :] = tmp
            jq += tmp**2
        return jq, atomic_factor_arr

    def compton_spline_calc(self, atomic_numbers, qvector):
        """spline the compton factors to correct qvector, outputs array (atoms, qvector)"""
        natoms = len(atomic_numbers)
        compton_array = np.zeros(
            (natoms, len(qvector))
        )  # inelastic component for each atom
        tmp = np.load("data_/Compton_Scattering_Intensities.npz")  # compton factors
        q_compton, arr = tmp["q_compton"], tmp["compton"]
        for i in range(natoms):
            tck = interpolate.splrep(q_compton, arr[atomic_numbers[i] - 1, :], s=0)
            compton_array[i, :] = interpolate.splev(qvector, tck, der=0)
        compton_total = np.sum(compton_array, axis=0)
        return compton_total, compton_array

    def Imol_calc(self, atomic_factor_arr, xyz, qvector):
        """Calculate the molecular term of IAM x-ray scattering, Imol(q)"""
        natoms = xyz.shape[0]
        Imol = np.zeros(len(qvector))  # total molecular factor, Imol(q)
        for i in range(natoms):
            for j in range(i + 1, natoms):  # j > i
                Imol += (
                    2
                    * np.multiply(atomic_factor_arr[i, :], atomic_factor_arr[j, :])
                    * np.sinc(qvector * np.linalg.norm(xyz[i, :] - xyz[j, :]) / np.pi)
                )
        return Imol


### End Xray class section
