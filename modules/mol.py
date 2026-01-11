import math
import numpy as np
from numpy.typing import NDArray, DTypeLike

# only need for rmsd function:
from scipy.spatial.transform import Rotation as R


######
class Xyz:
    """methods to manipulate molecular coordinates (xyz)"""

    def __init__(self):
        pass

    def periodic_table(self, element: str):
        """Outputs atomic number for each element in the periodic table"""
        with open("data_/pt.txt") as pt_file:
            for line in pt_file:
                if line.split()[0] == element:
                    return int(line.split()[1])

    def atomic_mass(self, element: str):
        """Outputs atomic mass for each element in the periodic table"""
        with open("data_/atomic_masses.txt") as am_file:
            for line in am_file:
                if line.split()[0] == element:
                    return float(line.split()[1])

    # read/write xyz files

    def read_xyz(self, fname: str):
        """Read a .xyz file"""
        with open(fname, "r") as xyzfile:
            xyzheader = int(xyzfile.readline())
            comment = xyzfile.readline()
        xyzmatrix = np.loadtxt(fname, skiprows=2, usecols=[1, 2, 3])
        atomarray = np.loadtxt(fname, skiprows=2, dtype=str, usecols=[0])
        return xyzheader, comment, atomarray, xyzmatrix

    def write_xyz(self, fname: str, comment: str, atoms, xyz):
        """Write .xyz file"""
        natom = len(atoms)
        xyz = xyz.astype("|S10")  # convert to string array (max length 10)
        atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
        np.savetxt(
            fname,
            atoms_xyz,
            fmt="%s",
            delimiter=" ",
            header=str(natom) + "\n" + comment,
            footer="",
            comments="",
        )
        return

    def read_xyz_traj(self, fname: str, ntsteps: int):
        """Read a .xyz trajectory file"""
        with open(fname, "r") as xyzfile:
            natoms = int(xyzfile.readline())
            comment = xyzfile.readline()
            xyztraj = np.zeros((natoms, 3, ntsteps))
            atomarray = []
            for line in range(natoms):
                atomarray.append(xyzfile.readline().split()[0])
                print(atomarray)
        with open(fname, "r") as xyzfile:
            for t in range(ntsteps):
                print("read_xyz_traj: reading frame: %i" % t)
                natoms = int(xyzfile.readline())
                comment = xyzfile.readline()
                for line in range(natoms):
                    xyztraj[line, :, t] = xyzfile.readline().split()[1:4]
        return natoms, comment, atomarray, xyztraj

    def write_xyz_traj(self, fname, atoms, xyz_traj):
        """converts xyz_traj array to traj.xyz"""
        natom = len(atoms)
        atoms_xyz_traj = np.empty((1, 4))
        for t in range(xyz_traj.shape[2]):
            comment = "iteration: %i" % t
            xyz = xyz_traj[:, :, t]
            xyz = xyz.astype("|S14")  # convert to string array (max length 14)
            tmp = np.array([[str(natom), "", "", ""], [comment, "", "", ""]])
            atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
            atoms_xyz = np.append(tmp, atoms_xyz, axis=0)
            atoms_xyz_traj = np.append(atoms_xyz_traj, atoms_xyz, axis=0)
        atoms_xyz_traj = atoms_xyz_traj[1:, :]  # remove 1st line of array
        print("writing %s..." % fname)
        np.savetxt(
            fname,
            atoms_xyz_traj,
            fmt="%s",
            delimiter=" ",
            header="",
            footer="",
            comments="",
        )
        return

    ### distances array

    def distances_array(self, xyz: NDArray):
        """Computes matrix of distances from xyz"""
        natom = xyz.shape[0]  # number of atoms
        dist_array = np.zeros((natom, natom))  # the array of distances
        for i in range(natom):
            dist_array[i, i] = 0
            for j in range(i + 1, natom):
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                dist_array[i, j] = dist
                dist_array[j, i] = dist  # opposite elements are equal
        return dist_array

    def rmsd_atoms(self, xyz: NDArray, xyz_: NDArray, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        natoms = len(indices)
        rmsd = 0.0
        for index in indices:
            rmsd += np.sum((xyz[index, :] - xyz_[index, :]) ** 2)
        rmsd = (rmsd / natoms) ** 0.5
        return rmsd

    def rmsd_kabsch(self, xyz, xyz_, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        # take the indices for xyz
        xyz = xyz[indices, :]
        xyz_ = xyz_[indices, :]
        # centre them (remove effect of translations)
        xyz -= xyz.mean(axis=0)
        xyz_ -= xyz_.mean(axis=0)
        # rotate xyz to have max coincidence with xyz_
        rot, rmsd = R.align_vectors(xyz, xyz_)  # gives rotation matrix and post-rotation rmsd
        # xyz = rot.apply(xyz)  # apply rotation
        return rmsd, rot

    def mapd_function(self, xyz, xyz_, indices, bond_print=False):
        """calculate MAPD as defined in Yong et al. Faraday Disc. (2021)"""
        # MAPD is calculated between structures xyz and xyz_
        # indices: calculates MAPD for specified atomic indices
        nind = len(indices)
        mapd = 0
        for i in range(nind):
            for j in range(i + 1, nind):
                rij = np.linalg.norm(xyz[indices[i], :] - xyz[indices[j], :])
                rij_ = np.linalg.norm(xyz_[indices[i], :] - xyz_[indices[j], :])
                delta = np.abs(rij - rij_)
                if bond_print:
                    print("r_{%i, %i}^' : %10.9f" % (i, j, rij))
                    print("r_{%i, %i}^0 : %10.9f" % (i, j, rij_))
                    print("  |delta_{%i, %i}|  : %10.9f" % (i, j, delta))
                mapd += delta / rij_
        mapd *= 100 / (nind * (nind - 1) / 2)
        return mapd

    def mapd_distances(self, rij, rij_, bond_print):
        """calculate MAPD as defined in Yong et al. Faraday Disc. (2021)"""
        # MAPD is calculated between distance arrays rij, rij_
        nind = rij.shape[0]
        mapd = 0
        for i in range(nind):
            for j in range(i + 1, nind):
                r1 = rij[i, j]
                r0 = rij_[i, j]
                delta = np.abs(r1 - r0)
                if bond_print:
                    print("r_{%i, %i}^' : %10.9f" % (i, j, r1))
                    print("r_{%i, %i}^0 : %10.9f" % (i, j, r0))
                    print("  |delta_{%i, %i}|  : %10.9f" % (i, j, delta))
                mapd += delta / r0
        mapd *= 100 / (nind * (nind - 1) / 2)
        return mapd

    def new_dihedral(self, p: tuple):
        """Praxeolitic formula
        1 sqrt, 1 cross product"""
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)

        # vector rejections
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
        return np.degrees(np.arctan2(y, x))

    def directional_angle_3d(self, A, B, C, normal):
        A, B, C, normal = map(np.array, (A, B, C, normal))
        v1 = A - B
        v2 = C - B

        # Normalize input vectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        # Angle (unsigned)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)

        # Sign of angle via direction of cross product
        cross = np.cross(v1, v2)
        sign = np.sign(np.dot(cross, normal))

        # Apply sign
        signed_angle = np.degrees(angle) * sign % 360
        return signed_angle

    def angle_2p_3d(self, a, b, c):
        """angle between two points in 3D"""
        v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
        v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)
        return math.degrees(angle_rad[0])


### End Molecule class section
