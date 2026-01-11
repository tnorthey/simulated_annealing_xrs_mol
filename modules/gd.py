"""
Gradient descent with IAM
"""
import numpy as np
# my modules
import modules.x as xray

# create class objects
x = xray.Xray()

class G:
    """Gradient descent functions"""

    def __init__(self):
        pass

    #### Functions for:
    #### The Analytical gradient descent of chi2 (defined by IAM)

    def dImoldx(self, xyz, fqi, qvector):
        """partial derviative of Imol(q) wrt xi"""
        ### Inputs:
        ###     xi,         xj 1D arrays of size natoms
        ###     fqi,        2D array of size (qlen, natoms)
        ###     qvector,    1D array of size qlen
        ### Outputs:
        ###     dImoldx_,    3D array of size (qlen, natoms, 3)
        ###      it is an array of partial derivatives of Imol(q) at each point in q,
        ###      and for each Cartesian (there are natoms * 3 of them)
        # print("start of dImoldx function")
        # distance array, rij
        # rij = m.distances_array(xyz_)
        natoms = xyz.shape[0]
        qlen = len(qvector)
        dImoldx_ = np.zeros((qlen, natoms, 3))
        for i in range(natoms):
            for j in range(i + 1, natoms):
                for k in range(3):
                    rij = (np.sum((xyz[i, :] - xyz[j, :]) ** 2)) ** 0.5
                    qr = qvector * rij
                    dImoldx_[:, i, k] += (  # is this right...?
                        2
                        * 2
                        * qvector
                        * (xyz[i, k] - xyz[j, k])
                        * fqi[i, :]
                        * fqi[j, :]
                        * (np.cos(qr) / qr - np.sin(qr) / qr ** 2)
                        / rij
                    )
        return dImoldx_


    def gradient_descent_cartesian(
        self,
        target_function,
        atomic_numbers,
        starting_xyz,
        qvector,
        nsteps=1000,
        step_size=0.1,
        pcd_mode=False,
        reference_iam=0,
    ):
        #### untested with pcd_mode! #### !!!!!
        """Gradient descent in Cartesians"""

        qlen = len(qvector)
        # starting xyz
        xyz_ = starting_xyz
        natoms = xyz_.shape[0]
        print("Natoms = %i" % natoms)

        # the target function, Y(q)
        Yq = target_function

        # Iat, atomic factor array
        Iat, fqi = x.jq_atomic_factors_calc(atomic_numbers, qvector)

        # Sq (this one only changes with xyz)
        Imol = x.Imol_calc(fqi, xyz_, qvector)

        # Compton effects
        Icompton, compton_array = x.compton_spline_calc(atomic_numbers, qvector)

        # while loop
        c = 0
        chi2_best = 1e9
        while c < nsteps:
            c += 1

            dImoldx_ = self.dImoldx(xyz_, fqi, qvector)  # partial derivatives of S(q)
            print('dImoldx_')
            print(dImoldx_)
            print('end dImoldx_')

            # chi
            if pcd_mode:
                chi1 = 100 * ((Iat + Imol + Icompton) / reference_iam - 1) - Yq
            else:
                chi1 = Iat + Imol + Icompton - Yq

            print('chi1')
            # The first term of chi1 is 0 (as expected?), is that why? It's possibly multiplied wrong?
            print(chi1)
            # multiply them appropriately
            for i in range(natoms):
                for k in range(3):
                    dImoldx_[:, i, k] *= chi1

            print(dImoldx_)
            #### propagated from here ... ERROR HERE, LAST ROW IS ALWAYS ALL ZEROS...
            # partial derivatives of chi2
            dchi2dxyz = 2 * step_size * np.sum(dImoldx_, axis=0) / qlen

            # go downhill
            print(dchi2dxyz)
            #### ... ERROR HERE, LAST ROW IS ALWAYS ALL ZEROS...
            xyz_ -= dchi2dxyz

            ## Calculate Imol (molecular term)
            Imol = x.Imol_calc(fqi, xyz_, qvector)
            ## full IAM (atomic term + molecular term + Compton effects)
            if pcd_mode:
                predicted_ = 100 * ((Iat + Icompton + Imol) / reference_iam - 1)
            else:
                predicted_ = Iat + Icompton + Imol

            ## Calculate chi2
            chi2_ = np.sum((predicted_ - Yq) ** 2) / qlen
            print(chi2_)
            if chi2_ < chi2_best:
                chi2_best = chi2_
                xyz_best = xyz_
                predicted_best = predicted_

        return chi2_best, predicted_best, xyz_best


    ##################################
    """ second derivative stuff... """
    ##################################

    def d2Imoldxi2(self, xyz, fqi, qvector):
        """2nd partial derviative of Imol(q) wrt xi"""
        ### Inputs:
        ###     xi,         xj 1D arrays of size natoms
        ###     fqi,        2D array of size (qlen, natoms)
        ###     qvector,    1D array of size qlen
        ### Outputs:
        ###     d2Imoldxi2_,    3D array of size (qlen, natoms, 3)
        ###      it is an array of double partial derivatives of S(q) at each point in q,
        ###      and for each Cartesian (there are natoms * 3 of them)
        # print("start of dsdxi function")
        # distance array, rij
        q = qvector
        rij = m.distances_array(xyz)
        natoms = xyz.shape[0]
        qlen = len(q)
        d2Imoldxi2_ = np.zeros((qlen, natoms, 3))
        for i in range(natoms):
            for j in range(i + 1, natoms):
                for k in range(3):
                    rij = (np.sum((xyz[i, :] - xyz[j, :]) ** 2)) ** 0.5
                    qr = q * rij
                    B = np.cos(qr) / qr - np.sin(qr) / qr ** 2
                    xyzij = xyz[i, k] - xyz[j, k]
                    common_factor = q * fqi[i, :] * fqi[j, :] / rij
                    tmp = xyzij ** 2 / rij ** 2
                    d2Imoldxi2_[:, i, k] += common_factor * (
                        B
                        - B * tmp
                        + tmp
                        * (2 * np.sin(qr) / q ** 2 - 2 * np.cos(qr) / qr - np.sin(qr))
                    )
        return d2Imoldxi2_


    def d2Idij(self, xyz, fqi, qvector):
        """2nd partial derviative of Imol(q) wrt xi and xj"""
        ### Inputs:
        ###     xi,         xj 1D arrays of size natoms
        ###     fqi,        2D array of size (qlen, natoms)
        ###     qvector,    1D array of size qlen
        ### Outputs:
        ###     d2Idij_,    3D array of size (qlen, natoms, 3)
        ###      it is an array of double partial derivatives of S(q) at each point in q,
        ###      and for each Cartesian (there are natoms * 3 of them)
        # print("start of dsdxi function")
        # distance array, rij
        q = qvector
        rij = m.distances_array(xyz)
        natoms = xyz.shape[0]
        qlen = len(q)
        d2Idij_ = np.zeros((qlen, natoms, 3))
        for i in range(natoms):
            for j in range(i + 1, natoms):
                for k in range(3):
                    rij = (np.sum((xyz[i, :] - xyz[j, :]) ** 2)) ** 0.5
                    qr = q * rij
                    xi = xyz[i, k]
                    xj = xyz[j, k]
                    xij = xi - xj
                    common_factor = q * fqi[i, :] * fqi[j, :]
                    A = -xij / rij ** 3
                    B = xij / rij
                    C = -xij / rij ** 2
                    d2Idij_[:, i, k] += common_factor * (
                        (-xi * A + xj * A - 1 / rij)
                        * (-np.sin(qr) / qr ** 2 + np.cos(qr) / qr)
                        - B
                        * (
                            2 * C * np.sin(qr) / q ** 2
                            + A * np.cos(qr) / q
                            - B * np.cos(qr) / qr
                            - B * np.sin(qr) / r
                        )
                    )
        return d2Idij_


    def create_d2chi2_matrix():
        """ create second derivative matrix of chi2 """

        d2chi2_arr = np.zeros((nat, nat))
        Imol = self.Imol_calc(atomic_factor_arr, xyz, qvector)

        # diagonal elements
        for i in range(nat):
            d2Imoldxi2_ = self.d2Imoldxi2( xyz, fqi, qvector )
            dImoldx_ = self.dImoldx( xyz, fqi, qvector )
            d2chi2_arr[i, i] = ( 2 / Nq ) * np.sum( dImoldx_ ** 2 + d2Imoldxi2_ * (Z + Imol) )

        # off-diagonal elements
        for i in range(nat):
            for j in range(i + 1, nat):
                pass
                #d2chi2_arr[i, j] = 

        return d2chi2_arr


    def eigen_(square_array):
        """NOT FINISHED: Solve for the eigenvalues and eigenvectors of a square array"""

        """
        from numpy import linalg as LA

        linalg.eig(a)
        Compute the eigenvalues and right eigenvectors of a square array.
 

        Example:
        eigenvalues, eigenvectors = LA.eig(np.diag((1, 2, 3)))

        eigenvalues
        array([1., 2., 3.])
        
        eigenvectors
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        
        Parameters:
        
            a
            (…, M, M) array
        
                Matrices for which the eigenvalues and right eigenvectors will be computed
        
        Returns:
        
            A namedtuple with the following attributes:
            eigenvalues
            (…, M) array
        
                The eigenvalues, each repeated according to its multiplicity. The eigenvalues are not necessarily ordered. The resulting array will be of complex type, unless the imaginary part is zero in which case it will be cast to a real type. When a is real the resulting eigenvalues will be real (0 imaginary part) or occur in conjugate pairs
            eigenvectors
            (…, M, M) array
        
                The normalized (unit “length”) eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

        """
        eigenvalues, eigenvectors = LA.eig(square_array)

        return eigenvalues, eigenvectors


