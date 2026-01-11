'''
million structure method
'''

class M:
    def __init__(self):
        pass

    ##############################################
    # 1 million structure method functions below #
    ##############################################

    def generate_pool(
        self,
        xyzpool_file,
        nframes,
        distance_indices,
        qvector,
        reference_xyz,
        inelastic=True,
    ):
        """
        Generates the pool file for the 1m structure method.
        creates: pool.npz
        -
        """

        # read xyz trajectory
        print("reading xyz_traj file: %s" % xyzpool_file)
        natoms, comment, atomarray, xyzpool = m.read_xyz_traj(xyzpool_file, nframes)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomarray]

        nind = len(distance_indices)  # number of atoms distances will be calculated for
        dist_arr = np.zeros((nind, nind, nframes))
        for i in range(nframes):
            print("calculating distances for frame %i ..." % i)
            dist_arr[:, :, i] = m.distances_array(xyzpool[distance_indices, :, i])

        # definitions
        compton_array = x.compton_spline(atomic_numbers, qvector)

        # calculate reference IAM curve
        reference_iam = x.iam_calc_compton(
            atomic_numbers, reference_xyz, qvector, inelastic, compton_array
        )

        # calculate pre-molecular IAM term
        aa, bb, cc = x.read_iam_coeffs()
        compton, atomic_total, pre_molecular = self.atomic_pre_molecular(
            atomic_numbers, qvector, aa, bb, cc
        )

        # calculate IAM and PCD for each frame
        qlen = len(qvector)
        pcd_arr = np.zeros((qlen, nframes))
        for i in range(nframes):
            print("calculating IAM and PCD for frame %i" % i)
            ##=#=#=# IAM CALCULATION #=#=#=##
            molecular = np.zeros(qlen)  # total molecular factor
            k = 0
            xyz_ = xyzpool[:, :, i]
            for ii in range(natoms):
                for jj in range(ii + 1, natoms):  # j > i
                    qdij = qvector * LA.norm(xyz_[ii, :] - xyz_[jj, :])
                    molecular += pre_molecular[k, :] * np.sin(qdij) / qdij
                    k += 1
            iam_ = atomic_total + 2 * molecular
            if inelastic:
                iam_ += compton
            ##=#=#=# END IAM CALCULATION #=#=#=##

            ##=#=#=# PCD & CHI2 CALCULATIONS #=#=#=##
            pcd_arr[:, i] = 100 * (iam_ / reference_iam - 1)

        # Finally, save pool.npz
        np.savez(
            "pool.npz",
            natoms=natoms,
            atomarray=atomarray,
            atomic_numbers=atomic_numbers,
            nframes=nframes,
            xyzpool=xyzpool,
            distance_indices=distance_indices,
            dist_arr=dist_arr,
            reference_xyz=reference_xyz,
            reference_iam=reference_iam,
            qvector=qvector,
            inelastic=inelastic,
            pcd_arr=pcd_arr,
        )
        return

    def fit_pool(self, pool_npz_file, target_function, nbins):
        """
        reads pool.npz, then fits to target_function
        """

        # read pool.npz
        data = np.load(pool_npz_file)  # load datafile (.npz)
        natoms = data["natoms"]
        atomarray = data["atomarray"]
        atomic_numbers = data["atomic_numbers"]
        nframes = data["nframes"]
        xyzpool = data["xyzpool"]
        distance_indices = data["distance_indices"]
        dist_arr = data["dist_arr"]
        reference_xyz = data["reference_xyz"]
        reference_iam = data["reference_iam"]
        qvector = data["qvector"]
        inelastic = data["inelastic"]
        pcd_arr = data["pcd_arr"]
        qlen = len(qvector)
        nind = len(distance_indices)

        gaussian_fit = False
        skewed_gaussian_fit = True

        # calculate CHI2 and 1/CHI2 for each structure
        chi2_arr = np.zeros(nframes)
        inv_chi2_arr = np.zeros(nframes)

        print("calculating CHI2 for %i frames..." % nframes)
        for k in range(nframes):

            # chi2
            chi2 = np.sum((pcd_arr[:, k] - target_function) ** 2) / qlen
            chi2_arr[k] = chi2

            # 1 / chi2
            inv_chi2 = 1 / chi2
            inv_chi2_arr[k] = inv_chi2

        ### simply which frame has the lowest chi2?
        print("Lowest chi2 frame...")
        lowest_frame = np.argmin(chi2_arr)
        xyz_lowest = xyzpool[:, :, lowest_frame]
        m.write_xyz("lowest_fit.xyz", "lowest", atomarray, xyz_lowest)
        np.savetxt(
            "pcd_lowest.dat", np.column_stack((qvector, pcd_arr[:, lowest_frame]))
        )
        print("Done. Wrote 'lowest_fit.xyz'.")
        # target_dist_arr =
        # mapd = m.mapd_distances(dist_arr[:, :, lowest_frame], target_dist_arr, bond_print)

        ### Binning and fitting the Gaussian
        # Fitting function
        def gaussian_func(xx, a, x0, sigma):
            return a * np.exp(-((xx - x0) ** 2) / (2 * sigma ** 2))

        # Define the probability density function (PDF) for the skewed normal distribution
        def skewed_normal(x, loc, scale, skew, height):
            normpdf = norm.pdf(x, loc, scale)
            normcdf = norm.cdf(skew * (x - loc) / scale)
            skewed_norm = height * normpdf * normcdf
            np.savetxt("normpdf.dat", np.column_stack((x, normpdf)))
            np.savetxt("normcdf.dat", np.column_stack((x, normcdf)))
            np.savetxt("skewed_norm.dat", np.column_stack((x, skewed_norm)))
            return skewed_norm

        # binning distances.
        ## distances i, j
        mu_arr = np.zeros((nind, nind))

        for i in range(nind):
            for j in range(i + 1, nind):
                r = dist_arr[i, j, :]
                # print("distance %i %i" % (i, j))
                start = np.min(r)
                # print("bin range start: %f" % start)
                end = np.max(r)
                # print("bin range end: %f" % end)
                bins = np.linspace(start, end, nbins, endpoint=True)
                # print("bin separation: %8.6f" % (bins[1] - bins[0]))
                # print(bins)

                inds = np.digitize(r, bins)
                # print(inds)

                # put 1/chi2 into the bins
                inv_chi2_bins = np.zeros(nbins)
                for k in range(nbins):
                    tmp = inv_chi2_arr[inds == k]
                    # ignore bins with low statistics
                    if len(tmp) > 10:
                        # in this way there can be values of 0 in the array, maybe bad for fitting?
                        # I could use append instead ?
                        ## only save the maximum 1/chi2 value in each bin
                        inv_chi2_bins[k] = np.max(tmp)

                # Executing curve_fit on noisy data
                xn = bins
                yn = inv_chi2_bins
                # remove yn = 0 values...
                zero_indices = np.where(yn == 0)[0]
                mask = np.ones(len(yn), dtype=bool)
                mask[zero_indices] = False
                xn = xn[mask]
                yn = yn[mask]
                # create a denser xn to output to file
                sd = np.std(xn)
                xn_dense = np.linspace(
                    xn[0] - sd, xn[-1] + sd, 4 * len(xn), endpoint=True
                )

                # popt returns the best fit values for parameters of the given model (func)
                # print('distance bins:')
                # print(xn)
                # print('1/chi2 per bin:')
                # print(yn)

                if gaussian_fit:
                    # Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 5 and 0 <= c <= 0.5
                    popt, pcov = curve_fit(
                        gaussian_func, xn, yn, bounds=(0, [100.0, 5.0, 0.5])
                    )
                    A = popt[0]
                    mu = popt[1]
                    sigma = popt[2]

                    ym = gaussian_func(xn, A, mu, sigma)
                    mu_arr[i, j] = mu

                elif skewed_gaussian_fit:
                    # Fit the skewed normal distribution to the data
                    # params, _ = curve_fit(skewed_normal, xn, yn)
                    # params, _ = curve_fit(skewed_normal, xn, yn, bounds=([-np.inf, 0.0, -0.001], [np.inf, np.inf, 0.001]))
                    params, _ = curve_fit(
                        skewed_normal,
                        xn,
                        yn,
                        bounds=([0.0, 0.01, -5, 0], [5.5, 20.0, 5, 10]),
                    )
                    loc, scale, skew, height = params
                    print((i, j))
                    print("loc: %10.8f" % loc)
                    print("scale: %10.8f" % scale)
                    print("skewness: %10.8f" % skew)
                    print("height: %10.8f" % height)
                    # ym = skewed_normal(xn, loc, scale, skew)
                    ym = skewed_normal(xn_dense, loc, scale, skew, height)

                    # mean = loc + scale * ((2/np.pi) * np.sqrt(2/np.pi))
                    ind = np.argmax(ym)
                    # peak = xn[ind]
                    peak = xn_dense[ind]
                    print("peak: %10.8f" % peak)
                    mu_arr[i, j] = peak
                else:
                    print("Must choose gaussian_fit or skewed_gaussian_fit")

                np.savetxt("ym_%i_%i.dat" % (i, j), np.column_stack((xn_dense, ym)))
                np.savetxt("yn_%i_%i.dat" % (i, j), np.column_stack((xn, yn)))

        ### replace with MAPD calc of everything (compared to mu)
        print("Finding closest MAPD structure to fitted distances...")
        mapd_best = 1e9
        bond_print = False
        print("mu_arr")
        print(mu_arr)
        np.savetxt("mu_arr.dat", mu_arr, fmt="%10.8f")
        for k in range(nframes):
            mapd = m.mapd_distances(dist_arr[:, :, k], mu_arr, bond_print)
            if mapd < mapd_best:
                mapd_best = mapd
                best_frame = k

        print("closest fit frame: %i" % best_frame)
        print("MAPD: %9.8f" % mapd_best)
        print("best_frame distances:")
        print(dist_arr[:, :, best_frame])
        xyz_best = xyzpool[:, :, best_frame]
        m.write_xyz("best_fit.xyz", "best", atomarray, xyz_best)
        print("Done. Wrote 'best_fit.xyz'.")
        # final distances
        dist_arr_best = m.distances_array(xyz_best)
        np.savetxt("dist_arr_best.dat", dist_arr_best, fmt="%10.8f")
        np.savetxt("pcd_best.dat", np.column_stack((qvector, pcd_arr[:, best_frame])))
        np.savetxt("target_function.dat", np.column_stack((qvector, target_function)))
        return

