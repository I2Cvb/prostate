from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.optimize import curve_fit
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from statsmodels.nonparametric.smoothers_lowess import lowess

from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection
from protoclass.preprocessing import MRSIFrequencyCorrection

from fdasrsf import srsf_align

import matplotlib.pyplot as plt

path_mrsi = '/data/prostate/experiments/Patient 1036/MRSI/CSI_SE_3D_140ms_16c.rda'


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _noise_estimate_spectrum(spectrum, nb_split=20):
    """Private function to estimate the noise in a spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (n_samples)
        Spectrum from which the noise has to be estimated.

    nb_split : int, option (default=20)
        The number of regions splitting each spectrum

    Returns
    -------
    sigma : float,
        The estimate of the noise standard deviation.

    """

    # Check if we will be able to make a split
    nb_elt_out = spectrum.size % nb_split
    if nb_elt_out > 0:
        spectrum = spectrum[:-nb_elt_out]

    # Split the arrays into multiple sections
    sections = np.array(np.split(spectrum, nb_split))

    # Compute the mean and variance for each section
    mean_sec = []
    var_sec = []
    for sec in sections:
        mean_sec.append(np.mean(sec))
        var_sec.append(np.var(sec))

    out = lowess(np.array(var_sec), np.array(mean_sec),
                 frac=.9, it=0)
    mean_reg = out[:, 0]
    var_reg = out[:, 1]

    # Find the value for a zero mean intensity or the nearest to zero
    idx_null_mean = _find_nearest(mean_reg, 0.)

    return np.sqrt(var_reg[idx_null_mean])


def _find_baseline_bxr(spectrum, noise_std=None, A=None, B=None,
                       A_star=5.*10e-9, B_star=1.25, max_iter=30,
                       min_err=10e-6):
    """Private function to estimate the baseline of an MRSI spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (n_samples, )
        The spectrum from which the baseline needs to be estimated.

    noise_std : float, optional (default=None)
        An estimate of the noise standard deviation in the spectrum.

    A : float, optional (default=None)
        The smoothing factor.

    B : float, optional (default=None)
        The negative factor.

    A_star : float, optional (default=5.*10e-9)
        Constant for the smoothing factor

    B_star : float, optional (default=1.25)
        Constant for the negative penalty

    max_iter : int, optional (default=30)
        The maximum of iteration for early stopping.

    min_err : float, optional (default=10e-6)
        Norm of the difference of the baseline between each iteration.

    Returns
    -------
    baseline : ndarray, shape (n_samples, )
        The baseline which was minimizing the cost.

    """

    # Find the standard deviation of the noise in the spectrum if necessary
    if noise_std is None:
        noise_std = _noise_estimate_spectrum(spectrum)
        print 'The noise std was estimated at {}'.format(noise_std)

    # Affect A and B if needed:
    if A is None:
        A = -(spectrum.size ** 4. * A_star) / noise_std

    if B is None:
        B = -B_star / noise_std

    # Initialize the baseline using the median value of the spectrum
    baseline = np.array([np.median(spectrum)] * spectrum.size)
    prev_baseline = spectrum.copy()

    # Compute the initial error and the number of iteration
    err = np.linalg.norm(baseline - prev_baseline)
    it = 0

    # Create the vector
    M0 = np.array([-1 / A] * spectrum.size)
    # Create the Hessian matrix
    D0 = lil_matrix((spectrum.size, spectrum.size))
    D0.setdiag(np.array([2.] * spectrum.size), -2)
    D0.setdiag(np.array([-8.] * spectrum.size), -1)
    D0.setdiag(np.array([12.] * spectrum.size), 0)
    D0.setdiag(np.array([-8.] * spectrum.size), 1)
    D0.setdiag(np.array([2.] * spectrum.size), 2)
    # Change the borders
    D0[0, 0] = 2.
    D0[-1, -1] = 2.
    D0[1, 0] = -4.
    D0[0, 1] = -4.
    D0[-1, -2] = -4.
    D0[-2, -1] = -4.
    D0[1, 1] = 10.
    D0[-2, -2] = 10.

    while True:
        if it > max_iter or err < min_err:
            break

        # Update the different element
        M = M0.copy()
        D = D0.copy()
        prev_baseline = baseline.copy()

        # For each element in the spectrum compute the cost
        for ii in range(spectrum.size):
            if baseline[ii] > spectrum[ii]:
                M[ii] += 2. * B * spectrum[ii] / A
                D[ii, ii] += 2. * B / A

        D = D.tocsr()
        baseline = spsolve(D, M)
        err = np.linalg.norm(baseline - prev_baseline)

        # Increment the number of iteration
        it += 1

        print 'Iteration #{} - Error={}'.format(it, err)

    return baseline


rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

# phase_correction = MRSIPhaseCorrection(rda_mod)
# rda_mod = phase_correction.transform(rda_mod)

freq_correction = MRSIFrequencyCorrection(rda_mod)
rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

# Find the index of interest
x = 9
y = 5
z = 5
ppm_limits = (2.2, 3.5)
idx_int = np.flatnonzero(np.bitwise_and(
    rda_mod.bandwidth_ppm[:, y, x, z] > ppm_limits[0],
    rda_mod.bandwidth_ppm[:, y, x, z] < ppm_limits[1]))

baseline = _find_baseline_bxr(np.real(rda_mod.data_[idx_int, y, x, z]),
                              B_star=10e2, A_star=5*10e-6)
