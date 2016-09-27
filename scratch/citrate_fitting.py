from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.stats import norm

from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection
from protoclass.preprocessing import MRSIFrequencyCorrection
from protoclass.preprocessing import MRSIBaselineCorrection

def _voigt_profile(x, alpha, mu, sigma, gamma):
    """Private function to fit a Voigt profile.

    Parameters
    ----------
    x : ndarray, shape (len(x))
        The input data.

    alpha : float,
        The amplitude factor.

    mu : float,
        The shift of the central value.

    sigma : float,
        sigma of the Gaussian.

    gamma : float,
        gamma of the Lorentzian.

    Returns
    -------
    y : ndarray, shape (len(x), )
        The Voigt profile.

    """

    # Define z
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))

    # Compute the Faddeva function
    w = wofz(z)

    return alpha * (np.real(w)) / (sigma * np.sqrt(2. * np.pi))


def _gaussian_profile(x, alpha, mu, sigma):
    return alpha * norm.pdf(x, loc=mu, scale=sigma)


def _loss_voigt_citrate_4_dof(x, ppm, y):
    signal = _voigt_profile(ppm, x[0], x[1], x[2], x[3])
    signal += _voigt_profile(ppm, x[4], x[1] + x[5], x[6], x[7])
    signal += _voigt_profile(ppm, x[8], x[1] - x[9], x[10], x[11])

    return  signal - y


def _loss_gaussian_citrate_4_dof(x, ppm, y):
    signal = _gaussian_profile(ppm, x[0], x[1], x[2])
    signal += _gaussian_profile(ppm, x[3], x[1] + x[4], x[5])
    signal += _gaussian_profile(ppm, x[6], x[1] - x[7], x[8])

    return  signal - y


def _citrate_voigt_fitting(ppm, spectrum):
    """Private function to fit a mixture of Voigt profile to
    citrate metabolites.

    """
    ppm_limits = (2.30, 2.90)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.08, .17)
    delta_3_bounds = (.08, .17)

    # Define the default shifts
    ppm_cit = np.linspace(mu_bounds[0], mu_bounds[1], num=1000)
    mu_dft = ppm_cit[np.argmax(f(ppm_cit))]
    # Redefine the maximum to avoid to much motion
    mu_bounds = (mu_dft - 0.01, mu_dft + 0.01)
    # Redefine the limit of ppm to use for the fitting
    ppm_interp = np.linspace(mu_dft - .20, mu_dft + 0.20, num=5000)
    delta_2_dft = .14
    delta_3_dft = .14

    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    # Create the vector for the default parameters
    popt_default = [alpha_1_dft, mu_dft, .001, .001,
                    alpha_2_dft, delta_2_dft, .001, .001,
                    alpha_3_dft, delta_3_dft, .001, .001]
    # Define the bounds properly
    param_bounds = ([0., mu_bounds[0], 0., 0.,
                     0., delta_2_bounds[0], 0., 0.,
                     #0., 0., 0.],
                     0., delta_3_bounds[0], 0., 0.],
                    [np.inf, mu_bounds[1], np.inf, np.inf,
                     np.inf, delta_2_bounds[1], np.inf, np.inf,
                     np.inf, delta_3_bounds[1], np.inf, np.inf])

    res_robust = least_squares(_loss_voigt_citrate_4_dof, popt_default,
                               loss='huber', f_scale=.1,
                               bounds=param_bounds,
                               args=(ppm_interp, f(ppm_interp)))

    return res_robust.x


def _citrate_gaussian_fitting(ppm, spectrum):
    """Private function to fit a mixture of Gaussian profile to
    citrate metabolites.

    """
    ppm_limits = (2.30, 2.90)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.10, .16)
    delta_3_bounds = (.10, .16)

    # Define the default shifts
    ppm_cit = np.linspace(mu_bounds[0], mu_bounds[1], num=1000)
    mu_dft = ppm_cit[np.argmax(f(ppm_cit))]
    # Redefine the maximum to avoid to much motion
    mu_bounds = (mu_dft - 0.01, mu_dft + 0.01)
    # Redefine the limit of ppm to use for the fitting
    ppm_interp = np.linspace(mu_dft - .20, mu_dft + 0.20, num=5000)
    delta_2_dft = .14
    delta_3_dft = .14

    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    # Create the vector for the default parameters
    popt_default = [alpha_1_dft, mu_dft, .01,
                    alpha_2_dft, delta_2_dft, .01,
                    alpha_3_dft, delta_3_dft, .01]
    # Define the bounds properly
    param_bounds = ([0., mu_bounds[0], 0.,
                     0., delta_2_bounds[0], 0.,
                     0., delta_3_bounds[0], 0.],
                    [np.inf, mu_bounds[1], np.inf,
                     np.inf, delta_2_bounds[1], 8*10e-2,
                     np.inf, delta_3_bounds[1], 8*10e-2])

    res_robust = least_squares(_loss_gaussian_citrate_4_dof, popt_default,
                               loss='huber', f_scale=1.,
                               bounds=param_bounds,
                               args=(ppm_interp, f(ppm_interp)))

    return res_robust.x

path_mrsi = '/data/prostate/experiments/Patient 1036/MRSI/CSI_SE_3D_140ms_16c.rda'

rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

phase_correction = MRSIPhaseCorrection(rda_mod)
rda_mod = phase_correction.transform(rda_mod)

freq_correction = MRSIFrequencyCorrection(rda_mod)
rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

baseline_correction = MRSIBaselineCorrection(rda_mod)
rda_mod = baseline_correction.fit(rda_mod).transform(rda_mod)

# x = 0
# y = 0
# z = 0

# out = _citrate_fitting(rda_mod.bandwidth_ppm[:, 5, 9, 5],
#                        np.real(rda_mod.data_[:, 5, 9, 5]))
