from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.optimize import curve_fit
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import least_squares

from statsmodels.nonparametric.smoothers_lowess import lowess

from gmr import GMM

from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection
from protoclass.preprocessing import MRSIFrequencyCorrection
from protoclass.preprocessing import MRSIBaselineCorrection

from fdasrsf import srsf_align

import matplotlib.pyplot as plt

path_mrsi = '/data/prostate/experiments/Patient 1036/MRSI/CSI_SE_3D_140ms_16c.rda'


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def _gaussian_profile(x, alpha, mu, sigma):
    return alpha * norm.pdf(x, loc=mu, scale=sigma)


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


def _ch_sp_cr_cit_model(x,
                        mu, sigma_1, gamma_1, alpha_1,
                        delta_2, sigma_2, gamma_2, alpha_2,
                        delta_3, sigma_3, gamma_3, alpha_3,
                        delta_4, sigma_4, gamma_4, alpha_4,
                        delta_5, sigma_5, gamma_5, alpha_5,
                        delta_6, sigma_6, gamma_6, alpha_6):
    """Private function to create the mixute of Voigt profiles."""

    signal = _voigt_profile(x, alpha_1, mu, sigma_1, gamma_1)
    signal += _voigt_profile(x, alpha_2, mu + delta_2, sigma_2, gamma_2)
    signal += _voigt_profile(x, alpha_3, mu - delta_3, sigma_3, gamma_3)
    signal += _voigt_profile(x, alpha_4, mu + delta_4, sigma_4, gamma_4)
    signal += _voigt_profile(x, alpha_5, mu + delta_4 - delta_5, sigma_5,
                             gamma_5)
    signal += _voigt_profile(x, alpha_6, mu + delta_4 - delta_6, sigma_6,
                             gamma_6)

    return signal


def _cit_model(x,
               mu, sigma_1, gamma_1, alpha_1,
               delta_2, sigma_2, gamma_2, alpha_2,
               delta_3, sigma_3, gamma_3, alpha_3):
    """Private function to create the mixute of Voigt profiles."""

    signal = _voigt_profile(x, alpha_1, mu, sigma_1, gamma_1)
    signal += _voigt_profile(x, alpha_2, mu + delta_2, sigma_2, gamma_2)
    signal += _voigt_profile(x, alpha_3, mu - delta_3, sigma_3, gamma_3)

    return signal

def voigt(ppm, x):
    signal = _voigt_profile(ppm, x[3], x[0], x[1], x[2])
    signal += _voigt_profile(ppm, x[7], x[0] + x[4], x[5], x[6])
    signal += _voigt_profile(ppm, x[11], x[0] - x[8], x[9], x[10])
    signal += _voigt_profile(ppm, x[15], x[0] + x[12], x[13], x[14])
    signal += _voigt_profile(ppm, x[19], x[0] + x[12] - x[16], x[17], x[18])
    signal += _voigt_profile(ppm, x[23], x[0] + x[12] - x[20], x[21], x[22])

    return  signal

def ls_voigt(x, ppm, y):
    signal = _voigt_profile(ppm, x[3], x[0], x[1], x[2])
    signal += _voigt_profile(ppm, x[7], x[0] + x[4], x[5], x[6])
    signal += _voigt_profile(ppm, x[11], x[0] - x[8], x[9], x[10])
    signal += _voigt_profile(ppm, x[15], x[0] + x[12], x[13], x[14])
    signal += _voigt_profile(ppm, x[19], x[0] + x[12] - x[16], x[17], x[18])
    signal += _voigt_profile(ppm, x[23], x[0] + x[12] - x[20], x[21], x[22])

    return  signal - y

def gauss(ppm, x):
    signal = _gaussian_profile(ppm, x[2], x[0], x[1])
    signal += _gaussian_profile(ppm, x[5], x[0] + x[3], x[4])
    signal += _gaussian_profile(ppm, x[8], x[0] - x[6], x[7])

    return  signal

def ls_gauss(x, ppm, y):
    signal = _gaussian_profile(ppm, x[2], x[0], x[1])
    signal += _gaussian_profile(ppm, x[5], x[0] + x[3], x[4])
    signal += _gaussian_profile(ppm, x[8], x[0] - x[6], x[7])

    return  signal - y

def _cit_gaussian_model(x,
                        mu, sigma_1, alpha_1,
                        delta_2, sigma_2, alpha_2,
                        delta_3, sigma_3, alpha_3):
    """Private function to create the mixute of Voigt profiles."""

    signal = _gaussian_profile(x, alpha_1, mu, sigma_1)
    #signal += _gaussian_profile(x, alpha_2, mu + delta_2, sigma_2)
    #signal += _gaussian_profile(x, alpha_3, mu - delta_3, sigma_3)

    return signal


def _ch_sp_cr_cit_fitting(ppm, spectrum):
    """Private function to fit a mixture of Voigt profile to
    choline, spermine, creatine, and citrate metabolites.

    """
    ppm_limits = (2.35, 3.25)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define the default shifts
    mu_dft = 2.56
    delta_2_dft = .14
    delta_3_dft = .14
    delta_4_dft = .58
    delta_5_dft = .12
    delta_6_dft = .16
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.08, .17)
    delta_3_bounds = (.08, .17)
    delta_4_bounds = (.55, .61)
    delta_5_bounds = (.11, .13)
    delta_6_bounds = (.13, .17)
    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_4_dft = (f(mu_dft + delta_4_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_5_dft = (f(mu_dft + delta_4_dft - delta_5_dft) /
                   _voigt_profile(0, 1., 0., .001, .001))
    alpha_6_dft = (f(mu_dft + delta_4_dft - delta_6_dft) /
                   _voigt_profile(0, 1., 0., .001, .001))
    # Create the vector for the default parameters
    popt_default = [mu_dft, .001, .001, alpha_1_dft,
                    delta_2_dft, .001, .001, alpha_2_dft,
                    delta_3_dft, .001, .001, alpha_3_dft,
                    delta_4_dft, .001, .001, alpha_4_dft,
                    delta_5_dft, .001, .001, alpha_5_dft,
                    delta_6_dft, .001, .001, alpha_6_dft]
    # Define the bounds properly
    param_bounds = ([mu_bounds[0], 0., 0., 0.,
                     delta_2_bounds[0], 0., 0., 0.,
                     delta_3_bounds[0], 0., 0., 0.,
                     delta_4_bounds[0], 0., 0., 0.,
                     delta_5_bounds[0], 0., 0., 0.,
                     delta_6_bounds[0], 0., 0., 0.],
                    [mu_bounds[1], np.inf, np.inf, np.inf,
                     delta_2_bounds[1], np.inf, np.inf, np.inf,
                     delta_3_bounds[1], np.inf, np.inf, np.inf,
                     delta_4_bounds[1], np.inf, np.inf, np.inf,
                     delta_5_bounds[1], np.inf, np.inf, np.inf,
                     delta_6_bounds[1], np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(_ch_sp_cr_cit_model, ppm_interp,
                            f(ppm_interp),
                            p0=popt_default, bounds=param_bounds)
    except RuntimeError:
        popt = popt_default

    return popt


def _cit_fitting(ppm, spectrum):
    """Private function to fit a mixture of Voigt profile to
    citrate metabolites.

    """
    ppm_limits = (2.35, 2.85)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define the default shifts
    mu_dft = 2.56
    delta_2_dft = .14
    delta_3_dft = .14
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.08, .17)
    delta_3_bounds = (.08, .17)
    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    # Create the vector for the default parameters
    popt_default = [mu_dft, .001, .001, alpha_1_dft,
                    delta_2_dft, .001, .001, alpha_2_dft,
                    delta_3_dft, .001, .001, alpha_3_dft]
    # Define the bounds properly
    param_bounds = ([mu_bounds[0], 0., 0., 0.,
                     delta_2_bounds[0], 0., 0., 0.,
                     delta_3_bounds[0], 0., 0., 0.],
                    [mu_bounds[1], np.inf, np.inf, np.inf,
                     delta_2_bounds[1], np.inf, np.inf, np.inf,
                     delta_3_bounds[1], np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(_cit_model, ppm_interp,
                            f(ppm_interp),
                            p0=popt_default, bounds=param_bounds)
    except RuntimeError:
        popt = popt_default

    return popt


def _cit_gaussian_fitting(ppm, spectrum):
    """Private function to fit a mixture of Voigt profile to
    citrate metabolites.

    """
    ppm_limits = (2.35, 3.25)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define the default shifts
    mu_dft = 2.56
    delta_2_dft = .14
    delta_3_dft = .14
    delta_4_dft = .58
    delta_5_dft = .12
    delta_6_dft = .16
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.12, .16)
    delta_3_bounds = (.12, .16)
    delta_4_bounds = (.55, .61)
    delta_5_bounds = (.11, .13)
    delta_6_bounds = (.13, .17)
    # # Define the default amplitude
    # alpha_1_dft = (f(mu_dft) /
    #                _gaussian_profile(0., 1., 0., .01))
    # alpha_2_dft = (f(mu_dft + delta_2_dft) /
    #                _gaussian_profile(0., 1., 0., .01))
    # alpha_3_dft = (f(mu_dft - delta_3_dft) /
    #                _gaussian_profile(0., 1., 0., .01))
    # # Create the vector for the default parameters
    # popt_default = [mu_dft, .01, alpha_1_dft,
    #                 delta_2_dft, .01, alpha_2_dft,
    #                 delta_3_dft, .01, alpha_3_dft]
    # # Define the bounds properly
    # param_bounds = ([mu_bounds[0], 0., 0.,
    #                  delta_2_bounds[0], 0., 0.,
    #                  delta_3_bounds[0], 0., 0.],
    #                 [mu_bounds[1], np.inf, np.inf,
    #                  delta_2_bounds[1], np.inf, np.inf,
    #                  delta_3_bounds[1], np.inf, np.inf])

    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_4_dft = (f(mu_dft + delta_4_dft) /
                   _voigt_profile(0., 1., 0., .001, .001))
    alpha_5_dft = (f(mu_dft + delta_4_dft - delta_5_dft) /
                   _voigt_profile(0, 1., 0., .001, .001))
    alpha_6_dft = (f(mu_dft + delta_4_dft - delta_6_dft) /
                   _voigt_profile(0, 1., 0., .001, .001))
    # Create the vector for the default parameters
    popt_default = [mu_dft, .001, .001, alpha_1_dft,
                    delta_2_dft, .001, .001, alpha_2_dft,
                    delta_3_dft, .001, .001, alpha_3_dft,
                    delta_4_dft, .001, .001, alpha_4_dft,
                    delta_5_dft, .001, .001, alpha_5_dft,
                    delta_6_dft, .001, .001, alpha_6_dft]
    # Define the bounds properly
    param_bounds = ([mu_bounds[0], 0., 0., 0.,
                     delta_2_bounds[0], 0., 0., 0.,
                     delta_3_bounds[0], 0., 0., 0.,
                     delta_4_bounds[0], 0., 0., 0.,
                     delta_5_bounds[0], 0., 0., 0.,
                     delta_6_bounds[0], 0., 0., 0.],
                    [mu_bounds[1], np.inf, np.inf, np.inf,
                     delta_2_bounds[1], np.inf, np.inf, np.inf,
                     delta_3_bounds[1], np.inf, np.inf, np.inf,
                     delta_4_bounds[1], np.inf, np.inf, np.inf,
                     delta_5_bounds[1], np.inf, np.inf, np.inf,
                     delta_6_bounds[1], np.inf, np.inf, np.inf])

    # # Create the vector for the default parameters
    # popt_default = np.array([mu_dft, .01, alpha_1_dft])
    # # Define the bounds properly
    # param_bounds = (np.array([mu_bounds[0], 0., 0.]),
    #                 np.array([mu_bounds[1], np.inf, np.inf]))
    # try:
    #     popt, _ = curve_fit(_cit_gaussian_model, ppm_interp,
    #                         f(ppm_interp),
    #                         p0=popt_default)#, bounds=param_bounds)
    # except RuntimeError:
    #     popt = popt_default

    res_robust = least_squares(ls_voigt, popt_default,
                               loss='huber', f_scale=.1,
                               bounds=param_bounds,
                               args=(ppm_interp, f(ppm_interp)))

    return res_robust.x


rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

phase_correction = MRSIPhaseCorrection(rda_mod)
rda_mod = phase_correction.transform(rda_mod)

freq_correction = MRSIFrequencyCorrection(rda_mod)
rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

baseline_correction = MRSIBaselineCorrection(rda_mod)
rda_mod = baseline_correction.fit(rda_mod).transform(rda_mod)

x = 9
y = 5
z = 5

# out = _cit_gaussian_fitting(rda_mod.bandwidth_ppm[:, y, x, z],
#                             np.real(rda_mod.data_[:, y, x, z]))


ppm = rda_mod.bandwidth_ppm[:, y, x, z]
spectrum = np.real(rda_mod.data_[:, y, x, z])

ppm_limits = (2.35, 2.85)
idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                        ppm < ppm_limits[1]))
sub_ppm = ppm[idx_ppm]
sub_spectrum = spectrum[idx_ppm]
