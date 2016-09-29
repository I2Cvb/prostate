from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.interpolate import interp1d
from lmfit import minimize
from lmfit import Parameters

from scipy.integrate import simps

# from scipy.optimize import differential_evolution
from scipy.stats import norm
# from scipy.optimize import basinhopping

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


# def _loss_gaussian_citrate_4_dof(x, ppm, y):
#     signal = _gaussian_profile(ppm, x[0], x[1], x[2])
#     signal += _gaussian_profile(ppm, x[3], x[1] + x[4], x[5])
#     signal += _gaussian_profile(ppm, x[6], x[1] - x[7], x[8])

#     return  signal - y




def citrate_model(params, ppm):
    # Define the list of parameters
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    alpha3 = params['alpha3']
    mu1 = params['mu1']
    delta2 = params['delta2']
    delta3 = params['delta3']
    sigma1 = params['sigma1']
    sigma2 = params['sigma2']
    sigma3 = params['sigma3']
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    gamma3 = params['gamma3']


    model = _voigt_profile(ppm, alpha1, mu1, sigma1, gamma1)
    model += _voigt_profile(ppm, alpha2, mu1 + delta2, sigma2, gamma2)
    model += _voigt_profile(ppm, alpha3, mu1 - delta3, sigma3, gamma3)

    return model


def creatine_model(params, ppm, data=None):
    alpha6 = params['alpha6']
    mu1 = params['mu1']
    delta4 = params['delta4']
    delta6 = params['delta6']
    sigma6 = params['sigma6']

    return _gaussian_profile(ppm, alpha6, mu1 + delta4 - delta6, sigma6)


def choline_model(params, ppm, data=None):
    # Define the list of parameters
    alpha4 = params['alpha4']
    mu1 = params['mu1']
    delta4 = params['delta4']
    sigma4 = params['sigma4']

    return _gaussian_profile(ppm, alpha4, mu1 + delta4, sigma4)


def residual_creatine(params, ppm, data=None):
    # Define the list of parameters
    alpha6 = params['alpha6']
    mu1 = params['mu1']
    delta4 = params['delta4']
    delta6 = params['delta6']
    sigma6 = params['sigma6']

    gauss_6 = _gaussian_profile(ppm, alpha6, mu1 + delta4 - delta6, sigma6)

    # Compute the window
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 + delta4 - delta6) -
                                                    2 * sigma6),
                                             ppm < ((mu1 + delta4 - delta6) +
                                                    2 * sigma6)))
    mask[idx_mask] = 1.
    res_6 = ((gauss_6 - data) * mask)# / simps(gauss_6, ppm)

    return res_6


def residual_choline(params, ppm, data=None):
    # Define the list of parameters
    alpha4 = params['alpha4']
    mu1 = params['mu1']
    delta4 = params['delta4']
    sigma4 = params['sigma4']

    gauss_4 = _gaussian_profile(ppm, alpha4, mu1 + delta4, sigma4)

    # Compute the window
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 + delta4) -
                                                    2 * sigma4),
                                             ppm < ((mu1 + delta4) +
                                                    2 * sigma4)))
    mask[idx_mask] = 1.
    res_4 = ((gauss_4 - data) * mask)# / simps(gauss_4, ppm)

    return res_4
    #return (gauss_4 + gauss_5 + gauss_6) - data


def residual(params, ppm, data=None):
    # Define the list of parameters
    alpha1 = np.abs(params['alpha1'])
    alpha2 = np.abs(params['alpha2'])
    alpha3 = np.abs(params['alpha3'])
    mu1 = np.abs(params['mu1'])
    delta2 = np.abs(params['delta2'])
    delta3 = np.abs(params['delta3'])
    sigma1 = np.abs(params['sigma1'])
    sigma2 = np.abs(params['sigma2'])
    sigma3 = np.abs(params['sigma3'])
    gamma1 = np.abs(params['gamma1'])
    gamma2 = np.abs(params['gamma2'])
    gamma3 = np.abs(params['gamma3'])


    gauss_1 = _voigt_profile(ppm, alpha1, mu1, sigma1, gamma1)
    gauss_2 = _voigt_profile(ppm, alpha2, mu1 + delta2, sigma2, gamma2)
    gauss_3 = _voigt_profile(ppm, alpha3, mu1 - delta3, sigma3, gamma3)

    # Compute the window
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > (mu1 - 2 * sigma1),
                                             ppm < (mu1 + 2 * sigma1)))
    mask[idx_mask] = 1.
    res_1 = ((gauss_1 - data) * mask) / simps(gauss_1, ppm)
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 + delta2) -
                                                    2 * sigma2),
                                             ppm < ((mu1 + delta2) +
                                                    2 * sigma2)))
    mask[idx_mask] = 1.
    res_2 = ((gauss_2 - data) * mask) / simps(gauss_2, ppm)
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 - delta3) -
                                                    2 * sigma1),
                                             ppm < ((mu1 - delta3) +
                                                    2 * sigma1)))
    mask[idx_mask] = 1.
    res_3 = ((gauss_3 - data) * mask) / simps(gauss_3, ppm)

    return res_1 + res_2 + res_3


def _citrate_fitting(ppm, spectrum):
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
    delta_2_bounds = (.05, .16)
    delta_3_bounds = (.05, .16)

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
    popt_default = np.array([alpha_1_dft, mu_dft, .001, .001,
                             alpha_2_dft, delta_2_dft, .001, .001,
                             alpha_3_dft, delta_3_dft, .001, .001])
    # Define the list of parameters
    params = Parameters()
    params.add('alpha1', value=alpha_1_dft, min=0.1, max=100)
    params.add('alpha2', value=alpha_2_dft, min=0.1, max=100)
    params.add('alpha3', value=alpha_3_dft, min=0.1, max=100)
    params.add('mu1', value=mu_dft, min=mu_bounds[0], max=mu_bounds[1])
    params.add('delta2', value=delta_2_dft, min=delta_2_bounds[0],
               max=delta_2_bounds[1])
    params.add('delta3', value=delta_3_dft, min=delta_3_bounds[0],
               max=delta_3_bounds[1])
    params.add('sigma1', value=.003, min=.001, max=0.1)
    params.add('sigma2', value=.003, min=.001, max=0.1)
    params.add('sigma3', value=.003, min=.001, max=0.1)
    params.add('gamma1', value=.003, min=.001, max=0.1)
    params.add('gamma2', value=.003, min=.001, max=0.1)
    params.add('gamma3', value=.003, min=.001, max=0.1)


    data = f(ppm_interp)
    res_robust = minimize(residual, params, args=(ppm_interp, ),
                          kws={'data':data}, method='least_squares')

    return res_robust


def _metabolite_fitting(ppm, spectrum):
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
    delta_2_bounds = (.10, .16)
    delta_3_bounds = (.10, .16)
    # Define the default shifts
    ppm_cit = np.linspace(mu_bounds[0], mu_bounds[1], num=1000)
    mu_dft = ppm_cit[np.argmax(f(ppm_cit))]
    # Redefine the maximum to avoid to much motion
    mu_bounds = (mu_dft - 0.04, mu_dft + 0.04)
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
    # popt_default = np.array([alpha_1_dft, mu_dft, .01,
    #                          alpha_2_dft, delta_2_dft, .01,
    #                          alpha_3_dft, delta_3_dft, .01])
    # Define the list of parameters
    params = Parameters()
    params.add('alpha1', value=alpha_1_dft, min=0.1, max=100)
    params.add('alpha2', value=alpha_2_dft, min=0.1, max=100)
    params.add('alpha3', value=alpha_3_dft, min=0.1, max=100)
    params.add('mu1', value=mu_dft, min=mu_bounds[0], max=mu_bounds[1])
    params.add('delta2', value=delta_2_dft, min=delta_2_bounds[0],
               max=delta_2_bounds[1])
    params.add('delta3', value=delta_3_dft, min=delta_3_bounds[0],
               max=delta_3_bounds[1])
    params.add('sigma1', value=.01, min=.01, max=0.1)
    params.add('sigma2', value=.01, min=.01, max=0.05)
    params.add('sigma3', value=.01, min=.01, max=0.05)

    data = f(ppm_interp)
    res_citrate = minimize(residual, params, args=(ppm_interp, ),
                           kws={'data':data}, method='least_squares')
    # res_citrate = minimize(residual, res_citrate.params, args=(ppm_interp, ),
    #                        kws={'data':data}, method='differential_evolution')


    # ppm_limits = (2.90, 3.25)
    mu_dft = res_citrate.params['mu1'].value
    delta_4_bounds = (.55, .59)
    # delta_6_bounds = (.18, .20)

    delta_4_dft = .57
    # delta_6_dft = .19

    print mu_dft + delta_4_dft

    ppm_limits = (mu_dft + delta_4_bounds[0] - .02,
                  mu_dft + delta_4_bounds[1] + .02)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    delta_4_dft = ppm_interp[np.argmax(f(ppm_interp))] - mu_dft

    ppm_limits = (mu_dft + delta_4_dft - .04,
                  mu_dft + delta_4_dft + .04)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    print sub_ppm

    alpha_4_dft = (f(mu_dft + delta_4_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    # alpha_6_dft = (f(mu_dft + delta_4_dft - delta_6_dft) /
    #                _gaussian_profile(0., 1., 0., .01))

    params = Parameters()
    params.add('alpha4', value=alpha_4_dft, min=0.01, max=100)
    params.add('mu1', value=mu_dft, vary=False)
    params.add('delta4', value=delta_4_dft, vary=False)
    #params.add('alpha6', value=alpha_6_dft, min=0.01, max=100)
    #params.add('delta6', value=delta_6_dft, min=delta_6_bounds[0],
    #           max=delta_6_bounds[1])
    params.add('sigma4', value=.01, min=.001, max=0.02)
    #params.add('sigma6', value=.005, min=.0001, max=0.005)

    data = f(ppm_interp)
    # res_choline = minimize(residual_choline, params, args=(ppm_interp, ),
    #                        kws={'data':data}, method='differential_evolution')
    res_choline = minimize(residual_choline, params, args=(ppm_interp, ),
                           kws={'data':data}, method='least_squares')


    # # Restart the optimisation by finding the maximum and setting the summit
    # delta_choline = res_choline.params['delta4']
    # idx_max_ch = np.flatnonzero(np.bitwise_and(ppm_interp >
    #                                            mu_dft - delta_choline,
    #                                            ppm_interp <
    #                                            mu_dft + delta_choline))
    # # Find the maximum associated with the max of the choline
    # print np.max(f(ppm_interp)[idx_max_ch])
    # delta_4_dft = ppm_interp[np.argmax(f(ppm_interp)[idx_max_ch])] - mu_dft
    # print delta_4_dft
    # params.add('delta4', value=delta_4_dft, vary=False)
    # res_choline = minimize(residual_choline, params, args=(ppm_interp, ),
    #                        kws={'data':data}, method='differential_evolution')

    # res_choline = minimize(residual_choline, params,
    #                        args=(ppm_interp, ), kws={'data':data},
    #                        method='least_squares')


    return res_citrate, res_choline


# path_mrsi = '/data/prostate/experiments/Patient 383/MRSI/CSI_SE_3D_140ms_16c.rda'

# rda_mod = RDAModality(1250.)
# rda_mod.read_data_from_path(path_mrsi)

# phase_correction = MRSIPhaseCorrection(rda_mod)
# rda_mod = phase_correction.transform(rda_mod)

# freq_correction = MRSIFrequencyCorrection(rda_mod)
# rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

# baseline_correction = MRSIBaselineCorrection(rda_mod)
# rda_mod = baseline_correction.fit(rda_mod).transform(rda_mod)

# out = _citrate_fitting(rda_mod.bandwidth_ppm[:, 5, 9, 5],
#                        np.real(rda_mod.data_[:, 5, 9, 5]))
