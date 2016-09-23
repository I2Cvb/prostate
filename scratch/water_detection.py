import numpy as np

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.optimize import curve_fit

from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection

from fdasrsf import srsf_align

path_mrsi = '/data/prostate/experiments/Patient 1036/MRSI/CSI_SE_3D_140ms_16c.rda'


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


def _fit_voigt_water(ppm, spectra):
    """Private function to fit water residual in one spectra.

    Parameters
    ----------
    ppm : ndarray, shape (n_samples, )
        The PPM array.

    spectra : ndarray, shape (n_samples, )
        The spectra on which the water has to be fitted.

    Returns
    -------
    popt : list of float,
        A list of the fitted parameters.

    """

    # Get the value between of the spectra between 4 and 6
    water_limits = (4., 6.)
    sub_ppm = ppm[np.flatnonzero(np.bitwise_and(ppm > water_limits[0],
                                                ppm < water_limits[1]))]
    sub_spectra = spectra[np.flatnonzero(np.bitwise_and(
        ppm > water_limits[0],
        ppm < water_limits[1]))]

    # Define the default parameters
    amp_dft = np.max(sub_spectra) / _voigt_profile(0., 1., 0., 1., 1.)
    popt_default = [amp_dft, 1., 1., 1.]
    # Define the bound
    param_bounds = ([0., 0., 0., 0.], [np.inf, np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(_voigt_profile, sub_ppm, np.real(sub_spectra),
                            p0=popt_default, bounds=param_bounds)
    except RuntimeError:
        popt = popt_default

    return popt

rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

phase_correction = MRSIPhaseCorrection(rda_mod)
rda_mod = phase_correction.transform(rda_mod)


# Process all the different spectra
all_spectra = np.reshape(rda_mod.data_, (rda_mod.data_.shape[0],
                                         rda_mod.data_.shape[1] *
                                         rda_mod.data_.shape[2] *
                                         rda_mod.data_.shape[3])).T
popts = Parallel(n_jobs=-1)(delayed(_fit_voigt_water)(rda_mod.bandwidth_ppm,
                                                      spectra)
                            for spectra in all_spectra)
