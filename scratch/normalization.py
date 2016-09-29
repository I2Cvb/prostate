from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from scipy.special import wofz
from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection
from protoclass.preprocessing import MRSIFrequencyCorrection
from protoclass.preprocessing import MRSIBaselineCorrection
from protoclass.preprocessing import WaterNormalization
from protoclass.preprocessing import LNormNormalization
from protoclass.extraction import RelativeQuantificationExtraction

path_mrsi = '/data/prostate/experiments/Patient 383/MRSI/CSI_SE_3D_140ms_16c.rda'

rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

phase_correction = MRSIPhaseCorrection(rda_mod)
rda_mod = phase_correction.transform(rda_mod)

freq_correction = MRSIFrequencyCorrection(rda_mod)
rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

baseline_correction = MRSIBaselineCorrection(rda_mod)
rda_mod = baseline_correction.fit(rda_mod).transform(rda_mod)

# normalization = WaterNormalization(rda_mod)
# rda_mod = normalization.fit(rda_mod).normalize(rda_mod)

# normalization = LNormNormalization(rda_mod)
# rda_mod = normalization.fit(rda_mod).normalize(rda_mod)

ext = RelativeQuantificationExtraction(rda_mod)
ext.fit(rda_mod)

# out = _citrate_fitting(rda_mod.bandwidth_ppm[:, 5, 9, 5],
#                        np.real(rda_mod.data_[:, 5, 9, 5]))
