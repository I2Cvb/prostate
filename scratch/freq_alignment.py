import numpy as np

from protoclass.data_management import RDAModality
from protoclass.preprocessing import MRSIPhaseCorrection

from fdasrsf import srsf_align

path_mrsi = '/data/prostate/experiments/Patient 1036/MRSI/CSI_SE_3D_140ms_16c.rda'

rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_mrsi)

phase_correction = MRSIPhaseCorrection(rda_mod)
rda_mod = phase_correction.transform(rda_mod)

# Get the data
data = np.reshape(rda_mod.data_, (rda_mod.data_.shape[0],
                                  rda_mod.data_.shape[1] *
                                  rda_mod.data_.shape[2] *
                                  rda_mod.data_.shape[3]))

# Apply the curve alignment using the FDS-SRSF
out = srsf_align(np.real(data), rda_mod.bandwidth_ppm, smoothdata=False)
