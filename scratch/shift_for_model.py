import os

import numpy as np

from protoclass.preprocessing import StandardTimeNormalization

path_root = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'


shift_patient = []

# We have to open each npy file
for root, dirs, files in os.walk(path_root):

    # Create the string for the file to read
    for f in files:
        filename = os.path.join(root, f)

        # Load the normalization object
        dce_norm = StandardTimeNormalization.load_from_pickles(filename)

        shift_patient.append(dce_norm.fit_params_['shift-int'])

# Stack the different array vetically
shift_patient = np.vstack(shift_patient)
shift_patient = np.max(shift_patient, axis=0)
