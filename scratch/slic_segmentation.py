from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import maskslic as seg

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

path_t2w = '/data/prostate/experiments/Patient 1036/T2W'
path_gt = ['/data/prostate/experiments/Patient 1036/GT_inv/prostate']
label_gt = ['prostate']

# Read the data
t2w_mod = T2WModality()
t2w_mod.read_data_from_path(path_t2w)

print t2w_mod.data_.shape

# Extract the information about the spacing
spacing_itk = t2w_mod.metadata_['spacing']
# Remember that this is in ITK format X, Y, Z and that we are in Y, X, Z
spacing = (spacing_itk[1], spacing_itk[0], spacing_itk[2])

print spacing

# Read the ground-truth
gt_mod = GTModality()
gt_mod.read_data_from_path(label_gt, path_gt)

print gt_mod.data_[0].shape

img = t2w_mod.data_[:, :, 32]
img = (img - np.min(img)) * ((1.) / (np.max(img) - np.min(img)))

roi = gt_mod.data_[0, :, :, 32].astype(bool)

# Make SLIC over-segmentation
segments = seg.slic(img, compactness=1,
                    seed_type='nplace',
                    multichannel=False,
                    convert2lab=False,
                    enforce_connectivity=True,
                    mask=roi, n_segments=50,
                    recompute_seeds=True,
                    plot_examples=True)
