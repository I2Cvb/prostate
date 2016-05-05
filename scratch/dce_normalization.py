import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

# Define the path for the DCE
path_dce = '/data/prostate/experiments/Patient 383/DCE'

# Define the list of path for the GT
path_gt = ['/data/prostate/experiments/Patient 383/GT_inv/prostate']
# Define the associated list of label for the GT
label_gt = ['prostate']

# Read the DCE
dce_mod = DCEModality()
dce_mod.read_data_from_path(path_dce)

# Read the GT
gt_mod = GTModality()
gt_mod.read_data_from_path(label_gt, path_gt)

# Create the object to normalize the DCE data
dce_norm = StandardTimeNormalization(dce_mod)
# Fit the data to get the normalization parameters
dce_norm.partial_fit_model(dce_mod, ground_truth=gt_mod,
                           cat='prostate')

print dce_norm.model_

# Define the path for the DCE
path_dce = '/data/prostate/experiments/Patient 387/DCE'

# Define the list of path for the GT
path_gt = ['/data/prostate/experiments/Patient 387/GT_inv/prostate']
# Define the associated list of label for the GT
label_gt = ['prostate']

# Read the DCE
dce_mod = DCEModality()
dce_mod.read_data_from_path(path_dce)

# Read the GT
gt_mod = GTModality()
gt_mod.read_data_from_path(label_gt, path_gt)

# Fit the data to get the normalization parameters
dce_norm.fit(dce_mod, ground_truth=gt_mod,
                           cat='prostate')

dce_mod_norm = dce_norm.normalize(dce_mod)
# Plot the figure
plt.figure()
heatmap, bins_heatmap = dce_mod.build_heatmap(np.nonzero(gt_mod.data_[0, :, :, :]))
sns.heatmap(heatmap, cmap='jet')
# plt.plot(dce_norm.shift_idx_, np.arange(0, dce_mod.n_serie_)[::-1] + .5 ,'ro')
# plt.plot(dce_norm.shift_idx_ + dce_norm.rmse,
#          np.arange(0, dce_mod.n_serie_)[::-1] + .5 ,'go')
# plt.plot(dce_norm.shift_idx_ - dce_norm.rmse,
#          np.arange(0, dce_mod.n_serie_)[::-1] + .5 ,'go')
plt.savefig('heatmap.png')
