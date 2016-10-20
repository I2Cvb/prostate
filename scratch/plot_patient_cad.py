import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np

from skimage import img_as_float
from skimage.color import gray2rgb
from skimage.measure import find_contours
from skimage.io import imshow

from sklearn.externals import joblib

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

# Open the results
path_results = '/data/prostate/results/mp-mri-prostate/exp-6/select-model-percentile/results.pkl'
results = joblib.load(path_results)[3][1]
prob_cancer = results[0][:, 1]

# Load the original data
path_t2w = '/data/prostate/experiments/Patient 1041/T2W'
t2w_mod = T2WModality()
t2w_mod.read_data_from_path(path_t2w)

# Load the ground truth
# Define the path of the ground for the prostate
path_gt = ['/data/prostate/experiments/Patient 1041/GT_inv/prostate',
           '/data/prostate/experiments/Patient 1041/GT_inv/pz',
           '/data/prostate/experiments/Patient 1041/GT_inv/cg',
           '/data/prostate/experiments/Patient 1041/GT_inv/cap']
label_gt = ['prostate', 'pz', 'cg', 'cap']
gt_mod = GTModality()
gt_mod.read_data_from_path(label_gt, path_gt)

# Create an empty volume of the size of the modality data
prob_vol = np.zeros(t2w_mod.data_.shape)
prob_ca = np.zeros(t2w_mod.data_.shape)

# Extract the index of the prostate index
prostate_idx = np.array(gt_mod.extract_gt_data('prostate'))
ca_idx = np.array(gt_mod.extract_gt_data('cap'))

for ii in range(ca_idx.shape[1]):
    coord = ca_idx[:, ii]
    prob_ca[coord[0], coord[1], coord[2]] = 1

# Assign the value in the volume
for ii in range(prob_cancer.size):
    coord = prostate_idx[:, ii]
    prob_vol[coord[0], coord[1], coord[2]] = prob_cancer[ii]

idx_sl = 30

plt.figure()
plt.imshow(t2w_mod.data_[:, :, idx_sl])
plt.savefig('slice.pdf',
            bbox_inches='tight')

plt.figure()
plt.imshow(prob_vol[:, :, idx_sl])
plt.savefig('prob.pdf',
            bbox_inches='tight')

plt.figure()
plt.imshow(prob_ca[:, :, idx_sl])
plt.savefig('cap.pdf',
            bbox_inches='tight')

# Get t2w image
t2w_sl = img_as_float(gray2rgb(t2w_mod.data_[:, :, idx_sl]))
t2w_sl /= np.max(t2w_sl)
plt.figure()
imshow(t2w_sl)
plt.savefig('slice.pdf',
            bbox_inches='tight')
plt.imshow(prob_vol[:, :, idx_sl], cmap=plt.cm.jet, alpha=.3)
contour = find_contours(prob_ca[:, :, idx_sl], 0.9)[0]
plt.plot(contour[:, 1], contour[:, 0])
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.axis('off')
plt.savefig('prob.pdf',
            bbox_inches='tight')
