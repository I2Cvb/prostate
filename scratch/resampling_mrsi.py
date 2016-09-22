from __future__ import division

import numpy as np
import SimpleITK as sitk

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt

from protoclass.data_management import T2WModality
from protoclass.data_management import RDAModality

path_rda = '/data/prostate/experiments/Patient 996/MRSI/CSI_SE_3D_140ms_16c.rda'
path_t2w = '/data/prostate/experiments/Patient 996/T2W'

# Read the ground-truth
t2w_mod = T2WModality()
t2w_mod.read_data_from_path(path_t2w)

# Read the rda
rda_mod = RDAModality(1250.)
rda_mod.read_data_from_path(path_rda)

# Get the sitk image from the T2W
# We need to convert from numpy array to ITK
# Our convention was Y, X, Z
# We need to convert it in Z, Y, X which will be converted in X, Y, Z by ITK
t2w_img = sitk.GetImageFromArray(np.swapaxes(
    np.swapaxes(t2w_mod.data_, 0, 1), 0, 2))
# Put all the spatial information
t2w_img.SetDirection(t2w_mod.metadata_['direction'])
t2w_img.SetOrigin(t2w_mod.metadata_['origin'])
t2w_img.SetSpacing(t2w_mod.metadata_['spacing'])

# Get the sitk image from the rda
rda_fake = np.random.randint(0, 255, size=(16, 16, 16))
rda_img = sitk.GetImageFromArray(rda_fake)
# Put all the spatial information
rda_img.SetDirection(rda_mod.metadata_['direction'])
rda_img.SetOrigin(rda_mod.metadata_['origin'])
rda_img.SetSpacing(rda_mod.metadata_['spacing'])

# Create a resampler object
transform = sitk.Transform()
transform.SetIdentity()

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(t2w_img)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(transform)
resampler.SetOutputOrigin(rda_img.GetOrigin())
resampler.SetOutputDirection(rda_img.GetDirection())

res_img = resampler.Execute(t2w_img)

# Compute the distance of the X, Y, Z to make a croping of the ROI
size_x = int(rda_img.GetSize()[0] * (rda_img.GetSpacing()[0] /
                                     t2w_img.GetSpacing()[0]))
size_y = int(rda_img.GetSize()[1] * (rda_img.GetSpacing()[1] /
                                     t2w_img.GetSpacing()[1]))
size_z = int(rda_img.GetSize()[2] * (rda_img.GetSpacing()[2] /
                                     t2w_img.GetSpacing()[2]))

out_np = sitk.GetArrayFromImage(res_img)
out_np = out_np[:size_z, :size_y, :size_x]

int_vol = np.zeros((rda_img.GetSize()))

for z in range(rda_img.GetSize()[0]):
    for x in range(rda_img.GetSize()[1]):
        for y in range(rda_img.GetSize()[2]):
            int_vol[y, x, z] = np.sum(np.real(rda_mod.data_[:, y, x, z]))

for z in range(rda_img.GetSize()[0]):
    int_vol /= np.sum(int_vol)

plt.figure()
plt.imshow(int_vol[:, :, 5])
plt.show()
