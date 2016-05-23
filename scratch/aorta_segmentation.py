import matplotlib as mpl
mpl.use('Agg')

import os

import matplotlib.pyplot as plt
import numpy as np

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from sklearn.cluster import KMeans

from skimage.measure import label
from skimage.measure import regionprops

# Define the path to data
path_data = '/data/prostate/experiments'
# Define the path to DCE
path_dce = 'DCE'
# Define the path to GT
path_gt = 'GT'
# Create the generator
id_patient_list = [name for name in os.listdir(path_data)
                   if os.path.isdir(os.path.join(path_data, name))]

def segmentation_aorta(mod, idx_patient):

    # Define the region of interest of the image
    # The upper-half part of the image will be enough
    # Let's take a slice in the middle of the prostate to try
    size_image = mod.metadata_['size']
    data = mod.data_[:, 50:size_image[1]/2, :, size_image[2]/2]

    # Reshape the data to feed the Kmeans
    data_kmeans = np.reshape(data,
                             (data.shape[0],
                              data.shape[1] *
                              data.shape[2])).T

    # Create the Kmeans object
    n_clusters = 6
    km = KMeans(n_clusters=n_clusters,
                n_jobs=-1)
    # Fit and predict the data
    cl_data = km.fit_predict(data_kmeans)

    # Find which cluster correspond to the highest enhancement signal
    cl_perc = []
    for i in range(n_clusters):

        # Compute the maximum enhancement of the current cluster
        # and find the 90 percentile
        perc = np.percentile(np.max(data_kmeans[cl_data == i], axis=1), 90)
        cl_perc.append(perc)

    # Find the cluster corresponding to the highest 90th percentile
    cl_aorta = np.argmax(cl_perc)
    # Create a binary image
    cl_image = np.reshape([cl_data == cl_aorta], (data.shape[1],
                                                  data.shape[2]))

    # Find the index of the image of the maximum enhancement
    max_idx_img = int(np.median(np.argmax(data_kmeans[cl_data == cl_aorta],
                                          axis=1)))
    print 'index of maximum enhanced image: {}'.format(max_idx_img)
    # Compute the region property for the maximum enhanced region
    # Compute the label image
    label_img = label(cl_image.astype(int))
    intensity_img = data[max_idx_img]

    region = regionprops(label_img, intensity_img)

    # For each region, create a feature vector
    region_feat_vec = np.empty((0, 12), dtype=float)
    for r in region:
        # Compute the feature vector
        feat_vec = np.hstack((r.eccentricity, r.equivalent_diameter, r.area,
                              r.max_intensity, r.mean_intensity,
                              np.ravel(r.moments_hu)))

        region_feat_vec = np.vstack((region_feat_vec, feat_vec))

    # Cluster the data again with two clusters this time
    km2 = KMeans(n_clusters=2,
                 n_jobs=-1)
    label_region = km2.fit_predict(region_feat_vec)
    print label_region
    print np.unique(label_img)

    for i in range(len(region)):
        label_img[label_img == (i+1)] = label_region[i]

    # Save the image
    plt.figure()
    plt.imshow(label_img)
    plt.savefig('image_{}.png'.format(idx_patient))
    plt.figure()
    plt.imshow(intensity_img)
    plt.savefig('original_{}.png'.format(idx_patient))

    return label_img


# Loop where we read every patient
for idx_lopo_cv in range(len(id_patient_list)):

    # Read the DCE data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(os.path.join(path_data,
                                             id_patient_list[idx_lopo_cv],
                                             path_dce))

    # Segment the aorta
    regions = segmentation_aorta(dce_mod, idx_lopo_cv)
