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

    # For each slice
    region_feat_vec = np.empty((0, 12), dtype=float)
    label_gt = []
    sl_to_go = []
    all_label_img = []
    for sl in range(size_image[2]):

        data = mod.data_[:, 50:size_image[1]/2, :, sl]

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
        for i in range(np.unique(cl_data).size):

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
        sl_to_go.append(sl)
        print 'index of maximum enhanced image: {}'.format(max_idx_img)
        # Compute the region property for the maximum enhanced region
        # Compute the label image
        label_img = label(cl_image.astype(int))
        intensity_img = data[max_idx_img]

        region = regionprops(label_img, intensity_img)

        # For each region, create a feature vector
        for idx_r, r in enumerate(region):
            # Compute the feature vector
            if r.eccentricity > .7:
                label_img[np.nonzero(label_img == idx_r+1)] = 0
                continue
            if r.equivalent_diameter < 10 or r.equivalent_diameter > 30:
                label_img[np.nonzero(label_img == idx_r+1)] = 0
                continue
            if r.area < 150 or r.area > 500:
                label_img[np.nonzero(label_img == idx_r+1)] = 0
                continue
            feat_vec = np.hstack((r.eccentricity, r.equivalent_diameter,
                                  r.area, r.max_intensity, r.mean_intensity,
                                  np.ravel(r.moments_hu)))

            print feat_vec

            region_feat_vec = np.vstack((region_feat_vec, feat_vec))
            label_gt.append((sl, idx_r))

        all_label_img.append(label_img)

    # Cluster the data again with two clusters this time
    km2 = KMeans(n_clusters=2,
                 n_jobs=-1)
    label_region = km2.fit_predict(region_feat_vec)

    for i, l in enumerate(label_region):
        # get the gt
        gt = label_gt[i]
        all_label_img[gt[0]][np.nonzero(all_label_img[gt[0]] == gt[1] + 1)] = l

    for sl in range(len(all_label_img)):
        plt.figure()
        plt.imshow(all_label_img[sl])
        plt.savefig('{}_image_{}.png'.format(idx_patient, sl))
        plt.figure()
        plt.imshow(mod.data_[10, 50:size_image[1]/2, :, sl])
        plt.savefig('{}_original_{}.png'.format(idx_patient, sl))

    return label_region, label_gt, region_feat_vec


# Loop where we read every patient
for idx_lopo_cv in range(len(id_patient_list)):

    # Read the DCE data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(os.path.join(path_data,
                                             id_patient_list[idx_lopo_cv],
                                             path_dce))

    # Segment the aorta
    lr, lgt, rfv = segmentation_aorta(dce_mod, idx_lopo_cv)
