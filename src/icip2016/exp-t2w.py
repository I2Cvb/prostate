# In order to plot some stuff
import matplotlib.pyplot as plt
# In order to manipulate some stuff
import numpy as np
# In order to classifiy some stuff
### Random forest
from sklearn.ensemble import RandomForestClassifier
# In order to quantify the performances of our stuff
### ROC Curve
from sklearn.metrics import roc_curve
### AUC Curve
from sklearn.metrics import roc_auc_score
# Confusion matrix
from sklearn.metrics import confusion_matrix

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# Define the function to compute the Normalised Mean Intensity
def nmi(data):
    # get the minimum 
    #min_data = np.min(data)
    min_data = -1.
    print 'mini: {}'.format(min_data)

    # get the maximum
    #max_data = np.max(data)
    max_data = 1.
    print 'maxi: {}'.format(max_data)

    # find the mean
    mean_data = np.mean(data)
    print 'mean: {}'.format(mean_data)

    # return the nmi
    return mean_data / (max_data - min_data)

# Load the data file from the numpy npz file
data_norm_rician = np.load('../../data/t2w/data_norm_rician.npy')
data_norm_parts = np.load('../../data/t2w/data_norm_parts.npy')
data_norm_gaussian = np.load('../../data/t2w/data_norm_gaussian.npy')
data_norm_fda = np.load('../../data/t2w/data_fdasrsf_norm.npy')
data_t2w_norm = np.load('../../data/t2w/data_raw_norm.npy')
label = np.load('../../data/t2w/label.npy')
patient_sizes = np.load('../../data/t2w/patient_sizes.npy')

print '---> Data loaded'
fig, axes = plt.subplots(nrows=5, ncols=2)
# Make the classification for each patient
nb = 100
global_hist_t2w = np.zeros((nb,))
global_norm_gaussian = np.zeros((nb,))
global_norm_rician = np.zeros((nb,))
global_norm_fda = np.zeros((100,))
global_hist_t2w_cap = np.zeros((nb,))
global_norm_gaussian_cap = np.zeros((nb,))
global_norm_rician_cap = np.zeros((nb,))
global_norm_fda_cap = np.zeros((100,))

# Make the list of the different histogram
list_raw = []
list_gaussian = []
list_rician = []
list_parts = []
list_fda = []
list_raw_cap = []
list_gaussian_cap = []
list_rician_cap = []
list_parts_cap = []
list_fda_cap = []

# Initialise the array in order to store the value of the nmi
nmi_raw = []
nmi_gaussian = []
nmi_rician = []
nmi_fda = []
for pt in xrange(len(patient_sizes)):
    
    # Find the index of the current patients
    if (pt == 0):
        start_idx = 0
        end_idx = patient_sizes[pt]
    else:
        start_idx = np.sum(patient_sizes[0 : pt])
        end_idx = np.sum(patient_sizes[0 : pt + 1])

    ##### RAW DATA #####
    # Compute the histogram for the whole data
    nb_bins = nb
    hist, bin_edges = np.histogram(data_t2w_norm[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[0, 0].plot(bin_edges[0 : -1], hist, label='Patient '+str(pt))
    global_hist_t2w = np.add(global_hist_t2w, hist)
    list_raw.append(hist)

    # Compute the NMI for the raw data
    nmi_raw.append(nmi(data_t2w_norm[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = nb
    sub_data = data_t2w_norm[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[0, 1].plot(bin_edges[0 : -1], hist)
    global_hist_t2w_cap = np.add(global_hist_t2w_cap, hist)
    list_raw_cap.append(hist)

    ##### GAUSSIAN NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = nb
    hist, bin_edges = np.histogram(data_norm_gaussian[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[1, 0].plot(bin_edges[0 : -1], hist)    
    global_norm_gaussian = np.add(global_norm_gaussian, hist)
    list_gaussian.append(hist)

    # Compute the NMI for the gaussian data
    nmi_gaussian.append(nmi(data_norm_gaussian[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = nb
    sub_data = data_norm_gaussian[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[1, 1].plot(bin_edges[0 : -1], hist)
    global_norm_gaussian_cap = np.add(global_norm_gaussian_cap, hist)
    list_gaussian_cap.append(hist)

    ##### PARTS NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = nb
    hist, bin_edges = np.histogram(data_norm_parts[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[2, 0].plot(bin_edges[0 : -1], hist)
    #global_norm_parts = np.add(global_norm_parts, hist)
    list_parts.append(hist)

    # Compute the NMI for the parts data
    # nmi_parts.append(nmi(data_norm_parts[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = nb
    sub_data = data_norm_parts[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[2, 1].plot(bin_edges[0 : -1], hist)
    #global_norm_parts_cap = np.add(global_norm_parts_cap, hist)
    list_parts_cap.append(hist)

    ##### RICIAN NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = nb
    hist, bin_edges = np.histogram(data_norm_rician[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[3, 0].plot(bin_edges[0 : -1], hist)
    global_norm_rician = np.add(global_norm_rician, hist)
    list_rician.append(hist)

    # Compute the NMI for the rician data
    nmi_rician.append(nmi(data_norm_rician[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = nb
    sub_data = data_norm_rician[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[3, 1].plot(bin_edges[0 : -1], hist)
    global_norm_rician_cap = np.add(global_norm_rician_cap, hist)
    list_rician_cap.append(hist)

    ##### FDA SRSF NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = 100
    hist, bin_edges = np.histogram(data_norm_fda[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[4, 0].plot(bin_edges[0 : -1], hist)    
    global_norm_fda = np.add(global_norm_fda, hist)
    list_fda.append(hist)

    # Compute the NMI for the gaussian data
    nmi_fda.append(nmi(data_norm_fda[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = 100
    sub_data = data_norm_fda[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[4, 1].plot(bin_edges[0 : -1], hist)
    global_norm_fda_cap = np.add(global_norm_fda_cap, hist)
    list_fda_cap.append(hist)

print np.min(data_norm_fda)
print np.max(data_norm_fda)

# Decorate the plot with the proper annotations
axes[0, 0].set_ylabel('Probabilities')
axes[0, 0].set_title('PDFs for raw data - Full prostate')

axes[0, 1].set_ylabel('Probabilities')
axes[0, 1].set_title('PDFs for raw data - CaP')

axes[1, 0].set_ylabel('Probabilities')
axes[1, 0].set_title('PDFs for Gaussian normalization - Full prostate')

axes[1, 1].set_ylabel('Probabilities')
axes[1, 1].set_title('PDFs for Gaussian normalization - CaP')

axes[2, 0].set_ylabel('Probabilities')
axes[2, 0].set_title('PDFs for linear normalization by parts - Full prostate')

axes[2, 1].set_ylabel('Probabilities')
axes[2, 1].set_title('PDFs for linear normalization by parts data - CaP')

axes[3, 0].set_ylabel('Probabilities')
axes[3, 0].set_title('PDFs for Rician normalization data - Full prostate')

axes[3, 1].set_ylabel('Probabilities')
axes[3, 1].set_title('PDFs for Rician normalization data - CaP')

axes[4, 0].set_ylabel('Probabilities')
axes[4, 0].set_title('PDFs for FDA SRSF normalization data - Full prostate')

axes[4, 1].set_ylabel('Probabilities')
axes[4, 1].set_title('PDFs for FDA SRSF normalization data - CaP')

sns.plt.savefig('qualitative.png')

# Compute the pca decomposition
from sklearn.decomposition import PCA

# Make the decomposition
### Initialize the pca object
pca_raw = PCA()
pca_gaussian = PCA()
pca_rician = PCA()
pca_parts = PCA()
pca_fda = PCA()
pca_raw_cap = PCA()
pca_gaussian_cap = PCA()
pca_rician_cap = PCA()
pca_parts_cap = PCA()
pca_fda_cap = PCA()

### Make the decomposition
pca_raw.fit(np.array(list_raw).T)
pca_gaussian.fit(np.array(list_gaussian).T)
pca_rician.fit(np.array(list_rician).T)
pca_parts.fit(np.array(list_parts).T)
pca_fda.fit(np.array(list_fda).T)
pca_raw_cap.fit(np.array(list_raw_cap).T)
pca_gaussian_cap.fit(np.array(list_gaussian_cap).T)
pca_rician_cap.fit(np.array(list_rician_cap).T)
pca_parts_cap.fit(np.array(list_parts_cap).T)
pca_fda_cap.fit(np.array(list_fda_cap).T)

# Plot the accumulated eigenvalues
from scipy.integrate import simps

# Full prostate
plt.figure()
plt.plot(np.cumsum(pca_raw.explained_variance_ratio_), marker='o',
         label='Raw data - AUC=' + 
         str(simps(np.cumsum(pca_raw.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_gaussian.explained_variance_ratio_), marker='^',
         label='Gaussian normalization - AUC=' + 
         str(simps(np.cumsum(pca_gaussian.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_parts.explained_variance_ratio_), marker='s',
         label='Linear normalization by parts - AUC=' + 
         str(simps(np.cumsum(pca_parts.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_rician.explained_variance_ratio_), marker='*',
         label='Rician normalization - AUC=' + 
         str(simps(np.cumsum(pca_rician.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_fda.explained_variance_ratio_), marker='*',
         label='FDA SRSF normalization - AUC=' + 
         str(simps(np.cumsum(pca_fda.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.legend(loc='lower right')
plt.xlabel('# eigenvalues')
plt.ylabel('CDF of the eigenvalues')
sns.plt.savefig('quantitative_1.png')

# CaP only
plt.figure()
plt.plot(np.cumsum(pca_raw_cap.explained_variance_ratio_), marker='o',
         label='Raw data - AUC=' + 
         str(simps(np.cumsum(pca_raw_cap.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_gaussian_cap.explained_variance_ratio_), marker='^',
         label='Gaussian normalization - AUC=' + 
         str(simps(np.cumsum(pca_gaussian_cap.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_parts_cap.explained_variance_ratio_), marker='s',
         label='Linear normalization by parts - AUC=' + 
         str(simps(np.cumsum(pca_parts_cap.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_rician_cap.explained_variance_ratio_), marker='*',
         label='Rician normalization - AUC=' + 
         str(simps(np.cumsum(pca_rician_cap.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.plot(np.cumsum(pca_fda_cap.explained_variance_ratio_), marker='*',
         label='FDA SRSF normalization - AUC=' + 
         str(simps(np.cumsum(pca_fda_cap.explained_variance_ratio_),np.linspace(0, 1, 17))))
plt.legend(loc='lower right')
plt.xlabel('# eigenvalues')
plt.ylabel('CDF of the eigenvalues')
sns.plt.savefig('quantitative_2.png')

# # Normalise the histogram
# global_hist_t2w = np.divide(global_hist_t2w, float(len(patient_sizes)))
# global_norm_gaussian = np.divide(global_norm_gaussian, float(len(patient_sizes)))
# global_norm_rician = np.divide(global_norm_rician, float(len(patient_sizes)))

# global_hist_t2w_cap = np.divide(global_hist_t2w_cap, float(len(patient_sizes)))
# global_norm_gaussian_cap = np.divide(global_norm_gaussian_cap, float(len(patient_sizes)))
# global_norm_rician_cap = np.divide(global_norm_rician_cap, float(len(patient_sizes)))

# Compute the entropy
from scipy.stats import entropy

# plt.figure()
# plt.plot(bin_edges[0 : -1], global_hist_t2w, label="No norm - Ent=" + str(entropy(global_hist_t2w)))
# plt.plot(bin_edges[0 : -1], global_norm_gaussian, label="Gaussian norm - Ent=" + str(entropy(global_norm_gaussian)))
# plt.plot(bin_edges[0 : -1], global_norm_rician, label="Rician norm - Ent=" + str(entropy(global_norm_rician)))
# plt.title('Accumulation of the PDFs for prostate')
# plt.legend()
# #plt.show()

# plt.figure()
# plt.plot(bin_edges[0 : -1], global_hist_t2w_cap, label="No norm - Ent=" + str(entropy(global_hist_t2w_cap)))
# plt.plot(bin_edges[0 : -1], global_norm_gaussian_cap, label="Gaussian norm - Ent=" + str(entropy(global_norm_gaussian_cap)))
# plt.plot(bin_edges[0 : -1], global_norm_rician_cap, label="Rician norm - Ent=" + str(entropy(global_norm_rician_cap)))
# plt.title('Accumulation of the PDFs for CaP')
# plt.legend()
# #plt.show()

# print 'Ratio entropy prostate vs cap'
# print 'Raw data: {}'.format(entropy(global_hist_t2w_cap) / entropy(global_hist_t2w))
# print 'Gaussian Normalisation: {}'.format(entropy(global_norm_gaussian_cap) / entropy(global_norm_gaussian))
# print 'Rician Normalisation: {}'.format(entropy(global_norm_rician_cap) / entropy(global_norm_rician))

# print ''
# print 'Information regarding the NMI'
# print 'std of raw NMI: {}'.format(np.std(nmi_raw))
# print 'std of gaussian NMI: {}'.format(np.std(nmi_gaussian))
# print 'std of rician NMI: {}'.format(np.std(nmi_rician))

# fig, axes = plt.subplots(nrows=1, ncols=3)
# axes[0, 0].plot(bin_edges[0 : -1], global_hist_t2w)
# axes[0, 1].plot(bin_edges[0 : -1], global_norm_gaussian)
# axes[0, 2].plot(bin_edges[0 : -1], global_norm_rician)
plt.show()
