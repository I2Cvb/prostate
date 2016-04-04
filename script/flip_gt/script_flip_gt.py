"""Script to flip the DICOM ground-truth."""

import sys
import os
import SimpleITK as sitk
import numpy as np

# Define the different variables for the different paths
path_gt_org = 'GT'
path_gt_rec = 'GT_inv'
path_gt_list = ['prostate', 'cg', 'pz', 'cap']

for org_gt in path_gt_list:

    # Get the patient path
    path_patient = sys.argv[1]
    # Load the data
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(path_patient,
                                                             path_gt_org,
                                                             org_gt))
    reader.SetFileNames(dicom_names)
    vol = reader.Execute()

    # Export into numpy format
    vol_numpy = sitk.GetArrayFromImage(vol)

    # Rotate the image around z
    vol_numpy_rot = vol_numpy.copy()
    for sl in range(vol_numpy.shape[0]):
        vol_numpy_rot[-sl, :, :] = vol_numpy[sl, :, :].copy()

    # Create a SimpleITK image
    vol_rot = sitk.GetImageFromArray(vol_numpy_rot)

    # Copy the meta-information of the image
    vol_rot.CopyInformation(vol)

    # Store the image
    # Create the folder if it is not existing
    if not os.path.exists(os.path.join(path_patient, path_gt_rec, org_gt)):
        os.makedirs(os.path.join(path_patient, path_gt_rec, org_gt))
    # Replace GT by GT_inv in the dicom list of names
    # Convert tuple to list
    dicom_names_list = list(dicom_names)
    for idx_file, filename in enumerate(dicom_names_list):
        dicom_names_list[idx_file] = filename.replace(path_gt_org, path_gt_rec)
    # Save the data
    sitk.WriteImage(vol_rot, dicom_names_list)
