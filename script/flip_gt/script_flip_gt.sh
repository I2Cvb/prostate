#!/bin/sh

#####################################################################
# Define the different path

# Define the path to resample the original DCE image using the resaved T2W images
pathToScript='../../bin/./flip_gt';

# Path of the original DCE
pathOriginal='/data/prostate/experiments/';

path_gt='/GT'
path_gt_inv='/GT_inv'

path_prostate='/prostate'
path_cg='/cg'
path_pz='/pz'
path_cap='/cap'

#####################################################################


echo "************************************************************************************"
echo "The binary that will be used is in the path:"
echo
echo $pathToScript
echo 
echo "The goal is to resave and rotate the GT"
echo 
echo "************************************************************************************"
echo " "

# For all the patients
for patient in $pathOriginal*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    # For the prostate GT
    $pathToScript "$pathOriginal$patient_folder$path_gt$path_prostate" "$pathOriginal$patient_folder$path_gt_inv$path_prostate"

    # For the cg
    $pathToScript "$pathOriginal$patient_folder$path_gt$path_cg" "$pathOriginal$patient_folder$path_gt_inv$path_cg"

    # For the pz
    $pathToScript "$pathOriginal$patient_folder$path_gt$path_pz" "$pathOriginal$patient_folder$path_gt_inv$path_pz"

    # For the cap
    $pathToScript "$pathOriginal$patient_folder$path_gt$path_cap" "$pathOriginal$patient_folder$path_gt_inv$path_cap"

done
