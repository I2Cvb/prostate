#!/bin/sh

#####################################################################
# Define the different path

# Define the path to resample the original DCE image using the resaved T2W images
pathToBin='../../bin/./resampling_dce_from_t2w';

# Path of the original DCE
pathOriginalDCE='/data/prostate/original_data/dijon/';
pathOriginalDCEModality='/Perfusion';
# Path of the resaved T2W
pathResavedT2W='/data/prostate/experiments/';
pathResavedT2WModality='/T2W';
# Path to save the resampled data
pathToSaveDCE='/data/prostate/experiments/';
pathToSaveDCEModality='/DCE';

#####################################################################


echo "************************************************************************************"
echo "The binary that will be used is in the path:"
echo
echo $pathToBin
echo 
echo "The goal is to resave the original DCE using spatial information of the T2W images"
echo 
echo "************************************************************************************"
echo " "

# For all the patients
for patient in $pathOriginalDCE*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    $pathToBin "$pathResavedT2W$patient_folder$pathResavedT2WModality" "$pathOriginalDCE$patient_folder$pathOriginalDCEModality" "$pathToSaveDCE$patient_folder$pathToSaveDCEModality"

done
