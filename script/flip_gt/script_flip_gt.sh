#!/bin/sh

#####################################################################
# Define the different path

# Define the path to resample the original DCE image using the resaved T2W images
pathToScript='script_flip_gt.py';

# Path of the original DCE
pathOriginal='/data/prostate/experiments/';

#####################################################################


echo "************************************************************************************"
echo "The binary that will be used is in the path:"
echo
echo $pathToBin
echo 
echo "The goal is to resave and rotate the GT"
echo 
echo "************************************************************************************"
echo " "

# For all the patients
for patient in $pathOriginal*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    python $pathToBin "$pathOriginal$patient_folder"

done
