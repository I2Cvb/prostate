#!/bin/sh

#####################################################################
# Define the different path

# Define the path to resample the original DCE image using the resaved T2W images
pathToBin='../../bin/./resampling_dce_from_t2w';

# Path of the original DCE
pathOriginalDCE='/data/prostate/original_data/';
pathOriginalDCEModality='Perfusion/';
# Path of the resaved T2W
pathResavedT2W='/data/prostate/experiments/';
pathResavedT2WModality='T2W/';
# Path to save the resampled data
pathToSaveDCE='/data/prostate/experiments/';
pathToSaveDCEModality= 'DCE/';

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
    
    # Show which patient will be processed
    echo 
    echo "Patient folder to be processed: "
    echo $patient
    echo 

done
