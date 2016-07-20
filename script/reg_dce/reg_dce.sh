#!/bin/bash

# Define the path to the executable
pathToBin='../../bin/./reg_dce';

# Define the path to the data
pathData='/data/prostate/experiments';
# Define the path to the DCE
pathDCE='/DCE'

# Define the path where to save the data
pathDCESave='/DCE_intra_reg'

# For all the patients
for patient in $pathOriginalDCE*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    $pathToBin "$pathData$pathDCE" "pathData$PathDCESave"

done
