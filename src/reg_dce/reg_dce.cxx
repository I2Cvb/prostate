
#include "itkVersion.h"
 
#include "itkImage.h"
#include "itkStatisticsImageFilter.h"
 
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
 
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
 
#include "itkResampleImageFilter.h"
#include "itkShiftScaleImageFilter.h"
 
#include "itkIdentityTransform.h"
#include "itkLinearInterpolateImageFunction.h"

#include "itkExtractImageFilter.h"
 
#if ITK_VERSION_MAJOR >= 4
#include "gdcmUIDGenerator.h"
#else
#include "gdcm/src/gdcmFile.h"
#include "gdcm/src/gdcmUtil.h"
#endif

#include <algorithm>
#include <string>
#include <cstddef>

// Declare the function to copy DICOM dictionary
static void CopyDictionary(itk::MetaDataDictionary &fromDict, 
			       itk::MetaDataDictionary &toDict);

// Function to be used to sort the serie ID
bool comparisonSerieID(std::string i, std::string j) {
    // We have to extract the number after the last point
    std::size_t found_last_point = i.find_last_of(".");
    // Convert string to integer for the last number of the chain
    int i_int = std::stoi(i.substr(found_last_point+1));
    // We have to extract the number after the last point
    found_last_point = j.find_last_of(".");
    // Convert string to integer for the last number of the chain
    int j_int = std::stoi(j.substr(found_last_point+1));
    return i_int < j_int;
}

int main( int argc, char* argv[] )
{

    const unsigned int InputDimension = 3;
    const unsigned int OutputDimension = 2;
 
    typedef signed short PixelType;
 
    typedef itk::Image< PixelType, InputDimension >
	InputImageType;
    typedef itk::Image< PixelType, OutputDimension >
	OutputImageType;
    typedef itk::ImageSeriesReader< InputImageType >
	ReaderType;
    typedef itk::GDCMImageIO
	ImageIOType;
    typedef itk::GDCMSeriesFileNames
	InputNamesGeneratorType;
    typedef itk::NumericSeriesFileNames
	OutputNamesGeneratorType;

    // Variable to generate the list of files
    InputNamesGeneratorType::Pointer dce_filenames_generator =
	InputNamesGeneratorType::New();
    dce_filenames_generator->SetInputDirectory(argv[1]);

    // Check the number of series that are available
    const itk::SerieUIDContainer& dce_serieuid =
	dce_filenames_generator->GetSeriesUIDs();
    std::cout << "The number of series in the DCE acquisition: "
	      << dce_serieuid.size() << std::endl;

    // Sort the vector of serie to be in proper order
    std::sort(dce_serieuid.begin(), dce_serieuid.end(), comparisonSerieID);

    // We will use the 9th serie of the DCE as fixed volume.
    // The other volumes will be register to this volume.
    const int serie_to_keep = 9;
    const ReaderType::FileNamesContainer& dce_fixed_filenames = 
	    dce_filenames_generator->GetFileNames(dce_serieuid[serie_to_keep]);
    ImageIOType::Pointer gdcm_dce_fixed = ImageIOType::New();
    ReaderType::Pointer dce_vol_fixed = ReaderType::New();
    dce_vol_fixed->SetImageIO(gdcm_dce_fixed);
    dce_vol_fixed->SetFileNames(dce_fixed_filenames);
    // Try to update to catch up any error
    try {
	dce_vol_fixed->Update();
    }
    catch (itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series" << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }
    
    unsigned int nFile = 0;
    for (unsigned int nSerie = 0; nSerie < dce_serieuid.size(); ++nSerie) {
	// Check that we don't have the fixed serie
	if (nSerie == serie_to_keep) continue;
	
        // Container with the different filenames
	const ReaderType::FileNamesContainer& dce_filenames = 
	    dce_filenames_generator->GetFileNames(dce_serieuid[nSerie]);
      
	// Reader corresponding to the actual mask volume 
	ImageIOType::Pointer gdcm_dce = ImageIOType::New();
	ReaderType::Pointer dce_volume = ReaderType::New();
	dce_volume->SetImageIO(gdcm_dce);
        dce_volume->SetFileNames(dce_filenames);

	// Try to update to catch up any error
	try {
	    dce_volume->Update();
	}
	catch (itk::ExceptionObject &excp) {
	    std::cerr << "Exception thrown while reading the series" << std::endl;
	    std::cerr << excp << std::endl;
	    return EXIT_FAILURE;
	}
}
