/*!
 * \file dicom_read_write.cxx
 * \brief This code read a serie of DICOM images and save it back. It is a routine to enforce the consistency of the coordinate system.
 * \author Guillaume Lemaitre - LE2I ViCOROB
 * \version 0.1
 * \date March, 2014
 */

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
 
#include <itksys/SystemTools.hxx>
 
#if ITK_VERSION_MAJOR >= 4
#include "gdcmUIDGenerator.h"
#else
#include "gdcm/src/gdcmFile.h"
#include "gdcm/src/gdcmUtil.h"
#endif

#include <algorithm>
#include <string>
#include <cstddef>

static void CopyDictionary (itk::MetaDataDictionary &fromDict,
			    itk::MetaDataDictionary &toDict);

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

    // Validate input parameters
    if( argc < 2 )
    {
	std::cerr << "Usage: " 
		  << argv[0]
		  << "Original_dicom_directory Output_dicom_directory"
		  << std::endl;
	return EXIT_FAILURE;
    }
 
 
    /////////////////////////////////////////////////
    // Defintion of the different types used

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
    typedef itk::IdentityTransform< double, InputDimension >
	TransformType;
    typedef itk::LinearInterpolateImageFunction< InputImageType, double >
	InterpolatorType;
    typedef itk::ResampleImageFilter< InputImageType, InputImageType >
	ResampleFilterType;
    typedef itk::ShiftScaleImageFilter< InputImageType, InputImageType >
	ShiftScaleType;
    typedef itk::ImageSeriesWriter< InputImageType, OutputImageType >
	SeriesWriterType;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // READ THE T2W IMAGE TO USE THE SPATIAL INFORMATION FOR THE RESAMPLING
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // List the file inside the directory
    InputNamesGeneratorType::Pointer t2w_filenames_generator = InputNamesGeneratorType::New();
    t2w_filenames_generator->SetInputDirectory( argv[1] );

    // Get the filename corresponding to the first serie
    const ReaderType::FileNamesContainer& t2w_filenames = 
	t2w_filenames_generator->GetFileNames( t2w_filenames_generator->GetSeriesUIDs().begin()->c_str() );

    // Load the volume inside the image object
    ImageIOType::Pointer gdcm_t2w = ImageIOType::New();
    ReaderType::Pointer t2w_volume = ReaderType::New();
    t2w_volume->SetImageIO( gdcm_t2w );
    t2w_volume->SetFileNames( t2w_filenamesOD );

    // Try to update to catch up any error
    try {
	t2w_volume->Update();
    }
    catch (itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series" << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    // Let's display some information about the DICOM
    std::cout << "The filename of the T2W serie are:" << std::endl;
    for(auto it = inputAnnotatedFilenames.begin(); it != inputAnnotatedFilenames.end(); ++it)
	std::cout << *it << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the annotated DICOM volume:" << std::endl;
    std::cout << "Spacing: " << t2w_volume->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << t2w_volume->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:" << std::endl << t2w_volume->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:" << t2w_volume->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // READ THE DCE IMAGES FROM THE DIFFERENT SERIES
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*---- Read the original data ----*/
    // Variable to generate the list of files
    InputNamesGeneratorType::Pointer inputFilenamesGenerator = InputNamesGeneratorType::New();
    inputFilenamesGenerator->SetInputDirectory( argv[1] );

    // Check the number of series that are available
    const itk::SerieUIDContainer& serieContainer = inputFilenamesGenerator->GetSeriesUIDs();
    std::cout << "The number of series present is: " << serieContainer.size() << std::endl;

    // Sort the vector of serie to be in proper order
    std::sort(serieContainer.begin(), serieContainer.end(), comparisonSerieID);
    
    unsigned int nFile = 0;
    for ( unsigned int nSerie = 0 ; nSerie < serieContainer.size() ; nSerie++ ) {
	// Container with the different filenames
	const ReaderType::FileNamesContainer& inputFilenames = 
	    inputFilenamesGenerator->GetFileNames( serieContainer[nSerie] );
      
	std::cout << "Serie UID: " << serieContainer[nSerie] << std::endl;
	for(auto it = inputFilenames.begin(); it != inputFilenames.end(); ++it)
	    std::cout << *it << std::endl;

	// Reader corresponding to the actual mask volume 
	ImageIOType::Pointer gdcmIO = ImageIOType::New();
	ReaderType::Pointer inputVolume = ReaderType::New();
	inputVolume->SetImageIO( gdcmIO );
	inputVolume->SetFileNames( inputFilenames );

	// Try to update to catch up any error
	try {
	    inputVolume->Update();
	}
	catch (itk::ExceptionObject &excp) {
	    std::cerr << "Exception thrown while reading the series" << std::endl;
	    std::cerr << excp << std::endl;
	    return EXIT_FAILURE;
	}

	// Let's display some information about the DICOM
	std::cout << "Information about the mask volume:" << std::endl;
	std::cout << "Spacing: " << inputVolume->GetOutput()->GetSpacing() << std::endl; 
	std::cout << "Origin:" << inputVolume->GetOutput()->GetOrigin() << std::endl;
	std::cout << "Direction:" << std::endl << inputVolume->GetOutput()->GetDirection() << std::endl;  
	std::cout << "Size:" << inputVolume->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;

	// Resample the volume now
	/* ----------------------------------------------------------------------*/
	/*----- Map the annotated volume to the serie of the volume to map ------*/
	/* ----------------------------------------------------------------------*/

	// Declare which interpolation and transform are needed
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	TransformType::Pointer transform = TransformType::New();
	transform->SetIdentity();

	// Resample now
	ResampleFilterType::Pointer resampledVolume = ResampleFilterType::New();
	resampledVolume->SetInput( inputVolume->GetOutput() );
	resampledVolume->SetTransform( transform );
	resampledVolume->SetInterpolator( interpolator );
	resampledVolume->SetOutputOrigin( inputAnnotatedVolume->GetOutput()->GetOrigin() );
	resampledVolume->SetOutputSpacing( inputAnnotatedVolume->GetOutput()->GetSpacing() );
	resampledVolume->SetOutputDirection( inputAnnotatedVolume->GetOutput()->GetDirection() );
	resampledVolume->SetSize( inputAnnotatedVolume->GetOutput()->GetLargestPossibleRegion().GetSize() );
	resampledVolume->Update();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create a Metadata dictionary for the output. Only the origin and direction should be different.

	// For each slice create a dictionary
	// Find the number of slice
	InputImageType::SizeType outputSize = resampledVolume->GetOutput()->GetLargestPossibleRegion().GetSize();
  
	// Array where the different dictionaries will be saved
	ReaderType::DictionaryArrayType outputArray;

	for ( unsigned int nSlice = 0; nSlice < outputSize[2] ; nSlice++ ) {

	    // Prepare the type of the extractimagefilter to get statistic
	    typedef itk::ExtractImageFilter< InputImageType, OutputImageType > ExtractFilterType;
	    ExtractFilterType::Pointer filter3dto2d = ExtractFilterType::New();
	    InputImageType::RegionType inputRegion = resampledVolume->GetOutput()->GetLargestPossibleRegion();
	    InputImageType::SizeType size2d = inputRegion.GetSize();
	    size2d[2] = 0;
	    InputImageType::IndexType start2d = inputRegion.GetIndex();
	    start2d[2] = nSlice;
	    InputImageType::RegionType desiredRegion;
	    desiredRegion.SetSize(  size2d  );
	    desiredRegion.SetIndex( start2d );
	    filter3dto2d->SetDirectionCollapseToIdentity();
	    filter3dto2d->SetInput( resampledVolume->GetOutput() );
	    filter3dto2d->SetExtractionRegion( desiredRegion );
	    filter3dto2d->Update();
	    
	    typedef itk::StatisticsImageFilter< OutputImageType > StatisticsImageFilterType;
	    StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
	    statisticsImageFilter->SetInput(filter3dto2d->GetOutput());
	    statisticsImageFilter->Update();

	    std::cout << "Mean: " << statisticsImageFilter->GetMean() << std::endl;
	    std::cout << "Std.: " << statisticsImageFilter->GetSigma() << std::endl;
	    std::cout << "Min: " << statisticsImageFilter->GetMinimum() << std::endl;
	    std::cout << "Max: " << statisticsImageFilter->GetMaximum() << std::endl;

	    // Get the dictionary of the input slice 
	    ReaderType::DictionaryRawPointer inputDict = (*(inputVolume->GetMetaDataDictionaryArray()))[0];

	    // Get the dictionary of the T2W slice
	    ReaderType::DictionaryRawPointer inputT2WDict = 
		(*(inputAnnotatedVolume->GetMetaDataDictionaryArray()))[nSlice];
    
	    // Create a the output dictionary
	    ReaderType::DictionaryRawPointer outputDict = new ReaderType::DictionaryType;

	    // Copy the input dictionary to the output dictionary
	    CopyDictionary( *inputDict, *outputDict );

	    // String that will be used to change each necessary field of the output dictionary
	    itksys_ios::ostringstream value;
	    
	    // We need to affect manually
	    // - (0010|0010) - Patient Name
	    // - (0008|0018) - SOP Instance UID
	    // - (0008|1050) - Attending Physician's Name
	    // - (0028|0106) - Smallest Image Pixel Value
	    // - (0028|0107) - Largest Image Pixel Value

	    // Patient Name
	    value.str("Anonym^Patient");
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0010|0010", value.str());

	    // SOP Instance UID
	    gdcm::UIDGenerator sopuid;
	    std::string sopInstanceUID = sopuid.Generate();
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0008|0018", sopInstanceUID);

	    // Attending Physician's Name
	    value.str("Anonym^Physician");
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0008|1050", value.str());

	    // Smallest Image Pixel Value
	    std::string s_pix_val = statisticsImageFilter->GetMinimum();
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0028|0106", s_pix_val);

	    // Largest Image Pixel Value
	    std::string l_pix_val = statisticsImageFilter->GetMaximum();
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0018|0107", l_pix_val);

	    // We need to keep the following element from the T2W serie
	    // - (0020|0013) - Image Number
	    // - (0020|0032) - Image Position
	    // - (0020|0037) - Image Orientation
	    // - (0020|0052) - Frame of Reference UID
	    // - (0020|1041) - Slice Location
	    // - (0028|0010) - Rows
	    // - (0028|0011) - Columns
	    // - (0028|0030) - Pixel Spacing
	    // - (0028|1050) - Window Center
	    // - (0028|1051) - Window Width
	    
	    // Image Number
	    value.str("");
	    value << nSlice + 1;
	    itk::EncapsulateMetaData<std::string>(*outputDict,"0020|0013", value.str());

	    // Create an vector with all the tag to recopy
	    std::vector<std::string> tags_t2w_out = { "0020|0032" , "0020|0037)", "0020|0052", "0020|1041", 
						      "0028|0010", "0028|0011", "0028|0030", "0028|1050", 
						      "0028|1051"};

	    for(auto it = tags_t2w_out.begin(); it != tags_t2w_out.end(); ++it) {
		std::string val_dict;
		// Get from the dictionnary of T2W
		itk::ExposeMetaData<std::string>(*inputT2WDict, *it, val_dict);
		// Copy to the output dictionnary
		itk::EncapsulateMetaData<std::string>(*outputDict, *it, val_dict);
	    }

	    std::string val_dict;
	    itk::ExposeMetaData<std::string>(*inputT2WDict, "0020|000e", val_dict);
	    std::cout << val_dict << std::endl;

	    // Save the dictionary
	    outputArray.push_back(outputDict);
	}


	//////////////////////////////////////////////////////////////
	// Let's save the data again with GDCM standard orientation

	ResampleFilterType::Pointer resampledVolume2 = ResampleFilterType::New();
	resampledVolume2->SetInput( inputVolume->GetOutput() );
	resampledVolume2->SetTransform( transform );
	resampledVolume2->SetInterpolator( interpolator );
	resampledVolume2->SetOutputOrigin( inputAnnotatedVolume->GetOutput()->GetOrigin() );
	resampledVolume2->SetOutputSpacing( inputAnnotatedVolume->GetOutput()->GetSpacing() );
	resampledVolume2->SetOutputDirection( inputAnnotatedVolume->GetOutput()->GetDirection() );
	resampledVolume2->SetSize( inputAnnotatedVolume->GetOutput()->GetLargestPossibleRegion().GetSize() );
	resampledVolume2->Update();
 
	// Make the output directory and generate the file names.
	itksys::SystemTools::MakeDirectory( argv[3] );

	// Generate the file names
	OutputNamesGeneratorType::Pointer outputNames = OutputNamesGeneratorType::New();
	std::string seriesFormat(argv[3]);
	seriesFormat = seriesFormat + "/" + "image%d.dcm";
	outputNames->SetSeriesFormat (seriesFormat.c_str());
	outputNames->SetStartIndex (nFile+1);
	outputNames->SetEndIndex (nFile+outputSize[2]);

	SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
	seriesWriter->SetInput( resampledVolume2->GetOutput() );
	gdcmIO->KeepOriginalUIDOn();
	seriesWriter->SetImageIO( gdcmIO );
	seriesWriter->SetFileNames( outputNames->GetFileNames() );
	seriesWriter->SetMetaDataDictionaryArray( &outputArray );
	try {
	    seriesWriter->Update();
	}
	catch( itk::ExceptionObject & excp ) {
	    std::cerr << "Exception thrown while writing the series " << std::endl;
	    std::cerr << excp << std::endl;
	    return EXIT_FAILURE;
	}


	std::cout << "Wrote one serie. Go to the next one !!!" << std::endl;

	nFile = nFile+outputSize[2];      
    }

    std::cout << "Rewriting completed" << std::endl;

    ///////////////////////////////////////////////////
    // Return success if everything was fine
    return EXIT_SUCCESS;
}
 
void CopyDictionary (itk::MetaDataDictionary &fromDict, itk::MetaDataDictionary &toDict)
{
    typedef itk::MetaDataDictionary DictionaryType;
 
    DictionaryType::ConstIterator itr = fromDict.Begin();
    DictionaryType::ConstIterator end = fromDict.End();
    typedef itk::MetaDataObject< std::string > MetaDataStringType;
 
    while( itr != end )
    {
	itk::MetaDataObjectBase::Pointer  entry = itr->second;
 
	MetaDataStringType::Pointer entryvalue = 
	    dynamic_cast<MetaDataStringType *>( entry.GetPointer() ) ;
	if( entryvalue )
	{
	    std::string tagkey   = itr->first;
	    std::string tagvalue = entryvalue->GetMetaDataObjectValue();
	    itk::EncapsulateMetaData<std::string>(toDict, tagkey, tagvalue); 
	    //std::cout << tagkey << " " << tagvalue << std::endl;
	}
	++itr;
    }
}
