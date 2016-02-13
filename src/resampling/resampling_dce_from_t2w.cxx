/*!
 * \file resampling_dce_from_t2w.cxx
 * \brief Pipeline to resample DCE image from T2W image and save DICOM accordingly.
 * \author Guillaume Lemaitre - LE2I ViCOROB
 * \version 0.1
 * \date March, 2016
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
    typedef itk::ImageSeriesWriter< InputImageType, OutputImageType >
	SeriesWriterType;
    typedef itk::StatisticsImageFilter< OutputImageType > 
        StatisticsImageFilterType;
    typedef itk::ExtractImageFilter< InputImageType, OutputImageType > 
        ExtractFilterType;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // READ THE T2W IMAGE TO USE THE SPATIAL INFORMATION FOR THE RESAMPLING
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // List the file inside the directory
    InputNamesGeneratorType::Pointer t2w_filenames_generator = InputNamesGeneratorType::New();
    t2w_filenames_generator->SetInputDirectory(argv[1]);

    // Get the filename corresponding to the first serie
    const ReaderType::FileNamesContainer& t2w_filenames = 
	t2w_filenames_generator->GetFileNames(t2w_filenames_generator->GetSeriesUIDs().begin()->c_str());

    // Load the volume inside the image object
    ImageIOType::Pointer gdcm_t2w = ImageIOType::New();
    ReaderType::Pointer t2w_volume = ReaderType::New();
    t2w_volume->SetImageIO(gdcm_t2w);
    t2w_volume->SetFileNames(t2w_filenames);

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
    std::cout << "The filenames of the T2W serie are:" << std::endl;
    for(auto it = t2w_filenames.begin(); it != t2w_filenames.end(); ++it)
	std::cout << *it << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the T2W DICOM volume:" << std::endl;
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

    // Variable to generate the list of files
    InputNamesGeneratorType::Pointer dce_filenames_generator = InputNamesGeneratorType::New();
    dce_filenames_generator->SetInputDirectory(argv[2]);

    // Check the number of series that are available
    const itk::SerieUIDContainer& dce_serieuid = dce_filenames_generator->GetSeriesUIDs();
    std::cout << "The number of series in the DCE acquisition: " << dce_serieuid.size() << std::endl;

    // Sort the vector of serie to be in proper order
    std::sort(dce_serieuid.begin(), dce_serieuid.end(), comparisonSerieID);
    
    unsigned int nFile = 0;
    for (unsigned int nSerie = 0; nSerie < dce_serieuid.size(); ++nSerie) {	
	
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

	// Let's display some information about the DICOM
	std::cout << "The filenames of the DCE serie are:" << std::endl;
	for(auto it = dce_filenames.begin(); it != dce_filenames.end(); ++it)
	    std::cout << *it << std::endl;
	std::cout << "" << std::endl;
	std::cout << "******************************" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "Information about the DCE DICOM volume:" << std::endl;
	std::cout << "Spacing: " << dce_volume->GetOutput()->GetSpacing() << std::endl;
	std::cout << "Origin:" << dce_volume->GetOutput()->GetOrigin() << std::endl;
	std::cout << "Direction:" << std::endl << dce_volume->GetOutput()->GetDirection() << std::endl;
	std::cout << "Size:" << dce_volume->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
	std::cout << "" << std::endl;
	std::cout << "******************************" << std::endl;
	std::cout << "" << std::endl;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// RESAMPLE THE DCE VOLUME USING THE INFORMATION OF THE T2W VOLUME
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Declare which interpolation and transform are needed
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	TransformType::Pointer transform = TransformType::New();
	transform->SetIdentity();

	// Resample now
	ResampleFilterType::Pointer resampledVolume = ResampleFilterType::New();
	resampledVolume->SetInput(dce_volume->GetOutput());
	resampledVolume->SetTransform(transform);
	resampledVolume->SetInterpolator(interpolator);
	resampledVolume->SetOutputOrigin(t2w_volume->GetOutput()->GetOrigin());
	resampledVolume->SetOutputSpacing(t2w_volume->GetOutput()->GetSpacing());
	resampledVolume->SetOutputDirection(t2w_volume->GetOutput()->GetDirection());
	resampledVolume->SetSize(t2w_volume->GetOutput()->GetLargestPossibleRegion().GetSize());
	resampledVolume->Update();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// CREATE A NEW DICTIONARY USING A MIXED OF T2W AND DCE INFORMATON
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// For each slice create a dictionary
	// Find the number of slice
	InputImageType::SizeType outputSize = resampledVolume->GetOutput()->GetLargestPossibleRegion().GetSize();
  
	// Array where the different dictionaries will be saved
	ReaderType::DictionaryArrayType outputArray;

	for (unsigned int nSlice = 0; nSlice < outputSize[2]; ++nSlice) 
	{
	    // Prepare the type of the extractimagefilter to get statistic
	    ExtractFilterType::Pointer filter3dto2d = ExtractFilterType::New();
	    InputImageType::RegionType dce_slice_region = resampledVolume->GetOutput()->GetLargestPossibleRegion();
	    InputImageType::SizeType dce_slice_size = dce_slice_region.GetSize();
	    dce_slice_size[2] = 0;
	    InputImageType::IndexType dce_slice_start_index = dce_slice_region.GetIndex();
	    dce_slice_start_index[2] = nSlice;
	    InputImageType::RegionType dce_roi;
	    dce_roi.SetSize(dce_slice_size);
	    dce_roi.SetIndex(dce_slice_start_index);
	    filter3dto2d->SetDirectionCollapseToIdentity();
	    filter3dto2d->SetInput(resampledVolume->GetOutput());
	    filter3dto2d->SetExtractionRegion(dce_roi);
	    filter3dto2d->Update();
	    	    
	    StatisticsImageFilterType::Pointer statistic_dce_slice = StatisticsImageFilterType::New();
	    statistic_dce_slice->SetInput(filter3dto2d->GetOutput());
	    statistic_dce_slice->Update();

	    // Get the dictionary of the input slice 
	    ReaderType::DictionaryRawPointer dce_dict = (*(dce_volume->GetMetaDataDictionaryArray()))[0];

	    // Get the dictionary of the T2W slice
	    ReaderType::DictionaryRawPointer t2w_dict = 
		(*(t2w_volume->GetMetaDataDictionaryArray()))[nSlice];
    
	    // Create a the output dictionary
	    ReaderType::DictionaryRawPointer out_dict = new ReaderType::DictionaryType;

	    // Copy the input dictionary to the output dictionary
	    CopyDictionary(*dce_dict, *out_dict);
	    
	    // We need to affect manually
	    // - (0010|0010) - Patient Name
	    // - (0008|0018) - SOP Instance UID
	    // - (0008|1050) - Attending Physician's Name
	    // - (0028|0106) - Smallest Image Pixel Value
	    // - (0028|0107) - Largest Image Pixel Value

	    // Patient Name
	    std::string patient_name = "Anonym^Patient";
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0010|0010", patient_name);

	    // SOP Instance UID
	    gdcm::UIDGenerator sopuid;
	    std::string sopInstanceUID = sopuid.Generate();
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0008|0018", sopInstanceUID);

	    // Attending Physician's Name
	    std::string physician_name = "Anonym^Physician";
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0008|1050", physician_name);

	    // Smallest Image Pixel Value
	    std::string s_pix_val = std::to_string(statistic_dce_slice->GetMinimum());
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0028|0106", s_pix_val);

	    // Largest Image Pixel Value
	    std::string l_pix_val = std::to_string(statistic_dce_slice->GetMaximum());
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0028|0107", l_pix_val);

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
	    std::string image_number = std::to_string(nSlice + 1);
	    itk::EncapsulateMetaData<std::string>(*out_dict,"0020|0013", image_number);

	    // Create an vector with all the tag to recopy
	    std::vector<std::string> tags_t2w_out = {"0020|0032" , "0020|0037)", "0020|0052", "0020|1041", 
						     "0028|0010", "0028|0011", "0028|0030", "0028|1050", 
						     "0028|1051"};

	    for(auto it = tags_t2w_out.begin(); it != tags_t2w_out.end(); ++it) {
		std::string val_dict;
		// Get from the dictionnary of T2W
		itk::ExposeMetaData<std::string>(*t2w_dict, *it, val_dict);
		// Copy to the output dictionnary
		itk::EncapsulateMetaData<std::string>(*out_dict, *it, val_dict);
	    }

	    // Save the dictionary
	    outputArray.push_back(out_dict);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// SAVE THE DATA AS SLICES 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	ResampleFilterType::Pointer resampledVolume2 = ResampleFilterType::New();
	resampledVolume2->SetInput(dce_volume->GetOutput());
	resampledVolume2->SetTransform(transform);
	resampledVolume2->SetInterpolator(interpolator);
	resampledVolume2->SetOutputOrigin(t2w_volume->GetOutput()->GetOrigin());
	resampledVolume2->SetOutputSpacing(t2w_volume->GetOutput()->GetSpacing());
	resampledVolume2->SetOutputDirection(t2w_volume->GetOutput()->GetDirection());
	resampledVolume2->SetSize(t2w_volume->GetOutput()->GetLargestPossibleRegion().GetSize());
	resampledVolume2->Update();
 
	// Make the output directory and generate the file names.
	itksys::SystemTools::MakeDirectory(argv[3]);

	// Generate the file names
	OutputNamesGeneratorType::Pointer outputNames = OutputNamesGeneratorType::New();
	std::string seriesFormat(argv[3]);
	seriesFormat = seriesFormat + "/" + "image%d.dcm";
	outputNames->SetSeriesFormat (seriesFormat.c_str());
	outputNames->SetStartIndex (nFile+1);
	outputNames->SetEndIndex (nFile+outputSize[2]);

	SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
	seriesWriter->SetInput(resampledVolume2->GetOutput());
	// To not generate any new UID
	gdcm_t2w->KeepOriginalUIDOn();
	seriesWriter->SetImageIO(gdcm_t2w);
	seriesWriter->SetFileNames(outputNames->GetFileNames());
	seriesWriter->SetMetaDataDictionaryArray(&outputArray);
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
