#include "itkVersion.h" 

#include "itkImage.h"
#include "itkStatisticsImageFilter.h"
 
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
 
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"

#include "itkCastImageFilter.h"

#include "itkImageRegistrationMethod.h"
#include "itkBSplineTransform.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkLBFGSOptimizer.h"
 
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
 
    typedef double PixelType;
 
    typedef itk::Image< PixelType, InputDimension >
	InputImageType;
    typedef itk::Image< double, InputDimension >
	ProcessImageType;
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

    std::cout << "The filenames of the 9th DCE are:" << std::endl;
    for(auto it = dce_fixed_filenames.begin(); it != dce_fixed_filenames.end(); ++it)
	std::cout << *it << std::endl;

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

    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the moving DCE volume:" << std::endl;
    std::cout << "Spacing: " << dce_vol_fixed->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << dce_vol_fixed->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:" << std::endl << dce_vol_fixed->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:" << dce_vol_fixed->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    unsigned int nFile = 0;
    for (unsigned int nSerie = 0; nSerie < dce_serieuid.size(); ++nSerie) {
	// Check that we don't have the fixed serie
	if (nSerie == serie_to_keep) continue;
	
        // Container with the different filenames
	const ReaderType::FileNamesContainer& dce_moving_filenames = 
	    dce_filenames_generator->GetFileNames(dce_serieuid[nSerie]);
      
	// Reader corresponding to the actual mask volume 
	ImageIOType::Pointer gdcm_dce_moving = ImageIOType::New();
	ReaderType::Pointer dce_vol_moving = ReaderType::New();
	dce_vol_moving->SetImageIO(gdcm_dce_moving);
        dce_vol_moving->SetFileNames(dce_moving_filenames);

	// Try to update to catch up any error
	try {
	    dce_vol_moving->Update();
	}
	catch (itk::ExceptionObject &excp) {
	    std::cerr << "Exception thrown while reading the series"
		      << std::endl;
	    std::cerr << excp << std::endl;
	    return EXIT_FAILURE;
	}

	std::cout << "" << std::endl;
	std::cout << "******************************" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "Information about the moving DCE volume:" << std::endl;
	std::cout << "Spacing: " << dce_vol_moving->GetOutput()->GetSpacing() << std::endl;
	std::cout << "Origin:" << dce_vol_moving->GetOutput()->GetOrigin() << std::endl;
	std::cout << "Direction:" << std::endl << dce_vol_moving->GetOutput()->GetDirection() << std::endl;
	std::cout << "Size:" << dce_vol_moving->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
	std::cout << "" << std::endl;
	std::cout << "******************************" << std::endl;
	std::cout << "" << std::endl;

	// We have to make the registration of the different image now
	// Define the parameters for the spline transform
	const unsigned int SpaceDimension = InputDimension;
	const unsigned int SplineOrder = 3;
	typedef double CoordinateRepType;

	// Define the transform type
	typedef itk::BSplineTransform<CoordinateRepType,
				      SpaceDimension,
				      SplineOrder> TransformType;
	// Define the optimizer
	typedef itk::LBFGSOptimizer OptimizerType;
	// Define the metric type
	typedef itk::MutualInformationImageToImageMetric<InputImageType,
							 InputImageType> 
	    MetricType;
	// Define the interpolation type
	typedef itk::LinearInterpolateImageFunction<InputImageType,
						    double> InterpolatorType;
	// Define the registration type
	typedef itk::ImageRegistrationMethod<InputImageType,
					     InputImageType>
	    RegistrationType;

	// Create new objects
	MetricType::Pointer metric = MetricType::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	RegistrationType::Pointer registration = RegistrationType::New();

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetInterpolator(interpolator);

	TransformType::Pointer transform = TransformType::New();
	registration->SetTransform(transform);

	// Setup the registration
	registration->SetFixedImage(dce_vol_fixed->GetOutput());
	registration->SetMovingImage(dce_vol_moving->GetOutput());

	// Define the parameters for the spline registration
	TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
	TransformType::MeshSizeType meshSize;
	for(unsigned int i=0; i < InputDimension; ++i)
	{
	    fixedPhysicalDimensions[i] = dce_vol_fixed->GetOutput()->GetSpacing()[i] *
		static_cast<double>(
		    dce_vol_fixed->GetOutput()->GetLargestPossibleRegion().GetSize()[i] - 1 );
	}
	unsigned int numberOfGridNodesInOneDimension = 10;
	meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder);
	transform->SetTransformDomainOrigin(dce_vol_fixed->GetOutput()->GetOrigin());
	transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
	transform->SetTransformDomainMeshSize(meshSize);
	transform->SetTransformDomainDirection(dce_vol_fixed->GetOutput()->GetDirection());

	typedef TransformType::ParametersType ParametersType;
	const unsigned int numberOfParameters =
	    transform->GetNumberOfParameters();
	ParametersType parameters( numberOfParameters );
	parameters.Fill( 0.0 );
	transform->SetParameters( parameters );
 
	//  We now pass the parameters of the current transform as the initial
	//  parameters to be used when the registration process starts.
 
	registration->SetInitialTransformParameters(transform->GetParameters());
 
	std::cout << "Intial Parameters = " << std::endl;
	std::cout << transform->GetParameters() << std::endl;
 
	//  Next we set the parameters of the LBFGS Optimizer.
 
	optimizer->SetGradientConvergenceTolerance( 0.05 );
	optimizer->SetLineSearchAccuracy( 0.9 );
	optimizer->SetDefaultStepLength( .5 );
	optimizer->TraceOn();
	optimizer->SetMaximumNumberOfFunctionEvaluations( 1000 );
 
	std::cout << std::endl << "Starting Registration" << std::endl;
 
	try
	{
	    registration->Update();
	    std::cout << "Optimizer stop condition = "
		      << registration->GetOptimizer()->GetStopConditionDescription()
		      << std::endl;
	}
	catch( itk::ExceptionObject & err )
	{
	    std::cerr << "ExceptionObject caught !" << std::endl;
	    std::cerr << err << std::endl;
	    return EXIT_FAILURE;
	}
 
	OptimizerType::ParametersType finalParameters =
	    registration->GetLastTransformParameters();
 
	std::cout << "Last Transform Parameters" << std::endl;
	std::cout << finalParameters << std::endl;
 
	transform->SetParameters( finalParameters );
    }
}
