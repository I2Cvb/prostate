#include "itkVersion.h" 

#include "itkImage.h"
#include "itkStatisticsImageFilter.h"
 
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "gdcmUIDGenerator.h"
#include "itkNiftiImageIO.h"
 
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"

#include "itkBSplineTransform.h"
#include "itkLBFGSOptimizer.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"

#include <algorithm>
#include <string>
#include <cstddef>

int main( int argc, char* argv[] )
{
    const unsigned int in_dim = 3; 
    typedef short signed PixelType;
 
    typedef itk::Image< PixelType, in_dim > InImType;
    typedef itk::ImageSeriesReader< InImType > SeriesReader;

    itk::GDCMSeriesFileNames::Pointer gt_t2w_gen =
	itk::GDCMSeriesFileNames::New();
    gt_t2w_gen->SetInputDirectory(argv[1]);
    const SeriesReader::FileNamesContainer& gt_t2w_filenames =
	gt_t2w_gen->GetInputFileNames();

    itk::GDCMImageIO::Pointer gdcm_gt_t2w = itk::GDCMImageIO::New();
    SeriesReader::Pointer gt_t2w = SeriesReader::New();
    gt_t2w->SetImageIO(gdcm_gt_t2w);
    gt_t2w->SetFileNames(gt_t2w_filenames);

    try {
	gt_t2w->Update();
    }
    catch (itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series" << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the gt_dce:" << std::endl;
    std::cout << "Spacing: " << gt_t2w->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << gt_t2w->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:" <<
	std::endl << gt_t2w->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:"
	      << gt_t2w->GetOutput()->GetLargestPossibleRegion().GetSize()
	      << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    typedef itk::ImageFileReader< InImType > FileReader;
    FileReader::Pointer gt_dce = FileReader::New();
    gt_dce->SetFileName(argv[2]);
    gt_dce->Update();

    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the gt_dce:" << std::endl;
    std::cout << "Spacing: " << gt_dce->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << gt_dce->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:"
	      << std::endl << gt_dce->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:"
	      << gt_dce->GetOutput()->GetLargestPossibleRegion().GetSize()
	      << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    const unsigned int space_dim = in_dim;
    const unsigned int spline_order = 3;
    typedef double CoordinateRepType;
 
    typedef itk::BSplineTransform< CoordinateRepType,
				   space_dim, spline_order > TransformType;
    typedef itk::LBFGSOptimizer OptimizerType;
    typedef itk::MeanSquaresImageToImageMetric< InImType,
						InImType > MetricType;
    typedef itk:: LinearInterpolateImageFunction< 
	InImType,
	CoordinateRepType > InterpolatorType;
 
    typedef itk::ImageRegistrationMethod<
	InImType,
	InImType >    RegistrationType;
 
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
    registration->SetFixedImage(gt_t2w->GetOutput());
    registration->SetMovingImage(gt_dce->GetOutput());
 
    InImType::RegionType fixedRegion =
	gt_t2w->GetOutput()->GetBufferedRegion();
    registration->SetFixedImageRegion(fixedRegion);

    TransformType::PhysicalDimensionsType fixed_phy_dim;
    TransformType::MeshSizeType mesh_sz;
    for( unsigned int i = 0; i < in_dim; i++ )
    {
	fixed_phy_dim[i] = gt_t2w->GetOutput()->GetSpacing()[i] *
	    static_cast<double>(
		gt_t2w->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
		- 1 );
    }
    unsigned int nb_nodes = 5;
    mesh_sz.Fill(nb_nodes - spline_order);
    transform->SetTransformDomainOrigin(gt_t2w->GetOutput()->GetOrigin());
    transform->SetTransformDomainPhysicalDimensions(fixed_phy_dim);
    transform->SetTransformDomainMeshSize(mesh_sz);
    transform->SetTransformDomainDirection(
	gt_t2w->GetOutput()->GetDirection());

    typedef TransformType::ParametersType     ParametersType;
 
    const unsigned int nb_params = transform->GetNumberOfParameters();
    ParametersType params(nb_params);
    params.Fill(0.0);
    transform->SetParameters(params);

    //  We now pass the parameters of the current transform as the initial
    //  parameters to be used when the registration process starts.

    registration->SetInitialTransformParameters(transform->GetParameters());

    std::cout << "Intial Parameters = " << std::endl;
    std::cout << transform->GetParameters() << std::endl;

    //  Next we set the parameters of the LBFGS Optimizer.

    optimizer->SetGradientConvergenceTolerance(0.05);
    optimizer->SetLineSearchAccuracy(0.9);
    optimizer->SetDefaultStepLength(.5);
    optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations(1000);

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

    transform->SetParameters(finalParameters);
}
