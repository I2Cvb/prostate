#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkFixedArray.h"
#include "itkFlipImageFilter.h"
#include <vector>
#include "itksys/SystemTools.hxx"
int main( int argc, char* argv[] )
{
    if( argc < 3 )
    {
	std::cerr << "Usage: " << argv[0] <<
	    " DicomDirectory  OutputDicomDirectory" << std::endl;
	return EXIT_FAILURE;
    }

    typedef signed short    PixelType;
    const unsigned int      Dimension = 3;
    typedef itk::Image< PixelType, Dimension >      ImageType;
    typedef itk::ImageSeriesReader< ImageType >     ReaderType;
    typedef itk::GDCMImageIO                        ImageIOType;
    typedef itk::GDCMSeriesFileNames                NamesGeneratorType;

    ImageIOType::Pointer gdcmIO = ImageIOType::New();

    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( argv[1] );
    const ReaderType::FileNamesContainer & filenames =
	namesGenerator->GetInputFileNames();
    unsigned int numberOfFilenames =  filenames.size();
    std::cout << numberOfFilenames << std::endl;

    for(unsigned int fni = 0; fni<numberOfFilenames; fni++)
    {
	std::cout << "filename # " << fni << " = ";
	std::cout << filenames[fni] << std::endl;
    }
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetImageIO( gdcmIO );
    reader->SetFileNames( filenames );

  try
  {
      reader->Update();
  }
  catch (itk::ExceptionObject &excp)
  {
      std::cerr << "Exception thrown while writing the image" << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
  }

  // Flip the image and setup to flip around z
  itk::FixedArray<bool, 3> flipAxes;
  flipAxes[0] = false;
  flipAxes[1] = false;
  flipAxes[2] = true;

  typedef itk::FlipImageFilter <ImageType>
      FlipImageFilterType;
 
  FlipImageFilterType::Pointer flipFilter
      = FlipImageFilterType::New ();
  flipFilter->SetInput(reader->GetOutput());
  flipFilter->SetFlipAxes(flipAxes);

  const char * outputDirectory = argv[2];
  itksys::SystemTools::MakeDirectory( outputDirectory );
  typedef signed short    OutputPixelType;
  const unsigned int      OutputDimension = 2;
  typedef itk::Image< OutputPixelType, OutputDimension >    Image2DType;
  typedef itk::ImageSeriesWriter<
      ImageType, Image2DType >  SeriesWriterType;
  SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
  seriesWriter->SetInput( flipFilter->GetOutput() );
  seriesWriter->SetImageIO( gdcmIO );
  namesGenerator->SetOutputDirectory( outputDirectory );
  seriesWriter->SetFileNames( namesGenerator->GetOutputFileNames() );
  seriesWriter->SetMetaDataDictionaryArray(
      reader->GetMetaDataDictionaryArray() );
  try
  {
      seriesWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
      std::cerr << "Exception thrown while writing the series " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
