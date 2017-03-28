//=============================================================================
// 利用双目相机，采集样本库图片，生成数据集，储存为png格式
//=============================================================================


//=============================================================================
// System Includes
//=============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2\opencv.hpp>
//=============================================================================
// PGR Includes
//=============================================================================
#include "triclops.h"
#include "fc2triclops.h"
#include <string>

#define _HANDLE_TRICLOPS_ERROR( description, error )	\
{ \
   if( error != TriclopsErrorOk ) \
   { \
      printf( \
	 "*** Triclops Error '%s' at line %d :\n\t%s\n", \
	 triclopsErrorToString( error ), \
	 __LINE__, \
	 description );	\
      printf( "Press any key to exit...\n" ); \
      getchar(); \
      exit( 1 ); \
   } \
}

namespace FC2 = FlyCapture2;
namespace FC2T = Fc2Triclops;
namespace CV = cv;

// struct containing image needed for processing
struct ImageContainer
{
	FC2::Image unprocessed[2];
	FC2::Image bgru[2];
	FC2::Image packed;
};

enum IMAGE_SIDE
{
	RIGHT = 0, LEFT
};

// configure the connected camera
int configureCamera(FC2::Camera & camera);

// generate triclops context from connected camera
int generateTriclopsContext(FC2::Camera & camera,
	TriclopsContext & triclops);

// capture image from connected camera
int grabImage(FC2::Camera & camera, FC2::Image& rGrabbedImage);

// generate triclops input from grabbed color image 
int generateTriclopsInput(const FC2::Image & grabbedImage,
	ImageContainer   & imageContainer,
	TriclopsInput    & triclopsColorInput , std::string);

int
main(int /* argc */, char** /* argv */)
{
	TriclopsInput triclopsColorInput;
	TriclopsPackedColorImage rectifiedPackedColorImage;
	TriclopsContext triclops;

	FC2::Camera camera;
	FC2::Image grabbedImage;

	camera.Connect();

	// configure camera
	if (configureCamera(camera))
	{
		return EXIT_FAILURE;
	}

	// generate the Triclops context 
	if (generateTriclopsContext(camera, triclops))
	{
		return EXIT_FAILURE;
	}

	for (int i = 1;i <=30;i++) {
		printf("\n采集图像 %d ......",i);
		std::stringstream stream;
		stream << i;
		std::string string_i = stream.str();
		

		// 获取未校正的双目图像
		if (grabImage(camera, grabbedImage))
		{
			return EXIT_FAILURE;
		}

		ImageContainer imageContainer;

		// generate color triclops input from grabbed image
		if (generateTriclopsInput(grabbedImage,
			imageContainer,
			triclopsColorInput,string_i)
			)
		{
			return EXIT_FAILURE;
		}
		// Close the camera
		camera.StopCapture();
	}
	
	camera.Disconnect();

	// clean up context
	TriclopsError te;
	te = triclopsDestroyContext(triclops);
	_HANDLE_TRICLOPS_ERROR("triclopsDestroyContext()", te);
	/*
	CV::Mat picture = CV::imread("packedColorImage.pgm");
	imshow("测试程序", picture);
	CV::waitKey(20150901);
	*/
	return EXIT_SUCCESS;
}

//配置相机
int configureCamera(FC2::Camera & camera)
{

	FC2T::ErrorType fc2TriclopsError;
	FC2T::StereoCameraMode mode = FC2T::TWO_CAMERA;
	fc2TriclopsError = FC2T::setStereoMode(camera, mode);
	if (fc2TriclopsError)
	{
		return FC2T::handleFc2TriclopsError(fc2TriclopsError, "setStereoMode");
	}


	return 0;
}
//配置Triclops环境
int generateTriclopsContext(FC2::Camera & camera,
	TriclopsContext & triclops)
{
	FC2::CameraInfo camInfo;
	FC2::Error fc2Error = camera.GetCameraInfo(&camInfo);
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	FC2T::ErrorType fc2TriclopsError;
	fc2TriclopsError = FC2T::getContextFromCamera(camInfo.serialNumber,
		&triclops);
	if (fc2TriclopsError != FC2T::ERRORTYPE_OK)
	{
		return FC2T::handleFc2TriclopsError(fc2TriclopsError,
			"getContextFromCamera");
	}

	std::string cal_filename = "input.cal";
	TriclopsError te = triclopsWriteCurrentContextToFile(triclops, const_cast<char *>(cal_filename.c_str()));
	if (te != TriclopsErrorOk)
	{
		printf("*** Triclops Error '%s' for call: %s\n",
			triclopsErrorToString(te), "triclopsWriteCurrentContextToFile");
	}

	printf("Calibration File successfully saved at %s", cal_filename);
	return 0;
}
//获取未处理的双目图像grabbedImage
int grabImage(FC2::Camera & camera, FC2::Image& grabbedImage)
{
	FC2::Error fc2Error = camera.StartCapture();
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	fc2Error = camera.RetrieveBuffer(&grabbedImage);
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	return 0;
}

int convertToBGRU(FC2::Image & image, FC2::Image & convertedImage)
{
	FC2::Error fc2Error;
	fc2Error = image.SetColorProcessing(FC2::HQ_LINEAR);
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	fc2Error = image.Convert(FC2::PIXEL_FORMAT_BGRU, &convertedImage);
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	return 0;
}

int generateTriclopsInput(const FC2::Image & grabbedImage,
	ImageContainer  & imageContainer,
	TriclopsInput   & triclopsColorInput,std::string string_i)
{
	FC2::Error fc2Error;
	FC2T::ErrorType fc2TriclopsError;
	FC2::PGMOption pgmOpt;
	FC2::Image * unprocessedImage = imageContainer.unprocessed;
	std::string data_name = "cup_";
	std::string packed_image_png = data_name + "packedImagePNG" + string_i+".png";
	std::string packed_image_pgm = data_name + "packedImagePGM" + string_i+".pgm";
	std::string left_image = data_name + "leftImage" + string_i+".png";
	
	//将拼合在一起的两张图片分出来
	fc2TriclopsError = FC2T::unpackUnprocessedRawOrMono16Image(
		grabbedImage,
		true /*assume little endian*/,
		unprocessedImage[RIGHT], unprocessedImage[LEFT]);
	
	pgmOpt.binaryFile = true;
	/*
	unprocessedImage[RIGHT].Save("rawRightImage.pgm", &pgmOpt);
	unprocessedImage[LEFT].Save("rawLeftImage.pgm", &pgmOpt);
	*/
	FC2::Image * bgruImage = imageContainer.bgru;

	for (int i = 0; i < 2; ++i)
	{
		if (convertToBGRU(unprocessedImage[i], bgruImage[i]))
		{
			return 1;
		}
	}

	FC2::PNGOption pngOpt;
	pngOpt.interlaced = false;
	pngOpt.compressionLevel = 0;
	
	bgruImage[LEFT].Save(left_image.c_str(), &pngOpt);

	FC2::Image & packedColorImage = imageContainer.packed;

	// pack BGRU right and left image into an image
	fc2TriclopsError = FC2T::packTwoSideBySideRgbImage(bgruImage[LEFT],
		bgruImage[RIGHT],
		packedColorImage);

	// Use the row interleaved images to build up a packed TriclopsInput.
	// A packed triclops input will contain a single image with 32 bpp.
	TriclopsError te;
	te = triclopsBuildPackedTriclopsInput(
		grabbedImage.GetCols(),
		grabbedImage.GetRows(),
		packedColorImage.GetStride(),
		(unsigned long)grabbedImage.GetTimeStamp().seconds,
		(unsigned long)grabbedImage.GetTimeStamp().microSeconds,
		packedColorImage.GetData(),
		&triclopsColorInput);


	// the following does not change the size of the image
	// and therefore it PRESERVES the internal buffer!
	packedColorImage.SetDimensions(
		packedColorImage.GetRows(),
		packedColorImage.GetCols(),
		packedColorImage.GetStride(),
		packedColorImage.GetPixelFormat(),
		FC2::NONE);

	packedColorImage.Save(packed_image_png.c_str(), &pngOpt);
	packedColorImage.Save(packed_image_pgm.c_str(), &pgmOpt);

	return 0;
}


