//=============================================================================
// ����Triclops�Դ���api������������˫ĿͼƬ���������ͼ
// �ؼ������ʹ��˫ĿͼƬ���� TriclopsContext & triclops
//=============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>

#include "triclops.h"
#include "fc2triclops.h"
#include "pnmutils.h"

#define _HANDLE_TRICLOPS_ERROR( function, error ) \
{ \
   if( error != TriclopsErrorOk ) \
   { \
      printf( \
	 "ERROR: %s reported %s.\n", \
	 function, \
	 triclopsErrorToString( error ) ); \
      exit( 1 ); \
   } \
} \

namespace CV=cv;
using namespace std;
namespace FC2 = FlyCapture2;
namespace FC2T = Fc2Triclops;

int main() {
	TriclopsContext triclops;
	TriclopsError error;
	TriclopsImage triclopsImage;
	TriclopsInput   triclopsInput;
	std::string szCalFile("input.cal");
	std::string szInputFile("packedColorImage.pgm");
	std::string szDisparityFile("disparity.pgm");
	
	// ������������ļ�
	error = triclopsGetDefaultContextFromFile(&triclops, const_cast<char *>(szCalFile.c_str()));
	_HANDLE_TRICLOPS_ERROR("triclopsGetDefaultContextFromFile(): "
		"Can't open calibration file",
		error);
	
	// ����˫Ŀ���ͼƬ
	if (!pgmReadToTriclopsInput(szInputFile.c_str(), &triclopsInput))
	{
		printf("pgmReadToTriclopsInput() failed. Can't read '%s'\n", szInputFile.c_str());
		return 1;
	}

	/*
	CV::Mat img_row = CV::imread("packedColorImage.png");
	printf("img_png_size: %d * %d \n",img_row.cols,img_row.rows);
	*/

	// �����������У������ͼ��
	triclopsRectify(triclops, &triclopsInput);
	triclopsSetSubpixelInterpolation(triclops, false);
	triclopsSetDisparity(triclops, 7, 164);
	triclopsSetEdgeMask(triclops, 3);
	triclopsSetEdgeCorrelation(triclops, false);
	triclopsSetSurfaceValidation(triclops, false);
	triclopsSetTextureValidation(triclops, false);

	// Do stereo processing
	error = triclopsStereo(triclops);
	printf("imgSize: %d * %d \n", triclopsInput.ncols,triclopsInput.nrows);
	// Retrieve the disparity image from the context
	error = triclopsGetImage(triclops, TriImg_DISPARITY, TriCam_REFERENCE, &triclopsImage);
	printf("disparityImgsize: %d * %d \n", triclopsImage.ncols, triclopsImage.nrows);
	// Save the disparity image
	error = triclopsSaveImage(&triclopsImage, const_cast<char *>(szDisparityFile.c_str()));
	printf("Wrote disparity image to '%s'\n", szDisparityFile.c_str());

	CV::Mat disparityPicture = CV::imread("disparity.pgm");
	CV::Rect rect = CV::Rect(324/4, 372/4, 228/4, 264/4);
	CV::rectangle(disparityPicture, rect.tl(), rect.br(), CV::Scalar(rand() & 255, rand() & 255, rand() & 255));
	imshow("���Գ���", disparityPicture);

	CV::waitKey(20150901);

	return EXIT_SUCCESS;
}