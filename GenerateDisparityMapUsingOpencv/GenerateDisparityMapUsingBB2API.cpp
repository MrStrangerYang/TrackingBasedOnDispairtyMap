//=============================================================================
// 利用Triclops自带的api，读入样本库双目图片，生成深度图
// 关键是如何使用双目图片构建 TriclopsContext & triclops
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

	const string sequence_name = "cup";
	string packed_image_format = sequence_name + "_packedImagePGM" + "%d.pgm";
	string disp_img_format = sequence_name + "_dispImg" + "%d.pgm";
	const int startFrame = 1;
	const int endFrame = 30;
	for (int i = startFrame;i <= endFrame;i++) {

		char packed_image_name[256];
		char disp_img_name[256];
		sprintf(packed_image_name, packed_image_format.c_str(), i);
		sprintf(disp_img_name, disp_img_format.c_str(), i);

		// 加载相机配置文件
		triclopsGetDefaultContextFromFile(&triclops, const_cast<char *>(szCalFile.c_str()));


		// 加载双目相机图片
		if (!pgmReadToTriclopsInput(packed_image_name, &triclopsInput))
		{
			printf("pgmReadToTriclopsInput() failed. Can't read '%s'\n", packed_image_name);
			return 1;
		}

		/*
		CV::Mat img_row = CV::imread("packedColorImage.png");
		printf("img_png_size: %d * %d \n",img_row.cols,img_row.rows);
		*/

		// 根据相机参数校正输入图像
		triclopsRectify(triclops, &triclopsInput);
		triclopsSetSubpixelInterpolation(triclops, false);
		triclopsSetDisparity(triclops, 7, 164);
		triclopsSetEdgeMask(triclops, 3);
		triclopsSetEdgeCorrelation(triclops, false);
		triclopsSetSurfaceValidation(triclops, false);
		triclopsSetTextureValidation(triclops, false);

		// Do stereo processing
		error = triclopsStereo(triclops);
		printf("imgSize: %d * %d \n", triclopsInput.ncols, triclopsInput.nrows);
		// Retrieve the disparity image from the context
		error = triclopsGetImage(triclops, TriImg_DISPARITY, TriCam_REFERENCE, &triclopsImage);
		printf("disparityImgsize: %d * %d \n", triclopsImage.ncols, triclopsImage.nrows);
		// Save the disparity image
		error = triclopsSaveImage(&triclopsImage, disp_img_name);
		printf("Wrote disparity image to '%s'\n", disp_img_name);

	}
	/*
	CV::Mat disparityPicture = CV::imread("disparity.pgm");
	CV::Rect rect = CV::Rect(324/4, 372/4, 228/4, 264/4);
	CV::rectangle(disparityPicture, rect.tl(), rect.br(), CV::Scalar(rand() & 255, rand() & 255, rand() & 255));
	imshow("测试程序", disparityPicture);

	CV::waitKey(20150901);
	*/
	return EXIT_SUCCESS;
}