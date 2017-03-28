//=============================================================================
// 利用Triclops自带的api，读入样本库双目图片，设置ROI参数（左上x,y坐标，长和宽），生成深度图
//=============================================================================
#include <stdio.h>
#include <stdlib.h>
#include "triclops.h"
#include "fc2triclops.h"
#include "pnmutils.h"
#include <opencv2\opencv.hpp>
#define _HANDLE_TRICLOPS_ERROR( description, error ){if( error != TriclopsErrorOk ){printf("*** Triclops Error '%s' at line %d :\n\t%s\n",triclopsErrorToString( error ),__LINE__,description );printf( "Press any key to exit...\n" );getchar();exit( 1 );}}

namespace FC2 = FlyCapture2;
namespace FC2T = Fc2Triclops;
namespace CV = cv;
struct ImageContainer
{
	FC2::Image tmp[2];
	FC2::Image unprocessed[2];
};

enum IMAGE_SIDE
{
	RIGHT = 0, LEFT
};

void configureROIs(const TriclopsContext & triclops);
void doStereo(const TriclopsContext & triclops,
	const TriclopsInput   & triclopsInput);


int main() {
	TriclopsContext triclops;
	TriclopsError error;
	TriclopsImage triclopsImage;
	TriclopsInput   triclopsInput;
	std::string szCalFile("input.cal");
	std::string szInputFile("packedColorImage.pgm");
	std::string szDisparityFile("disparity.pgm");



	// 加载相机配置文件
	error = triclopsGetDefaultContextFromFile(&triclops, const_cast<char *>(szCalFile.c_str()));
	_HANDLE_TRICLOPS_ERROR("triclopsGetDefaultContextFromFile(): "
		"Can't open calibration file",
		error);

	// 加载双目相机图片
	if (!pgmReadToTriclopsInput(szInputFile.c_str(), &triclopsInput))
	{
		printf("pgmReadToTriclopsInput() failed. Can't read '%s'\n", szInputFile.c_str());
		return 1;
	}

	// 根据相机参数校正输入图像
	error = triclopsRectify(triclops, &triclopsInput);

	configureROIs(triclops);
	doStereo(triclops, triclopsInput);
	CV::Mat picture = CV::imread("disparity.pgm");
	imshow("测试程序", picture);
	CV::waitKey(20150901);

	return 0;
}

void configureROIs(const TriclopsContext &triclops)
{
	TriclopsROI * pRois;
	int           nMaxRois;
	TriclopsError te;
	TriclopsImage packedImage;
	pgmReadToTriclopsImage("packedColorImage.pgm",&packedImage);
	printf("packeColorImage size : %d * %d",packedImage.ncols,packedImage.nrows);
	// Get the pointer to the regions of interest array  nMaxRois=100
	te = triclopsGetROIs(triclops, &pRois, &nMaxRois);
	int count = 0;
	for (int i = 0;i < 5;i++) {
		for (int j = 0;j < 6;j++) {
			pRois[count].row = 160 * i;
			pRois[count].col = 120 * j;
			pRois[count].ncols = 120;
			pRois[count].nrows = 160;
			count++;
		}
	}
	printf("Rois num : %d", count);

	te = triclopsSetNumberOfROIs(triclops, count);
}

void doStereo(const TriclopsContext & triclops,
	const TriclopsInput   & triclopsInput) {
	
	TriclopsImage disparityImage;
	// Set to 640x480 output images
	triclopsSetResolution(triclops, 900, 720);
	// Set disparity range to be quite wide
	triclopsSetDisparity(triclops, 0, 200);
	triclopsStereo(triclops);
	triclopsGetImage(triclops,TriImg_DISPARITY,TriCam_REFERENCE,&disparityImage);
	printf("%d * %d",disparityImage.ncols,disparityImage.nrows);

	const char * pDisparityFilename = "disparity.pgm";
	triclopsSaveImage(&disparityImage, const_cast<char *>(pDisparityFilename));
}
