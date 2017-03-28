#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"   
#include <iostream>

using namespace cv;
int main(int argc, char ** argv)
{
	Mat L = imread("leftImg.png", CV_LOAD_IMAGE_GRAYSCALE), R = imread("rightImg.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat disp;
	Rect roi1(30,30,120,120);
	Rect roi2(30, 30, 120, 120);
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);
	bm->setPreFilterCap(31);
	bm->setBlockSize(21);
	bm->setMinDisparity(-16);
	bm->setNumDisparities(64);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(5);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	Mat left = imread("leftImg.png", CV_LOAD_IMAGE_GRAYSCALE), right = imread("rightImg.png", CV_LOAD_IMAGE_GRAYSCALE);
	bm->compute(left, right, disp);

	disp.convertTo(disp, CV_32F, 1.0 / 16);
	namedWindow("disp", CV_WINDOW_AUTOSIZE);
	imshow("disp", disp);
	waitKey(0);

	system("PAUSE");
	return 0;
}