#include "Particle.h"
#include <cvaux.h>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;

Particle::Particle() {
	x = 0.0;			// 当前x坐标
	y = 0.0;			// 当前y坐标
	scale = 1.0;		// 窗口比例系数
	xPre = 0.0;			// x坐标预测位置
	yPre = 0.0;			// y坐标预测位置
	scalePre = 1.0;		// 窗口预测比例系数
	xOri = 0.0;			// 原始x坐标
	yOri = 0.0;			// 原始y坐标
	roi = cv::Rect(x, y, 0, 0);			// 原始区域大小
	descripter = vector<float>();
	weight = 0.0;
}
// 由给定的图像和bounding box区域生成特征
Particle::Particle(Mat img, cv::Rect bb)
{
	x = bb.x;			// 当前x坐标
	y = bb.y;			// 当前y坐标
	scale = 1.0;		// 窗口比例系数
	xPre = bb.x;			// x坐标预测位置
	yPre = bb.y;			// y坐标预测位置
	scalePre = 1.0;		// 窗口预测比例系数
	xOri = 0.0;			// 原始x坐标
	yOri = 0.0;			// 原始y坐标
	roi = bb;			// 原始区域大小
	descripter = vector<float>();
	weight = 0.0;
}

void Particle::computeHOGFeature(Mat img, cv::Rect bb)
{
	//                                     滑动窗口大小  , block大小     ,block移动步长,   cell大小    ,bins个数
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
	Mat roi_img(img, bb);
	hog->compute(roi_img, this->descripter, Size(64, 28), Size(0, 0));
}

