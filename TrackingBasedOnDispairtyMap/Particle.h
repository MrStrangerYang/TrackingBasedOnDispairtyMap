#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;
class  Particle
{
	double x;			// 当前x坐标
	double y;			// 当前y坐标
	double scale;		// 窗口比例系数
	double xPre;			// x坐标预测位置
	double yPre;			// y坐标预测位置
	double scalePre;		// 窗口预测比例系数
	double xOri;			// 原始x坐标
	double yOri;			// 原始y坐标
	cv::Rect roi;			// roi
	vector<float> descripter;
	double weight;			// 该粒子的权重
	Mat img;

public:
	Particle();

	Particle(Mat img,cv::Rect bb);
	void computeHOGFeature(Mat img, cv::Rect bb);
};