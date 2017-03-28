#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;
class  Particle
{
	double x;			// ��ǰx����
	double y;			// ��ǰy����
	double scale;		// ���ڱ���ϵ��
	double xPre;			// x����Ԥ��λ��
	double yPre;			// y����Ԥ��λ��
	double scalePre;		// ����Ԥ�����ϵ��
	double xOri;			// ԭʼx����
	double yOri;			// ԭʼy����
	cv::Rect roi;			// roi
	vector<float> descripter;
	double weight;			// �����ӵ�Ȩ��
	Mat img;

public:
	Particle();

	Particle(Mat img,cv::Rect bb);
	void computeHOGFeature(Mat img, cv::Rect bb);
};