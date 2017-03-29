#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Core>
#include "Rect.h"
using namespace std;
using namespace cv;
using namespace Eigen;
class  Particle
{
public:
	double x;			// ��ǰx����
	double y;			// ��ǰy����
	double scale;		// ���ڱ���ϵ��
	double xPre;			// x����Ԥ��λ��
	double yPre;			// y����Ԥ��λ��
	double scalePre;		// ����Ԥ�����ϵ��
	double xOri;			// ԭʼx����
	double yOri;			// ԭʼy����
	FloatRect roi;			// roi
	vector<float> descripter;
	double weight;			// �����ӵ�Ȩ��
	Mat img;


	Particle();

	Particle(Mat img, FloatRect bb);
	void computeHOGFeature(Mat img, FloatRect bb);
};