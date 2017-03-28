#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;
struct  Particle
{
	double x;			// ��ǰx����
	double y;			// ��ǰy����
	double scale;		// ���ڱ���ϵ��
	double xPre;			// x����Ԥ��λ��
	double yPre;			// y����Ԥ��λ��
	double scalePre;		// ����Ԥ�����ϵ��
	double xOri;			// ԭʼx����
	double yOri;			// ԭʼy����
	Rect rect;			// ԭʼ�����С
	MatND hist;			// �������������ֱ��ͼ
	double weight;		// �����ӵ�Ȩ��
};