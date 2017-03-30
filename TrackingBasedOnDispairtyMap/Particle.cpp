#include "Particle.h"
#include <cvaux.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

Particle::Particle() {
	x = 0.0;			// ��ǰx����
	y = 0.0;			// ��ǰy����
	scale = 1.0;		// ���ڱ���ϵ��
	xPre = 0.0;			// x����Ԥ��λ��
	yPre = 0.0;			// y����Ԥ��λ��
	scalePre = 1.0;		// ����Ԥ�����ϵ��
	xOri = 0.0;			// ԭʼx����
	yOri = 0.0;			// ԭʼy����
	roi = FloatRect(x, y, 0, 0);			// ԭʼ�����С
	descripter = vector<float>();
	weight = 0.0;
}
// �ɸ�����ͼ���bounding box������������
Particle::Particle(Mat img, FloatRect bb)
{
	x = bb.XMin();			// ��ǰx����
	y = bb.YMin();			// ��ǰy����
	scale = 1.0;		// ���ڱ���ϵ��
	xPre = bb.YMin();			// x����Ԥ��λ��
	yPre = bb.YMin();			// y����Ԥ��λ��
	scalePre = 1.0;		// ����Ԥ�����ϵ��
	xOri = x;			// ԭʼx����
	yOri = y;			// ԭʼy����
	roi = bb;			// ԭʼ�����С
	computeHOGFeature(img,bb);
	weight = 0.0;		// ���ͼ����ֵ
}

void Particle::computeHOGFeature(Mat img, FloatRect bb)
{
	//                                     �������ڴ�С  , block��С     ,block�ƶ�����,   cell��С    ,bins����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
	Mat roi_img(img, cv::Rect(bb.XMin(),bb.YMin(),bb.Width(),bb.Height()));
	hog->compute(roi_img, this->descripter, Size(64, 28), Size(0, 0));
	normalize(this->descripter,this->descripter);
}

