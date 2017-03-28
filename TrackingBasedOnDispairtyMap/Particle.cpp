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
	roi = cv::Rect(x, y, 0, 0);			// ԭʼ�����С
	descripter = vector<float>();
	weight = 0.0;
}
// �ɸ�����ͼ���bounding box������������
Particle::Particle(Mat img, cv::Rect bb)
{
	x = bb.x;			// ��ǰx����
	y = bb.y;			// ��ǰy����
	scale = 1.0;		// ���ڱ���ϵ��
	xPre = bb.x;			// x����Ԥ��λ��
	yPre = bb.y;			// y����Ԥ��λ��
	scalePre = 1.0;		// ����Ԥ�����ϵ��
	xOri = 0.0;			// ԭʼx����
	yOri = 0.0;			// ԭʼy����
	roi = bb;			// ԭʼ�����С
	descripter = vector<float>();
	weight = 0.0;
}

void Particle::computeHOGFeature(Mat img, cv::Rect bb)
{
	//                                     �������ڴ�С  , block��С     ,block�ƶ�����,   cell��С    ,bins����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
	Mat roi_img(img, bb);
	hog->compute(roi_img, this->descripter, Size(64, 28), Size(0, 0));
}

