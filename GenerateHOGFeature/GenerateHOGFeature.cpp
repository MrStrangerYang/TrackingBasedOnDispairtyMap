#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <math.h>
#include <Eigen\Core>
using namespace cv;
using namespace std;
using namespace Eigen;
int main() {
	Mat image1;
	image1 = imread("cup_leftImage8.png", 0);//ͼ���С640 * 480
	if (!image1.data) {
		printf("��ȡͼ��ʧ��.\n");
		return -1;
	}
	cout << image1.size() << endl;

	/*
	//                                     �������ڴ�С  , block��С     ,block�ƶ�����,   cell��С    ,bins����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(32, 32), cvSize(8, 8), cvSize(16, 16), 9);
	vector<float> descriptors1;
	//           ͼ��, ��������,     window�ƶ�����, padding���������ͼƬ����Ӧ��С
	hog->compute(roi1, descriptors1, Size(2, 2), Size(0, 0));
	hog1.compute(image1, descriptors1);
	cout << "descriptors.size = " << descriptors1.size() << endl;//���hog��������ӵ�ά��
	*/


	Mat roi1(image1, Rect(200.0, 200.0, 300, 450));
	Mat roi2(image1, Rect(210.2, 210.2, 300, 450));
	Mat roi3(image1, Rect(0.1, 0.1, 300, 450));
	//                                     �������ڴ�С  , block��С     ,block�ƶ�����,   cell��С    ,bins����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
	vector<float> descriptors1;
	vector<float> descriptors2;
	vector<float> descriptors3;

	hog->compute(roi1, descriptors1, Size(64, 28), Size(0, 0));

	hog->compute(roi2, descriptors2, Size(64, 28), Size(0, 0));
	hog->compute(roi3, descriptors3, Size(64, 28), Size(0, 0));
	normalize(descriptors1,descriptors1);
	normalize(descriptors2, descriptors2);
	normalize(descriptors3, descriptors3);

	VectorXf v1(descriptors1.size());
	VectorXf v2(descriptors1.size());
	VectorXf v3(descriptors1.size());
	for (int i = 0;i < descriptors1.size();i++) {
		v1[i] = descriptors1[i];
		v2[i] = descriptors2[i];
		v3[i] = descriptors3[i];
	}
	float norm1 = (v1 - v2).transpose()*(v1 - v2);
	float norm2 = (v2 - v3).transpose()*(v2 - v3);
	cout << norm1 << ":" << exp(-norm1) << endl;
	cout << norm2 << ":" << exp(-norm2) << endl;

	return 0;

}