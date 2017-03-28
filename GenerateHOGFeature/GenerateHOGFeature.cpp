#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <cvaux.h>
#include <iostream>
#include <math.h>
#include <Eigen\Core>
using namespace cv;
using namespace std;
using namespace Eigen;
int main() {
	const int FRAME_NUM = 100;
	const int DP = 2; // ״̬������ά��
	const int MP = 2; // �۲�������ά��
	const int SamplesNum = 300; // �������ӵ�����
	


	Mat image;
	Mat result;

	namedWindow("result");
	
	image = imread("cup_leftImage1.png",0);//ͼ���С640 * 480
	if (!image.data) {
		printf("��ȡͼ��ʧ��.\n");
		return -1;
	}
	
	for (int i = 0;i < FRAME_NUM;i++) {
	
	
	}
	// Mat roi(image, Rect(200,200,300,450));

	return 0;
	
}







double computeConfidence(Mat img_pre,Mat img_new) {
	//                                     �������ڴ�С  , block��С     ,block�ƶ�����,   cell��С    ,bins����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
	vector<float> descriptors_pre;
	vector<float> descriptors_new;

	hog->compute(img_pre, descriptors_pre, Size(64, 28), Size(0, 0));
	hog->compute(img_new, descriptors_new, Size(64, 28), Size(0, 0));
	int vector_size = descriptors_pre.size();
	VectorXf v1(vector_size);
	VectorXf v2(vector_size);
	for (int i = 0;i < vector_size;i++) {
		v1[i] = descriptors_pre[i];
		v2[i] = descriptors_new[i];
	}
	float norm = (v1 - v2).transpose()*(v1 - v2);
	double confidence = exp(-norm / 250);
	cout << norm << ":" << confidence << endl;
	return confidence;
}