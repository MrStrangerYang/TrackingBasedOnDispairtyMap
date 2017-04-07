#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <Eigen/Core>
using namespace std;
using namespace cv;
int main() {
	/*vector<int> vector;
	cout << vector.size() << endl;
	vector.push_back(2);
	cout << vector.size() << endl;
	*/

	//Mat image = cv::imread("cup_leftImage1.png",0);
	Mat image = cv::imread("cup_dispImg1.pgm", 0);
	cout << image.size() << endl;
	Rect rect(286,153,84,103);
	resize(image,image,Size(640,480));
	rectangle(image, rect, CV_RGB(0, 255, 0));
	namedWindow("show");
	imshow("show",image);
	waitKey();
}