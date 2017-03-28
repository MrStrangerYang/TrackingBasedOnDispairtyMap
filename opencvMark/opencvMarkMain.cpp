#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

Mat image;
Mat rowImage;
Rect rect;
bool drawimage = false;

//函数声明
void on_mouse(int event, int x, int y, int type, void *param);
void draw_rectangle(cv::Mat& img, cv::Rect box);
int main() {

	Mat image = imread("cup_packedImagePNG1.png");
	image.copyTo(rowImage);
	Mat tempimage;
	/*
	namedWindow("opencv标注");
	setMouseCallback("opencv标注", on_mouse, (void*)&image);
	while (1)
	{
		image.copyTo(tempimage);
		if (drawimage)
			draw_rectangle(tempimage, rect);
		imshow("opencv标注", tempimage);
		if (waitKey(10) == 'q')
			break;
	}
	printf("image size : %d * %d",image.cols/2,image.rows);
	//(568,304,176,212)
	printf("Bounding box : x %d , y %d , width %d ,height %d \n", rect.x, rect.y, rect.width, rect.height);*/
	namedWindow("opencv");
	Rect labelRect(568,304,176,212);
	Mat image2 = imread("cup_leftImage1.png");
	draw_rectangle(image2, labelRect);
	imshow("opencv", image2);
	waitKey(10000);
	return 1;
}

void on_mouse(int event, int x, int y, int type, void *param)
{
	Mat& img = *(Mat*)param;
	switch (event)
	{
	case(EVENT_LBUTTONDOWN):
	{
		drawimage = true;
		rowImage.copyTo(img);
		rect = Rect(x, y, 0, 0);
	}
	break;
	case(EVENT_MOUSEMOVE):
	{
		if (drawimage)
		{
			rect.width = x - rect.x;
			rect.height = y - rect.y;
		}

	}
	break;
	case(EVENT_LBUTTONUP):
	{
		drawimage = false;
		draw_rectangle(img, rect);
	}
	break;
	}
	
	
}
void draw_rectangle(cv::Mat& img, cv::Rect box)
{
	rectangle(img, box.tl(), box.br(), Scalar(rand() & 255, rand() & 255, rand() & 255));
}