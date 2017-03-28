/*
* ���������˲����
*/
#include <opencv2\opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Core>
#include "Tracker.h"
#include "Config.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;
using namespace Eigen;

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}
int main() {
	// ���������ļ�
	string configPath = "config.txt";
	Config conf(configPath);

	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}

	// ��ʼ��Tracker
	Tracker tracker(conf);
	
	int startFrame = -1;
	int endFrame = -1;
	FloatRect initBB;
	string imgFormat;
	string disp_img_format;
	float scaleW = 1.f;
	float scaleH = 1.f;

	//***_frames.txt���ļ�·�������ļ�����***�ļ��������Ϊ 1,���ݼ���С��  ���� 1,100
	//                            sequences         /           motorbike /           motorbike_frames.txt
	string framesFilePath = conf.sequenceBasePath + "/" + conf.sequenceName + "/" + conf.sequenceName + "_frames.txt";
	ifstream framesFile(framesFilePath.c_str(), ios::in);
	if (!framesFile)
	{
		cout << "error: could not open sequence frames file: " << framesFilePath << endl;
		return EXIT_FAILURE;
	}
	string framesLine;
	getline(framesFile, framesLine);
	sscanf(framesLine.c_str(), "%d,%d", &startFrame, &endFrame);
	if (framesFile.fail() || startFrame == -1 || endFrame == -1)
	{
		cout << "error: could not parse sequence frames file" << endl;
		return EXIT_FAILURE;
	}
	// ����targetͼƬ                        %05d�ǲ�����λ��ǰ�油0����23�����Ϊ00023
	imgFormat = conf.sequenceBasePath + "/" + conf.sequenceName + "/imgs/%04d.png";

	// read first frame to get size
	char imgPath[256];
	sprintf(imgPath, imgFormat.c_str(), startFrame);
	// imread 0  CV_LOAD_IMAGE_GRAYSCALE ����Ҷ�ͼ
	Mat tmp = imread(imgPath, 0);
	scaleW = (float)conf.frameWidth / tmp.cols;
	scaleH = (float)conf.frameHeight / tmp.rows;

	// �������ͼ
	disp_img_format = conf.sequenceBasePath + "/" + conf.sequenceName + "/imgs/%04d.png";
	// read first frame to get size
	char disp_imgPath[256];
	sprintf(disp_imgPath, disp_img_format.c_str(), startFrame);
	// imread 0  CV_LOAD_IMAGE_GRAYSCALE ����Ҷ�ͼ
	Mat tmp = imread(disp_imgPath, 0);

	// read init box from ground truth file   �����ʼ��  288,36,  26,43 
	string gtFilePath = conf.sequenceBasePath + "/" + conf.sequenceName + "/" + conf.sequenceName + "_gt.txt";
	ifstream gtFile(gtFilePath.c_str(), ios::in);
	if (!gtFile)
	{
		cout << "error: could not open sequence gt file: " << gtFilePath << endl;
		return EXIT_FAILURE;
	}
	string gtLine;
	getline(gtFile, gtLine);
	float xmin = -1.f;
	float ymin = -1.f;
	float width = -1.f;
	float height = -1.f;
	sscanf(gtLine.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);
	if (gtFile.fail() || xmin < 0.f || ymin < 0.f || width < 0.f || height < 0.f)
	{
		cout << "error: could not parse sequence gt file" << endl;
		return EXIT_FAILURE;
	}

	//�����ݼ��Ŀ��������
	initBB = FloatRect(xmin*scaleW, ymin*scaleH, width*scaleW, height*scaleH);
	namedWindow("result");

	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3);
	bool paused = false;
	bool doInitialise = false;

	srand(conf.seed);
	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	{
		Mat frame;
		
		char imgPath[256];
		sprintf(imgPath, imgFormat.c_str(), frameInd);
		Mat frameOrig = cv::imread(imgPath, 0);
		if (frameOrig.empty())
		{
			cout << "error: could not read frame: " << imgPath << endl;
			return EXIT_FAILURE;
		}
		resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
		cvtColor(frame, result, CV_GRAY2RGB);
		// ʹ����֡Ŀ���ʼ��
		if (frameInd == startFrame)
		{
			tracker.Initialise(frame, initBB);
		}

		if (tracker.IsInitialised())
		{
			//��ÿһ֡����Trackʱ�����в�������
			tracker.Track(frame);
			//bunding box��
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
		}

		//��ʾresult
		imshow("result", result);
		int key = waitKey(paused ? 0 : 1);
		if (key != -1)
		{
			if (key == 27 || key == 113) // esc q
			{
				break;
			}
			else if (key == 112) // p
			{
				paused = !paused;
			}
		}

	}
	return EXIT_SUCCESS;
}