/*
* Struck: Structured Output Tracking with Kernels
*
* Code to accompany the paper:
*   Struck: Structured Output Tracking with Kernels
*   Sam Hare, Amir Saffari, Philip H. S. Torr
*   International Conference on Computer Vision (ICCV), 2011
*
* Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
*
* This file is part of Struck.
*
* Struck is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Struck is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Struck.  If not, see <http://www.gnu.org/licenses/>.
*
*/

#include "Tracker.h"
#include "Config.h"

#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "vot.hpp"

using namespace std;
using namespace cv;

static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

int main(int argc, char* argv[])
{
	// read config file
	string configPath = "config.txt";
	cout << configPath << endl;
	Config conf(configPath);
	cout << conf << endl;
	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}

	Tracker tracker(conf);
	ofstream outFile;
	if (conf.resultsPath != "")
	{
		outFile.open(conf.resultsPath.c_str(), ios::out);
		if (!outFile)
		{
			cout << "error: could not open results file: " << conf.resultsPath << endl;
			return EXIT_FAILURE;
		}
	}

	int startFrame = -1;
	int endFrame = -1;
	FloatRect initBB;
	string imgFormat;
	string disp_format;


	// parse frames file   
	//***_frames.txt的文件路径，该文件放在***文件夹里，内容为1,数据集大小。  例： 1,100
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

	imgFormat = conf.sequenceBasePath + "/" + conf.sequenceName + "/imgs/" + conf.sequenceName + "_leftImage%d.png";
	disp_format = conf.sequenceBasePath + "/" + conf.sequenceName + "/dispImgs/" + conf.sequenceName + "_dispImg%d.pgm";
	// read first frame to get size
	char imgPath[256];
	char disp_img_path[256];
	sprintf(imgPath, imgFormat.c_str(), startFrame);
	sprintf(disp_img_path, disp_format.c_str(), startFrame);

	// imread 0  CV_LOAD_IMAGE_GRAYSCALE 读入灰度图
	Mat tmp = cv::imread(imgPath, 0);

	// read init box from ground truth file   初始框  288,36,  26,43
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

	//将bonding box进行缩放
	initBB = FloatRect(xmin, ymin, width, height);

	namedWindow("result");

	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3);
	bool paused = false;
	bool doInitialise = false;
	srand(conf.seed);
	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	{
		Mat frame;
		char imgPath[256];
		// 将startFrame传入imgPath,形成下一帧图片的地址
		sprintf(imgPath, imgFormat.c_str(), frameInd);
		sprintf(disp_img_path, disp_format.c_str(), frameInd);
		// 读入图像、深度图像（灰度图）
		Mat frameOrig = cv::imread(imgPath, 0);
		Mat frame_disp = cv::imread(disp_img_path, 0);
		
		if (frameOrig.empty())
		{
			cout << "error: could not read frame: " << imgPath << endl;
			return EXIT_FAILURE;
		}
		resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
		resize(frame_disp, frame_disp, Size(conf.frameWidth, conf.frameHeight));
		cvtColor(frame, result, CV_GRAY2RGB);

		if (frameInd == startFrame)
		{
			tracker.Initialise(frame, frame_disp, initBB);
		}
		if (tracker.IsInitialised())
		{
			//用每一帧进行Track时，进行参数调整
			tracker.Track(frame, frame_disp);
			//绘制bunding box框
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
			if (outFile)
			{
				const FloatRect& bb = tracker.GetBB();
				outFile << bb.XMin() << "," << bb.YMin() << "," << bb.Width() << "," << bb.Height() << endl;
			}
		}

		//显示result
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

	if (outFile.is_open())
	{
		outFile.close();
	}

	return EXIT_SUCCESS;
}
