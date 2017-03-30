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
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils/GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"

#include "Kernels.h"

#include "LaRank.h"

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <Eigen/Core>
#include <cvaux.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

Tracker::Tracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2 * conf.searchRadius + 1, 2 * conf.searchRadius + 1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
	particles = vector<Particle>(m_config.particle_num);
	cv::Rect rect;
}

Tracker::~Tracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void Tracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();

	m_needsIntegralImage = false;
	m_needsIntegralHist = false;

	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
		}
		featureCounts.push_back(m_features.back()->GetCount());

		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}

	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);

		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);
	}

	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}


void Tracker::Initialise(const cv::Mat& frame, const cv::Mat disp_frame, FloatRect bb)
{
	m_bb = IntRect(bb);
	//������Ҫʵ���˻���ͼ����  �����������ֱ�Ϊtrue��false 
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)
	{
		//��ʼ��ʱ������һ�β���
		//��ʼ�����������ռ�
		UpdateLearner(image);
	}
	m_initialised = true;

	/*********�ڴ˳�ʼ�������˲���֡************/

	// step 1: ��ʼ��particle ��ȡĿ������ ���ͼHOG����
	for (int i = 0; i < m_config.particle_num; i++)
	{
		Particle tmp_particle = Particle(disp_frame, bb);
		particles.push_back(tmp_particle);
	}

}
// ��ÿһ֡����Trackʱ�����в�������
void Tracker::Track(const cv::Mat& frame)
{
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);	 //��õ�ǰ֡�Ļ���ͼ 
	/******�������ͼHOG������������������Ȩ��******/
	vector<FloatRect> rects;  //�����˲������Գ���  
	RNG rng;
	for (int i = 0;i < m_config.particle_num;i++) {
		double x, y, s, width, height;
		Particle tmp_particle = particles[i];

		tmp_particle.xPre = tmp_particle.x;
		tmp_particle.yPre = tmp_particle.y;
		tmp_particle.scalePre = tmp_particle.scale;
		//�����������Χ�����
		x = m_config.A1 * (tmp_particle.x - tmp_particle.xOri) + m_config.A2 * (tmp_particle.xPre - tmp_particle.xOri) +
			m_config.B0 * rng.gaussian(m_config.SIGMA_X) + tmp_particle.xOri;
		tmp_particle.x = max(0.0, min(x, frame.cols - 1.0));

		y = m_config.A1 * (tmp_particle.y - tmp_particle.yOri) + m_config.A2 * (tmp_particle.yPre - tmp_particle.yOri) +
			m_config.B0 * rng.gaussian(m_config.SIGMA_Y) + tmp_particle.yOri;
		tmp_particle.y = max(0.0, min(y, frame.rows - 1.0));

		s = m_config.A1 * (tmp_particle.scale - 1.0) + m_config.A2 * (tmp_particle.scalePre - 1.0) +
			m_config.B0 * rng.gaussian(m_config.SIGMA_SCALE) + 1.0;
		tmp_particle.scale = max(0.1, min(s, 3.0));

		x = max(0, min(cvRound(tmp_particle.x - 0.5 * tmp_particle.roi.Width() * tmp_particle.scale), frame.cols - 1));		// 0 <= x <= img.rows-1
		y = max(0, min(cvRound(tmp_particle.y - 0.5 * tmp_particle.roi.Width() * tmp_particle.scale), frame.rows - 1));	// 0 <= y <= img.cols-1
		width = min(tmp_particle.roi.Width() * tmp_particle.scale, frame.cols - x);
		height = min(tmp_particle.roi.Height() * tmp_particle.scale, frame.rows - y);

		tmp_particle.roi.Set(x, y, width, height);

		// ����HOG����
		HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 48), cvSize(16, 16), cvSize(8, 8), cvSize(16, 16), 9);
		Mat roi_img(frame, cv::Rect(x, y, width, height));
		vector<float> tmp_descripter;
		hog->compute(roi_img, tmp_descripter, Size(64, 28), Size(0, 0));
		normalize(tmp_descripter, tmp_descripter);

		// ���ͼȨ��
		VectorXf v1(tmp_descripter.size());
		VectorXf v2(tmp_descripter.size());
		for (int i = 0;i < tmp_descripter.size();i++) {
			v1[i] = tmp_descripter[i];
			v2[i] = tmp_particle.descripter[i];
		}
		float norm = (v1 - v2).transpose()*(v1 - v2);
		tmp_particle.weight = exp(-norm);
		rects.push_back(FloatRect(x, y, width, height));  //����������������
	}



	MultiSample sample(image, rects);     //�������࣬��Ҫ�����������Լ�ImageRep image  

	vector<double> scores;     //scores���ŵ��������й�ʽ��10����벿�� 

	//�õ�SVM����scores   sample.GetRects()[0]����
	m_pLearner->Eval(sample, scores, m_bb);
	//Эͬģ��
	vector<double> weightVector;
	double sum = 0.0;
	for (int i = 0;i < m_config.particle_num;i++) {
		weightVector.push_back(scores[i]*particles[i].weight);
		sum += weightVector[i];
	}
	for (int i = 0; i<m_config.particle_num; i++)
	{
		weightVector[i] /= sum;
		particles[i].weight = weightVector[i];
	}
	// step 7: resample�������ӵ�Ȩ�صĺ�����ʷֲ����²���
	sort(particles.begin(), particles.end(), SortByWeight);
	int np, k = 0;
	vector<Particle> tmp_particles;
	for (int i = 0;i < m_config.particle_num;i++) {
		np = cvRound(particles[i].weight*m_config.particle_num);
		for (int j = 0; j < np; j++)
		{
			tmp_particles.push_back(particles[i]);
			if (k == m_config.particle_num)
				goto EXITOUT;
		}
		//����k֮���ֵ����Ϊparticles[0]
		while (k < m_config.particle_num)
		{
			tmp_particles.push_back(particles[0]);
		}
	EXITOUT:
		for (int i = 0; i<m_config.particle_num; i++)
		{
			particles[i] = tmp_particles[i];
		}
	}
	sort(particles.begin(), particles.end(), SortByWeight);

	// step 8: �������ӵ���������Ϊ���ٽ��
	FloatRect tracking_result_rect(0, 0, 0, 0);
	for (int i = 0; i < m_config.particle_num; i++)
	{
		FloatRect roi = particles[i].roi;
		tracking_result_rect.Set(roi.XMin()* particles[i].weight, roi.YMin*particles[i].weight, roi.Width()*particles[i].weight, roi.Height()*particles[i].weight);
	}
	m_bb = tracking_result_rect;
	UpdateLearner(image);
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius + y, m_config.searchRadius + x) = (float)((scores[i] - mn) / (mx - mn));
	}
}

void Tracker::Debug()
{
	imshow("tracker", m_debugImage); //С����
	m_pLearner->Debug();
}

//���뵱ǰ֡��ͼ��
void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0  ��ȡ���Ȳ����������㼯��5*16+1 �������㣩
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2 * m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);

	vector<FloatRect> keptRects; //����������������
	keptRects.push_back(rects[0]); // the true sample   ԭʼ����
	for (int i = 1; i < (int)rects.size(); ++i)
	{	//�޳�����ͼ��߽��������
		if (!rects[i].IsInside(image.GetRect())) continue;

		keptRects.push_back(rects[i]);
	}

#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif

	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);
}
bool SortByWeight(const Particle &p1, const Particle &p2)  
{
	return p1.weight > p2.weight;//��������  
}
