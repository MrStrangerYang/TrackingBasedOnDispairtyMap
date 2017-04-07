/* 
 * 添加了粒子滤波的相关参数和向量
 */

#ifndef TRACKER_H
#define TRACKER_H

#include "Rect.h"

#include <vector>
#include <Eigen/Core>
#include <opencv/cv.h>
#include "Particle.h"
class Config;
class Features;
class Kernel;
class LaRank;
class ImageRep;

class Tracker
{
public:
	Tracker(const Config& conf);
	~Tracker();
	
	void Initialise(const cv::Mat& frame, const cv::Mat disp_frame, FloatRect bb);
	void Reset();
	void Track(const cv::Mat& frame, const cv::Mat& frame_disp);
	void Debug();
	
	inline const FloatRect& GetBB() const { return m_bb; }
	inline bool IsInitialised() const { return m_initialised; }
	
private:
	const Config& m_config;
	bool m_initialised;
	std::vector<Features*> m_features;
	std::vector<Kernel*> m_kernels;
	LaRank* m_pLearner;
	FloatRect m_bb;
	cv::Mat m_debugImage;
	bool m_needsIntegralImage;
	bool m_needsIntegralHist;

	// 粒子滤波参数
	vector<Particle> particles;
	Mat disp_img;
	
	void UpdateLearner(const ImageRep& image);
	void UpdateDebugImage(const std::vector<FloatRect>& samples, const FloatRect& centre, const std::vector<double>& scores);
};

#endif
