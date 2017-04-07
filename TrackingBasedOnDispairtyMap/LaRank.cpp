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

#include "LaRank.h"

#include "Config.h"
#include "Features.h"
#include "Kernels.h"
#include "Sample.h"
#include "Rect.h"
#include "GraphUtils/GraphUtils.h"

#include <Eigen/Core>

# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc/imgproc.hpp>

static const int kTileSize = 30;
using namespace cv;

using namespace std;
using namespace Eigen;

static const int kMaxSVs = 2000; // TODO (only used when no budget)


LaRank::LaRank(const Config& conf, const Features& features, const Kernel& kernel) :
	m_config(conf),
	m_features(features),
	m_kernel(kernel),
	m_C(conf.svmC)   //惩罚项m_C(conf.svmC) svmC = 100.0 
{	// N为特征向量的个数  100                    100           
	int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;  //N=100+2，特征向量的个数不能超过这个阈值  
	m_K = MatrixXd::Zero(N, N);   //m_K表示核矩阵，初始化为102*102的零矩阵
	m_debugImage = Mat(800, 600, CV_8UC3);
}

LaRank::~LaRank()
{
}

double LaRank::Evaluate(const Eigen::VectorXd& x, const FloatRect& y) const   //论文中公式10后半部分计算，即F  评估函数
{
	double f = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector& sv = *m_svs[i];
		f += sv.b*m_kernel.Eval(x, sv.x->x[sv.y]);  //beta*高斯核  beta*<x,前面n帧的支持向量>  关键是前面n帧得到的beta值
	}
	return f;
}

void LaRank::Eval(const MultiSample& sample, std::vector<double>& results)
{
	const FloatRect& centre(sample.GetRects()[0]);      //原始目标框  
	vector<VectorXd> fvs;
	const_cast<Features&>(m_features).Eval(sample, fvs);     //fvs存放haar特征值  
	results.resize(fvs.size());
	for (int i = 0; i < (int)fvs.size(); ++i)
	{
		// express y in coord frame of centre sample
		FloatRect y(sample.GetRects()[i]);
		y.Translate(-centre.XMin(), -centre.YMin());
		results[i] = Evaluate(fvs[i], y);
	}
}
void LaRank::Eval(const MultiSample& sample, std::vector<double>& results, FloatRect centre)
{
	vector<VectorXd> fvs;
	const_cast<Features&>(m_features).Eval(sample, fvs);     //fvs存放haar特征值  
	results.resize(fvs.size());
	for (int i = 0; i < (int)fvs.size(); ++i)
	{
		// express y in coord frame of centre sample
		FloatRect y(sample.GetRects()[i]);
		y.Translate(-centre.XMin(), -centre.YMin());
		results[i] = Evaluate(fvs[i], y);
	}
}
//sample 当中有当前处理的image 和所有的样本框，y为原始目标框
void LaRank::Update(const MultiSample& sample, int y)
{
	// add new support pattern
	SupportPattern* sp = new SupportPattern;  //定义一个sp  
	const vector<FloatRect>& rects = sample.GetRects(); //获得所有的样本框
	FloatRect centre = rects[y];					 //原始目标框 
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		// express r in coord frame of centre sample
		FloatRect r = rects[i];
		r.Translate(-centre.XMin(), -centre.YMin());   //这就表示帧间目标位置变化关系
		sp->yv.push_back(r);
		
	}
	// evaluate features for each sample  评估每个样本
	sp->x.resize(rects.size());  //有多少个感兴趣的框，就有多少个特征值向量。  vector<Eigen::VectorXd> x    VectorXd：Dynamic column vector of doubles  
	const_cast<Features&>(m_features).Eval(sample, sp->x); //将每个样本框计算得到的haar特征存入sp->x   x为 column vector of doubles 
	sp->y = y;
	sp->refCount = 0;
	m_sps.push_back(sp); //存储sp  

	ProcessNew((int)m_sps.size()-1);   //执行该步骤，添加支持向量，并对beta值进行调整； 每处理一帧图像，m_sps的数量都增加1，这样定义ind能够保证ProcessNew所处理的样本都是最新的样本。
	BudgetMaintenance();       //保证支持向量没有超出限定阈值  
	
	for (int i = 0; i < 10; ++i)
	{
		Reprocess();          //包括processold：增加新的sv；optimize：在现有的sv基础上调整beta值 
		BudgetMaintenance();
	}
}

void LaRank::BudgetMaintenance()
{
	if (m_config.svmBudgetSize > 0)
	{
		while ((int)m_svs.size() > m_config.svmBudgetSize)
		{
			BudgetMaintenanceRemove();//支持向量的个数超出阈值后，找到对于F函数影响最小的负sv，并移除。  
		}
	}
}

void LaRank::Reprocess()
{
	ProcessOld();
	for (int i = 0; i < 10; ++i)
	{
		Optimize();
	}
}

double LaRank::ComputeDual() const
{
	double d = 0.0;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		const SupportVector* sv = m_svs[i];
		d -= sv->b*Loss(sv->x->yv[sv->y], sv->x->yv[sv->x->y]);
		for (int j = 0; j < (int)m_svs.size(); ++j)
		{
			d -= 0.5*sv->b*m_svs[j]->b*m_K(i,j);
		}
	}
	return d;
}
//Algorithm 1,更新beta和gradient值  参数为vector<SupportVector*> m_svs中新加入的正、负支持向量的编号
void LaRank::SMOStep(int ipos, int ineg)
{
	if (ipos == ineg) return;

	SupportVector* svp = m_svs[ipos];			 //定义一个正支持向量  
	SupportVector* svn = m_svs[ineg];		     //定义一个负支持向量  
	assert(svp->x == svn->x);
	SupportPattern* sp = svp->x;	    //定义一个支持模式sp，将正支持向量的支持模式赋予sp  

#if VERBOSE
	cout << "SMO: gpos:" << svp->g << " gneg:" << svn->g << endl;
#endif	
	if ((svp->g - svn->g) < 1e-5)
	{
#if VERBOSE
		cout << "SMO: skipping" << endl;
#endif		
	}
	else
	{   //论文中的Algorithm步骤  
		double kii = m_K(ipos, ipos) + m_K(ineg, ineg) - 2*m_K(ipos, ineg);
		double lu = (svp->g-svn->g)/kii;
		// no need to clamp against 0 since we'd have skipped in that case
		double l = min(lu, m_C*(int)(svp->y == sp->y) - svp->b); //惩罚项m_C(conf.svmC) svmC = 100.0 

		svp->b += l;
		svn->b -= l;

		// update gradients
		for (int i = 0; i < (int)m_svs.size(); ++i)
		{
			SupportVector* svi = m_svs[i];
			svi->g -= l*(m_K(i, ipos) - m_K(i, ineg));
		}
#if VERBOSE
		cout << "SMO: " << ipos << "," << ineg << " -- " << svp->b << "," << svn->b << " (" << l << ")" << endl;
#endif		
	}
	
	// check if we should remove either sv now
	
	if (fabs(svp->b) < 1e-8)
	{
		RemoveSupportVector(ipos);
		if (ineg == (int)m_svs.size())
		{
			// ineg and ipos will have been swapped during sv removal
			ineg = ipos;
		}
	}

	if (fabs(svn->b) < 1e-8)
	{
		RemoveSupportVector(ineg);
	}
}

pair<int, double> LaRank::MinGradient(int ind)
{
	const SupportPattern* sp = m_sps[ind];
	pair<int, double> minGrad(-1, DBL_MAX);
	for (int i = 0; i < (int)sp->yv.size(); ++i)
	{
		double grad = -Loss(sp->yv[i], sp->yv[sp->y]) - Evaluate(sp->x[i], sp->yv[i]);
		if (grad < minGrad.second)
		{
			minGrad.first = i;
			minGrad.second = grad;
		}
	}
	return minGrad;
}
//  ProcessNew((int)m_sps.size() - 1);  执行该步骤，添加支持向量，并对beta值进行调整； 每处理一帧图像，m_sps的数量都增加1
//  ind为m_sps新样本的pattern（haar特征向量 double）的序号
void LaRank::ProcessNew(int ind)  //可以添加新的支持向量，增加的正负支持向量(sv)具有相同的支持模式
{
	// gradient is -f(x,y) since loss=0
	//  Evaluate函数 论文中公式10后半部分计算，即F评估函数    当前帧的特征值向量x与前面的原始目标框的核函数相似度之和
	int ip = AddSupportVector(m_sps[ind], m_sps[ind]->y, -Evaluate(m_sps[ind]->x[m_sps[ind]->y],m_sps[ind]->yv[m_sps[ind]->y]));  //处理当前新样本，将上一帧目标位置作为正向量加入  

	pair<int, double> minGrad = MinGradient(ind);  //int，double分别是具有最小梯度的样本框存放的位置，最小梯度的数值  公式10 当中g(y)梯度的最小值和序号
	int in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);    //将当前具有最小梯度的样本作为负向量加入 

	SMOStep(ip, in);   //Algorithm 1,更新beta和gradient值  参数为vector<SupportVector*> m_svs中新加入的正、负支持向量的编号
}

// ProcessOld()主要对已经存在的SupportPattern进行随机选取并处理。
void LaRank::ProcessOld()
{
	if (m_sps.size() == 0) return;

	// choose pattern to process 对已经存在的SupportPattern进行随机选取
	int ind = rand() % m_sps.size();

	// find existing sv with largest grad and nonzero beta
	int ip = -1;
	double maxGrad = -DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		//  m_svs[i]->x == m_sps[ind]
		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
	}
	assert(ip != -1);
	if (ip == -1) return;

	// find potentially new sv with smallest grad
	pair<int, double> minGrad = MinGradient(ind);
	int in = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		if (m_svs[i]->y == minGrad.first)
		{
			in = i;
			break;
		}
	}
	if (in == -1)
	{
		// add new sv
		in = AddSupportVector(m_sps[ind], minGrad.first, minGrad.second);
	}

	SMOStep(ip, in);
}

void LaRank::Optimize()
{
	if (m_sps.size() == 0) return;
	
	// choose pattern to optimize
	int ind = rand() % m_sps.size();

	int ip = -1;
	int in = -1;
	double maxGrad = -DBL_MAX;
	double minGrad = DBL_MAX;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->x != m_sps[ind]) continue;

		const SupportVector* svi = m_svs[i];
		if (svi->g > maxGrad && svi->b < m_C*(int)(svi->y == m_sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
		if (svi->g < minGrad)
		{
			in = i;
			minGrad = svi->g;
		}
	}
	assert(ip != -1 && in != -1);
	if (ip == -1 || in == -1)
	{
		// this shouldn't happen
		cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		return;
	}

	SMOStep(ip, in);
}
// SupportPattern* x  
int LaRank::AddSupportVector(SupportPattern* x, int y, double g)
{
	SupportVector* sv = new SupportVector;
	sv->b = 0.0;  // double  beta值
	sv->x = x;	// SupportPattern*  支持模式数量
	sv->y = y;	//int
	sv->g = g;	//double  同前面的所有正样本（y=0 初始框）的相似度之和    一个评估值

	int ind = (int)m_svs.size(); //  vector<SupportVector*> m_svs;
	m_svs.push_back(sv);
	x->refCount++;

#if VERBOSE
	cout << "Adding SV: " << ind << endl;
#endif

	// update kernel matrix
	for (int i = 0; i < ind; ++i)
	{
		// m_K   Eigen::MatrixXd   SupportVector->SupportPattern->haar特征
		m_K(i,ind) = m_kernel.Eval(m_svs[i]->x->x[m_svs[i]->y], x->x[y]);  //  exp(-m_sigma*(x1-x2).squaredNorm()); 高斯核
		m_K(ind,i) = m_K(i,ind);
	}
	m_K(ind,ind) = m_kernel.Eval(x->x[y]);

	return ind;
}

void LaRank::SwapSupportVectors(int ind1, int ind2)
{
	SupportVector* tmp = m_svs[ind1];
	m_svs[ind1] = m_svs[ind2];
	m_svs[ind2] = tmp;
	
	VectorXd row1 = m_K.row(ind1);
	m_K.row(ind1) = m_K.row(ind2);
	m_K.row(ind2) = row1;
	
	VectorXd col1 = m_K.col(ind1);
	m_K.col(ind1) = m_K.col(ind2);
	m_K.col(ind2) = col1;
}

void LaRank::RemoveSupportVector(int ind)
{
#if VERBOSE
	cout << "Removing SV: " << ind << endl;
#endif	

	m_svs[ind]->x->refCount--;
	if (m_svs[ind]->x->refCount == 0)
	{
		// also remove the support pattern
		for (int i = 0; i < (int)m_sps.size(); ++i)
		{
			if (m_sps[i] == m_svs[ind]->x)
			{
				delete m_sps[i];
				m_sps.erase(m_sps.begin()+i);
				break;
			}
		}
	}

	// make sure the support vector is at the back, this
	// lets us keep the kernel matrix cached and valid
	if (ind < (int)m_svs.size()-1)
	{
		SwapSupportVectors(ind, (int)m_svs.size()-1);
		ind = (int)m_svs.size()-1;
	}
	delete m_svs[ind];
	m_svs.pop_back();
}

void LaRank::BudgetMaintenanceRemove()
{
	// find negative sv with smallest effect on discriminant function if removed
	double minVal = DBL_MAX;
	int in = -1;
	int ip = -1;
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		if (m_svs[i]->b < 0.0)
		{
			// find corresponding positive sv
			int j = -1;
			for (int k = 0; k < (int)m_svs.size(); ++k)
			{
				if (m_svs[k]->b > 0.0 && m_svs[k]->x == m_svs[i]->x)
				{
					j = k;
					break;
				}
			}
			double val = m_svs[i]->b*m_svs[i]->b*(m_K(i,i) + m_K(j,j) - 2.0*m_K(i,j));
			if (val < minVal)
			{
				minVal = val;
				in = i;
				ip = j;
			}
		}
	}

	// adjust weight of positive sv to compensate for removal of negative
	m_svs[ip]->b += m_svs[in]->b;

	// remove negative sv
	RemoveSupportVector(in);
	if (ip == (int)m_svs.size())
	{
		// ip and in will have been swapped during support vector removal
		ip = in;
	}
	
	if (m_svs[ip]->b < 1e-8)
	{
		// also remove positive sv
		RemoveSupportVector(ip);
	}

	// update gradients
	// TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
	for (int i = 0; i < (int)m_svs.size(); ++i)
	{
		SupportVector& svi = *m_svs[i];
		svi.g = -Loss(svi.x->yv[svi.y],svi.x->yv[svi.x->y]) - Evaluate(svi.x->x[svi.y], svi.x->yv[svi.y]);
	}	
}

void LaRank::Debug()
{
	cout << m_sps.size() << "/" << m_svs.size() << " support patterns/vectors" << endl;
	UpdateDebugImage();
	imshow("learner", m_debugImage);
}

void LaRank::UpdateDebugImage() //该函数主要用于样本显示
{
	m_debugImage.setTo(0);
	
	int n = (int)m_svs.size();
	
	if (n == 0) return;
	
	const int kCanvasSize = 600;
	int gridSize = (int)sqrtf((float)(n-1)) + 1;
	int tileSize = (int)((float)kCanvasSize/gridSize);
	
	if (tileSize < 5)
	{
		cout << "too many support vectors to display" << endl;
		return;
	}
	
	Mat temp(tileSize, tileSize, CV_8UC1);
	int x = 0;
	int y = 0;
	int ind = 0;
	float vals[kMaxSVs];
	memset(vals, 0, sizeof(float)*n);
	int drawOrder[kMaxSVs];
	
	for (int set = 0; set < 2; ++set)
	{
		for (int i = 0; i < n; ++i)
		{
			if (((set == 0) ? 1 : -1)*m_svs[i]->b < 0.0) continue;
			
			drawOrder[ind] = i;
			vals[ind] = (float)m_svs[i]->b;
			++ind;
			
			Mat I = m_debugImage(cv::Rect(x, y, tileSize, tileSize));
			resize(m_svs[i]->x->images[m_svs[i]->y], temp, temp.size());
			cvtColor(temp, I, CV_GRAY2RGB);
			double w = 1.0;
			rectangle(I, Point(0, 0), Point(tileSize-1, tileSize-1), (m_svs[i]->b > 0.0) ? CV_RGB(0, (uchar)(255*w), 0) : CV_RGB((uchar)(255*w), 0, 0), 3);
			x += tileSize;
			if ((x+tileSize) > kCanvasSize)
			{
				y += tileSize;
				x = 0;
			}
		}
	}
	
	const int kKernelPixelSize = 2;
	int kernelSize = kKernelPixelSize*n;
	
	double kmin = m_K.minCoeff();
	double kmax = m_K.maxCoeff();
	
	if (kernelSize < m_debugImage.cols && kernelSize < m_debugImage.rows)
	{
		Mat K = m_debugImage(cv::Rect(m_debugImage.cols-kernelSize, m_debugImage.rows-kernelSize, kernelSize, kernelSize));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				Mat Kij = K(cv::Rect(j*kKernelPixelSize, i*kKernelPixelSize, kKernelPixelSize, kKernelPixelSize));
				uchar v = (uchar)(255*(m_K(drawOrder[i], drawOrder[j])-kmin)/(kmax-kmin));
				Kij.setTo(Scalar(v, v, v));
			}
		}
	}
	else
	{
		kernelSize = 0;
	}
	
	
	Mat I = m_debugImage(cv::Rect(0, m_debugImage.rows - 200, m_debugImage.cols-kernelSize, 200));
	I.setTo(Scalar(255,255,255));
	IplImage II = I;
	setGraphColor(0);
	drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
}
