#include "Config.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

Config::Config(const std::string& path)
{
	SetDefaults();

	ifstream f(path.c_str());
	if (!f)
	{
		cout << "error: could not load config file: " << path << endl;
		return;
	}

	string line, name, tmp;
	while (getline(f, line))
	{
		istringstream iss(line);
		iss >> name >> tmp;

		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#') continue;
		if (name == "particle_num") iss >> particle_num;
		else if (name == "A1") iss >> A1;
		else if (name == "A2") iss >> A2;
		else if (name == "B0") iss >> B0;
		else if (name == "SIGMA_X") iss >> SIGMA_X;
		else if (name == "SIGMA_Y") iss >> SIGMA_Y;
		else if (name == "SIGMA_SCALE") iss >> SIGMA_SCALE;

		else if (name == "seed") iss >> seed;
		else if (name == "sequenceBasePath") iss >> sequenceBasePath;
		else if (name == "sequenceName") iss >> sequenceName;
		else if (name == "resultsPath") iss >> resultsPath;
		else if (name == "frameWidth") iss >> frameWidth;
		else if (name == "frameHeight") iss >> frameHeight;
		else if (name == "seed") iss >> seed;
		else if (name == "searchRadius") iss >> searchRadius;
		else if (name == "svmC") iss >> svmC;
		else if (name == "svmBudgetSize") iss >> svmBudgetSize;
		else if (name == "feature")
		{
			string featureName, kernelName;
			double param;
			iss >> featureName >> kernelName >> param;

			FeatureKernelPair fkp;

			if (featureName == FeatureName(kFeatureTypeHaar)) fkp.feature = kFeatureTypeHaar;
			else if (featureName == FeatureName(kFeatureTypeRaw)) fkp.feature = kFeatureTypeRaw;
			else if (featureName == FeatureName(kFeatureTypeHistogram)) fkp.feature = kFeatureTypeHistogram;
			else
			{
				cout << "error: unrecognised feature: " << featureName << endl;
				continue;
			}

			if (kernelName == KernelName(kKernelTypeLinear)) fkp.kernel = kKernelTypeLinear;
			else if (kernelName == KernelName(kKernelTypeIntersection)) fkp.kernel = kKernelTypeIntersection;
			else if (kernelName == KernelName(kKernelTypeChi2)) fkp.kernel = kKernelTypeChi2;
			else if (kernelName == KernelName(kKernelTypeGaussian))
			{
				if (iss.fail())
				{
					cout << "error: gaussian kernel requires a parameter (sigma)" << endl;
					continue;
				}
				fkp.kernel = kKernelTypeGaussian;
				fkp.params.push_back(param);
			}
			else
			{
				cout << "error: unrecognised kernel: " << kernelName << endl;
				continue;
			}

			features.push_back(fkp);
		}
	}
}

void Config::SetDefaults()
{
	particle_num = 100;	// ���Ӹ���
	// ���ӷ�����������
	A1 = 2.0;
	A2 = -1.0;
	B0 = 1.0;
	// ��˹�����sigma����
	SIGMA_X = 1.0;
	SIGMA_Y = 0.5;
	SIGMA_SCALE = 0.001;

	sequenceBasePath = "";
	sequenceName = "";
	resultsPath = "";

	frameWidth = 320;
	frameHeight = 240;

	seed = 0;
	searchRadius = 30;
	svmC = 1.0;
	svmBudgetSize = 0;

	features.clear();
}

std::string Config::FeatureName(FeatureType f)
{
	switch (f)
	{
	case kFeatureTypeRaw:
		return "raw";
	case kFeatureTypeHaar:
		return "haar";
	case kFeatureTypeHistogram:
		return "histogram";
	default:
		return "";
	}
}

std::string Config::KernelName(KernelType k)
{
	switch (k)
	{
	case kKernelTypeLinear:
		return "linear";
	case kKernelTypeGaussian:
		return "gaussian";
	case kKernelTypeIntersection:
		return "intersection";
	case kKernelTypeChi2:
		return "chi2";
	default:
		return "";
	}
}

ostream& operator<< (ostream& out, const Config& conf)
{
	out << "config:" << endl;
	out << "  sequenceBasePath   = " << conf.sequenceBasePath << endl;
	out << "  sequenceName       = " << conf.sequenceName << endl;
	out << "  resultsPath        = " << conf.resultsPath << endl;
	out << "  frameWidth         = " << conf.frameWidth << endl;
	out << "  frameHeight        = " << conf.frameHeight << endl;
	out << "  seed               = " << conf.seed << endl;
	out << "  searchRadius       = " << conf.searchRadius << endl;
	out << "  svmC               = " << conf.svmC << endl;
	out << "  svmBudgetSize      = " << conf.svmBudgetSize << endl;

	for (int i = 0; i < (int)conf.features.size(); ++i)
	{
		out << "  feature " << i << endl;
		out << "    feature: " << Config::FeatureName(conf.features[i].feature) << endl;
		out << "    kernel:  " << Config::KernelName(conf.features[i].kernel) << endl;
		if (conf.features[i].params.size() > 0)
		{
			out << "    params: ";
			for (int j = 0; j < (int)conf.features[i].params.size(); ++j)
			{
				out << " " << conf.features[i].params[j];
			}
			out << endl;
		}
	}

	return out;
}