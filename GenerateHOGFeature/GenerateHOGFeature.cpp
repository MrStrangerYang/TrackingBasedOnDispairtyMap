#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <math.h>
#include <Eigen\Core>
using namespace cv;
using namespace std;
using namespace Eigen;
//�Զ���������  
bool SortByM1(const float &v1, const float &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	return v1 < v2;//��������  
}
int main() {
	vector<float> v;
	v.push_back(1);
	v.push_back(12);
	v.push_back(11);
	v.push_back(6);
	v.push_back(7);
	std::cout << "Before Sort:" << std::endl;
	for (int i = 0;i < v.size();i++)
		cout << v[i] << "  ";
	cout << endl;

	sort(v.begin(), v.end(), SortByM1);
	std::cout << "After Sort:" << std::endl;
	for (int i = 0;i < v.size();i++)
		cout << v[i] << "  ";
	cout << endl;

	return 0;

}