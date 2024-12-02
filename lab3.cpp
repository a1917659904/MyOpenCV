//
// Created by wyt on 24-12-2.
//
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include"cmake-build-debug/MyOpenCV.h"
#include"cmake-build-debug/demo.h"
using namespace cv;
using namespace std;


/*
int main() {
	String str1 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验三\\灰度.jpg";
	String str2 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验三\\彩色图像.jpg";
	Mat img1, img2, img3, img4;
	ImageRead(str1, img1);
	ImageRead(str2, img2);
	ImageOpen(img1, "灰度图像");
	ImageOpen(img2, "彩色图像");
	blur_demo(img1, 3);
	blur_demo(img1, 5);
	blur_demo(img1, 9);
	GaussianDemo(img1, 3);
	GaussianDemo(img1, 5);
	GaussianDemo(img1, 9);
	LaplacianDemom(img1, 3);
	Mat kernel1 = (cv::Mat_<float>(2, 2) << -1, 0, 0, 1);
	Mat kernel2 = (cv::Mat_<float>(2, 2) << 0, -1, 1, 0);
	RobertDemom(img1, kernel1, kernel2);
	SobelDemom(img1, 1, 1);
	HBFilter(img1, 5);
	blur_demo(img2, 9);
	GaussianDemo(img2, 9);
	LaplacianColorDemom(img2, 3);
	RobertColorDemom(img2, kernel1, kernel2);
	SobelColorDemom(img2, 1, 1);
	waitKey(0);
	return 0;
}
*/