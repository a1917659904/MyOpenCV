//
// Created by wyt on 24-12-2.
//
/*
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include"MyOpenCV.h"

using namespace cv;
using namespace std;

int main() {
	String str1 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验二\\灰度.jpg";
	String str2 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验二\\彩色图像.jpg";
	Mat img1, img2, img3, img4;
	ImageRead(str1, img1);
	ImageRead(str2, img2);
	ImageOpen(img1, "灰度图像");
	double p[256];
	MyCalcHist(img1, p);
	Draw(p, "灰度图像的归一化直方图");
	int shadow[256] = { 0 };
	double prob[256] = { 0 };
	MyEqualizeHist(img1, img3, p, shadow, prob);
	Draw(prob, "灰度图像均衡后的直方图", 2);
	ImageOpen(img3, "灰度图像均衡后的图像");
	MyColorEqualize(img2, img4);
	ImageOpen(img4, "彩色图像均衡后的图像");
	waitKey(0);
	return 0;
}
*/