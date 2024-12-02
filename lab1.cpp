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
	String str1 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验一\\灰度图像.jpg";
	Mat img1, img2, img3, img4, img5, img6;
	ImageRead(str1, img1);
	ImageOpen(img1, "灰度图像");
	MyThreshold(img1,img2);
	ImageOpen(img2,"灰度图像二值化处理后");
	MyLogChange(img1, img3, NORM_MINMAX);
	ImageOpen(img3, "灰度图像的对数变换");
	MyGammaChange(img1, img4, NORM_MINMAX, 2);
	ImageOpen(img4, "灰度图像的伽马变换");
	String str2 = "C:\\Users\\wyt\\Desktop\\高级图像处理\\实验一\\彩色图像.jpg";
	ImageRead(str2, img5);
	ImageOpen(img5, "彩色图像");
	MyComplementaryChange(img5, img6);
	ImageOpen(img6, "彩色图像的补色变换");
	waitKey(0);
	return 0;
}
*/