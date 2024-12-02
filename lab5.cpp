//
// Created by wyt on 24-12-2.
//
//#include<opencv2/opencv.hpp>
//#include<iostream>
//#include<string>
//#include"demo.h"
//#include"MyOpenCV.h"
//using namespace std;
//
//int main() {
//	String str1 = "C:\\Users\\wyt\\Desktop\\作业留档\\高级图像处理\\实验二\\灰度.jpg";
//	String str2 = "C:\\Users\\wyt\\Desktop\\作业留档\\高级图像处理\\实验二\\彩色图像.jpg";
//	Mat img1, img2;
//	ImageRead(str1, img1);
//	cout << img1.type();
//	cvtColor(img1, img1, COLOR_BGR2GRAY);
//	img1.convertTo(img1, CV_8U);
//	ImageOpen(img1, "灰度图像");
//	Mat imgDFT1;
//	DFTDemo(img1, imgDFT1);
//	//DFTButterDemo(img1, 1);
//
//
//	waitKey(0);
//	return 0;
//}