//
// Created by wyt on 24-12-2.
//

#ifndef MYOPENCV_DEMO_H
#define MYOPENCV_DEMO_H
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace cv;
using namespace std;

void blur_demo(Mat img, int size);
void GaussianDemo(Mat img, int size);
void LaplacianDemom(Mat img, int size);
void RobertDemom(Mat img, Mat kernel1, Mat kernel2);
void SobelDemom(Mat img, int dx, int dy);
void HBFilter(Mat img, int k);
void LaplacianColorDemom(Mat img, int size);
void RobertColorDemom(Mat img, Mat kernel1, Mat kernel2);
void SobelColorDemom(Mat img, int dx, int dy);
void GaussBlurDemo(Mat img);
void SaltBlurDemo(Mat img);
void PepperBlurDemo(Mat img);
void MedianBlurDemo(Mat img, int size);
void AdaptiveMeanBlurDemo(Mat img, int size, int std);
void AdaptiveMedianBlurDemo(Mat img, int size);
void ColorMeanBlurDemo(Mat img, int size);
void ColorGeoDemo(Mat img, int size);
void DFTDemo(Mat img, Mat& imgDFT, int flag = 1);
void DFTButterDemo(Mat img, int flag);
#endif MYOPENCV_DEMO_H
