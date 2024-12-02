//
// Created by wyt on 24-12-2.
//

#ifndef MYOPENCV_MYOPENCV_H
#define MYOPENCV_MYOPENCV_H
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<cmath>
#include<ctime>
#define PI 3.14159
using namespace cv;
using namespace std;
void ImageRead(String imgPath, Mat& image);
void ImageOpen(Mat& image, String windowName);
void MyThreshold(Mat image, Mat& binary, int thresholdvValue = 128);
void MyLogChange(Mat image, Mat& logImg, int normType = NORM_MINMAX, double c = 1.0);
void MyExpChange(Mat image, Mat& logImg, int normType = NORM_MINMAX, double c = 1.0);
void MyGammaChange(Mat image, Mat& logImg, int normType = NORM_MINMAX, double gamma = 0.5, double c = 1.0);
void MyComplementaryChange(Mat image, Mat& reverseImg);
void MyCalcHist(Mat img, double *p);
void Draw(double* p, String imgName, int k = 1);
void MyEqualizeHist(Mat image, Mat& img, double* p, int* shadow, double* prob);
void MyMerge(Mat imgB, Mat imgG, Mat imgR, Mat& image);
void MySplit(Mat image, Mat& imgB, Mat& imgG, Mat& imgR);
void MyColorEqualize(Mat image, Mat& imageE);
void MyGaussNoisy(Mat image, Mat& imgNoisy, double mean = 0, double std = 30);
void MySaltNoisy(Mat image, Mat& imgNoisy);
void MyPepperNoisy(Mat image, Mat& imgNoisy);
void MySaltAndPepperNoisy(Mat image, Mat& imgNoisy);
void GeometricBlur(Mat img, Mat& imgBlur, int size);
void HarmonicBlur(Mat img, Mat& imgBlur, int size);
void InverseHarmonicBlur(Mat img, Mat& imgBlur, double Q, int size);
void SaltAndPepperBlurDemo(Mat img);
void MedianBlur(Mat img, Mat& imgBlur, int size);
void AdaptiveMeanBlur(Mat img, Mat& imgBlur, int size, int std);
void AdaptiveMedianBlur(Mat img, Mat& imgBlur, int size);
void GeometricBlur(Mat img, Mat& imgBlur, int size, int channel);
void MyGaussNoisy(Mat image, Mat& imgNoisy, int channel , double mean, double std);
void MySaltNoisy(Mat image, Mat& imgNoisy, int channel);
void MySaltAndPepperNoisy(Mat image, Mat& imgNoisy, int channel);
void MyPepperNoisy(Mat image, Mat& imgNoisy, int channel);
Mat createLowHighPassFilter(const cv::Size& sz, int radius, bool isLowPass);
Mat createButterworthFilter(const cv::Size& sz, int d0, int n, bool isLowPass);
#endif MYOPENCV_MYOPENCV_H
