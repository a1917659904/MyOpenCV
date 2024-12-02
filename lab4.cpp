//
// Created by wyt on 24-12-2.
//
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include"cmake-build-debug/demo.h"
#include"cmake-build-debug/MyOpenCV.h"
using namespace std;

int main() {
        String str1 = "C:\\Users\\wyt\\Desktop\\作业留档\\高级图像处理\\实验二\\灰度.jpg";
        String str2 = "C:\\Users\\wyt\\Desktop\\作业留档\\高级图像处理\\实验二\\彩色图像.jpg";
        Mat img1, img2;
        ImageRead(str1, img1);
        ImageRead(str2, img2);
        cvtColor(img1, img1, COLOR_BGR2GRAY);
        img1.convertTo(img1, CV_8U);
        ImageOpen(img1, "灰度图像");

        ////一
        GaussBlurDemo(img1);
        SaltBlurDemo(img1);
        PepperBlurDemo(img1);
        //SaltAndPepperBlurDemo(img1);

        //二、
        //MedianBlurDemo(img1, 5);
        //MedianBlurDemo(img1, 9);

        //三、
        //AdaptiveMeanBlurDemo(img1, 7, 70);

        //四、
        //AdaptiveMedianBlurDemo(img1, 7);
        //五、
        // 算数均值
        //ColorMeanBlurDemo(img2, 5);
        //几何均值
        //ColorGeoDemo(img2, 5);
        waitKey(0);
        return 0;
}
