//
// Created by wyt on 24-12-2.
//
#include"cmake-build-debug/demo.h"
#include"cmake-build-debug/MyOpenCV.h"
//1、5利用均值模板平滑灰度/彩色图像。
void blur_demo(Mat img, int size) {
        //img:传入图像
        //size:模板大小
        Mat imgBlur(img.size(), img.type());
        blur(img, imgBlur, Size(size, size));
        string name = "均值平滑";
        string sizeName = to_string(size) + "*" + to_string(size);
        sizeName += name;
        ImageOpen(imgBlur, sizeName);
}

//2、6利用高斯模板平滑灰度/彩色图像。
void GaussianDemo(Mat img, int size) {
        //img:传入图像
        //size:模板大小
        Mat imgBlur(img.size(), img.type());
        GaussianBlur(img, imgBlur, Size(size, size), 0);
        string name = "高斯平滑";
        string sizeName = to_string(size) + "*" + to_string(size);
        sizeName += name;
        ImageOpen(imgBlur, sizeName);
}

//3、利用Laplacian、Robert、Sobel模板锐化灰度图像。
//(1)Laplacian
void LaplacianDemom(Mat img, int size) {
        //img:传入图像
        //size:int类型的ksize，用于计算二阶导数的滤波器的核的大小，必须是正奇数，默认值是1.
        //void Laplacian(InputArray src, outputArray dst, int ddepth, int ksize=1, double scale=1, doubel delta=0, int borderType=BORDER_DEFAULT)
        //第一个参数：输入图像，Mat类的对象即可，需为单通道的8位图像。
        //第二个参数：输出的边缘图，需要和输入图像有一样的尺寸和通道数。
        //第三个参数：int类型的ddepth，目标图像的深度。
        //第四个参数：int类型的ksize，用于计算二阶导数的滤波器的核的大小，必须是正奇数，默认值是1.
        //第五个参数：double类型的scale，计算拉普拉斯值的时候可选的比例因子，默认值是1.
        //第六个参数：double类型的delta，表示结果存入目标图之前可选的delta值，默认值是0.
        //第七个参数：int类型borderType。
        Mat imgLaplacian(img.size(), img.type());
        Laplacian(img, imgLaplacian, 8, size);
        string name = "Laplacian";
        ImageOpen(imgLaplacian, name);
}
//(2)Robert
void RobertDemom(Mat img, Mat kernel1, Mat kernel2) {
        //img:输入的图像
        //kernel1:方向1的卷积核
        //kernel2:方向2的卷积核
        Mat imgx(img.size(), img.type());
        Mat imgy(img.size(), img.type());
        Mat imgRobert(img.size(), img.type());
        //利用filter2D进行处理
        filter2D(img, imgx, -1, kernel1);
        filter2D(img, imgy, -1, kernel2);
        //结果取绝对值
        convertScaleAbs(imgx, imgx);
        convertScaleAbs(imgy, imgy);
        //转换为二值图
        //threshold(imgx, imgx, 30, 255, 0);
        //threshold(imgy, imgy, 30, 255, 0);
        imgRobert = imgx + imgy;
        string name = "Roberts";
        ImageOpen(imgRobert, name);
}
//(3)Sobel
void SobelDemom(Mat img, int dx, int dy) {
        //img:传入图像
        // dx:x方向的导数阶数
        // dy:y方向的导数阶数
        //void Sobel(InputArray src, OutputArray dst, int ddpth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);
        //	第一个参数：输入图像，Mat类型即可；
        //	第二个参数：输出，需要和源图像有一样的大小和类型；
        //	第三个参数：int类型的ddepth，输入图像的深度
        //	第四个参数：int类型的dx，x方向上的差分阶数。
        //	第五个参数：it类型dy，y方向上的差分阶数。
        //	第六个参数：int类型的ksize，默认值是3，卷积核的大小，只能取1，3，5，7.
        //	第七个参数：double类型的scale，计算导数时的可选尺度因子，默认值是1，表示默认情况下是没有应用缩放的。
        //	第八个参数：double类型的delta，表示在结果存入目标图之前可选的delta值，默认值是0.
        //	第九个参数：int类型的borderType，边界模式，默认值是BORDER_DEFALULT。
        Mat imgSobel(img.size(), img.type());

        Sobel(img, imgSobel, CV_16S, dx, dy);
        convertScaleAbs(imgSobel, imgSobel);
        string name = "Sobel";
        ImageOpen(imgSobel, name);
}

//4、利用高提升滤波算法增强灰度图像。
void HBFilter(Mat img, int k) {
        //img:输入的图像
        //k:倍数
        Mat blurs;
        GaussianBlur(img, blurs, Size(5, 5), 0);
        Mat mask;
        Mat HB;
        subtract(img, blurs, mask);
        add(img, k * mask, HB);
        string name = "高提升滤波";
        ImageOpen(HB, name);
}

//7.利用Laplacian、Robert、Sobel模板锐化彩色图像。
//(1)Laplacian
void LaplacianColorDemom(Mat img, int size) {
        //img:传入图像
        //size:int类型的ksize，用于计算二阶导数的滤波器的核的大小，必须是正奇数，默认值是1.
        //void Laplacian(InputArray src, outputArray dst, int ddepth, int ksize=1, double scale=1, doubel delta=0, int borderType=BORDER_DEFAULT)
        //第一个参数：输入图像，Mat类的对象即可，需为单通道的8位图像。
        //第二个参数：输出的边缘图，需要和输入图像有一样的尺寸和通道数。
        //第三个参数：int类型的ddepth，目标图像的深度。
        //第四个参数：int类型的ksize，用于计算二阶导数的滤波器的核的大小，必须是正奇数，默认值是1.
        //第五个参数：double类型的scale，计算拉普拉斯值的时候可选的比例因子，默认值是1.
        //第六个参数：double类型的delta，表示结果存入目标图之前可选的delta值，默认值是0.
        //第七个参数：int类型borderType。
        Mat imgGray;
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        Mat imgLaplacian(img.size(), img.type());
        Laplacian(imgGray, imgLaplacian, 8, size);
        string name = "Laplacian彩色";
        ImageOpen(imgLaplacian, name);
}
//(2)Robert
void RobertColorDemom(Mat img, Mat kernel1, Mat kernel2) {
        //img:输入的图像
        //kernel1:方向1的卷积核
        //kernel2:方向2的卷积核
        Mat imgx(img.size(), img.type());
        Mat imgy(img.size(), img.type());
        Mat imgRobert(img.size(), img.type());
        Mat imgGray;
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        //利用filter2D进行处理
        filter2D(imgGray, imgx, -1, kernel1);
        filter2D(imgGray, imgy, -1, kernel2);
        //结果取绝对值
        convertScaleAbs(imgx, imgx);
        convertScaleAbs(imgy, imgy);
        //转换为二值图
        //threshold(imgx, imgx, 30, 255, 0);
        //threshold(imgy, imgy, 30, 255, 0);
        imgRobert = imgx + imgy;
        string name = "Roberts彩色";
        ImageOpen(imgRobert, name);
}
//(3)Sobel
void SobelColorDemom(Mat img, int dx, int dy) {
        //img:传入图像
        // dx:x方向的导数阶数
        // dy:y方向的导数阶数
        //void Sobel(InputArray src, OutputArray dst, int ddpth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);
        //	第一个参数：输入图像，Mat类型即可；
        //	第二个参数：输出，需要和源图像有一样的大小和类型；
        //	第三个参数：int类型的ddepth，输入图像的深度
        //	第四个参数：int类型的dx，x方向上的差分阶数。
        //	第五个参数：it类型dy，y方向上的差分阶数。
        //	第六个参数：int类型的ksize，默认值是3，卷积核的大小，只能取1，3，5，7.
        //	第七个参数：double类型的scale，计算导数时的可选尺度因子，默认值是1，表示默认情况下是没有应用缩放的。
        //	第八个参数：double类型的delta，表示在结果存入目标图之前可选的delta值，默认值是0.
        //	第九个参数：int类型的borderType，边界模式，默认值是BORDER_DEFALULT。
        Mat imgSobel(img.size(), img.type());
        Mat imgGray;
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        Sobel(imgGray, imgSobel, CV_16S, dx, dy);
        convertScaleAbs(imgSobel, imgSobel);
        string name = "Sobel彩色";
        ImageOpen(imgSobel, name);
}

void GaussBlurDemo(Mat img) {
        //img:原图
        Mat imgGauss;
        Mat imgGaussDemo1, imgGaussDemo2, imgGaussDemo3, imgGaussDemo4, imgGaussDemo5, imgGaussDemo6;
        ////高斯噪声
        MyGaussNoisy(img, imgGauss);
        ImageOpen(imgGauss, "高斯噪声");
        //算数均值滤波
        blur(imgGauss, imgGaussDemo1, Size(5, 5));
        ImageOpen(imgGaussDemo1, "算数均值滤波");
        //几何均值滤波
        GeometricBlur(imgGauss, imgGaussDemo2, 5);
        ImageOpen(imgGaussDemo2, "几何均值滤波");
        //谐波均值滤波
        HarmonicBlur(imgGauss, imgGaussDemo3, 5);
        ImageOpen(imgGaussDemo3, "谐波均值滤波");
        //逆谐波均值滤波
        InverseHarmonicBlur(imgGauss, imgGaussDemo4, 1.5, 5);
        ImageOpen(imgGaussDemo4, "逆谐波均值滤波");
        ////中值滤波
        //MedianBlur(imgGauss, imgGaussDemo5, 7);
        //ImageOpen(imgGaussDemo5, "中值滤波");
        ////自适应均值滤波


}

void SaltBlurDemo(Mat img) {
        //img:原图
        Mat imgSalt;
        Mat imgSaltDemo1, imgSaltDemo2, imgSaltDemo3, imgSaltDemo4;
        //盐噪声
        MySaltNoisy(img, imgSalt);
        ImageOpen(imgSalt, "盐噪声");
        //算数均值滤波
        blur(imgSalt, imgSaltDemo1, Size(5, 5));
        ImageOpen(imgSaltDemo1, "算数均值滤波");
        //几何均值滤波
        GeometricBlur(imgSalt, imgSaltDemo2, 5);
        ImageOpen(imgSaltDemo2, "几何均值滤波");
        //谐波均值滤波
        HarmonicBlur(imgSalt, imgSaltDemo3, 5);
        ImageOpen(imgSaltDemo3, "谐波均值滤波");
        //逆谐波均值滤波
        InverseHarmonicBlur(imgSalt, imgSaltDemo4, 1.5, 5);
        ImageOpen(imgSaltDemo4, "逆谐波均值滤波");
}

void PepperBlurDemo(Mat img) {
        //img:原图
        Mat imgPepper;
        Mat imgPepperDemo1, imgPepperDemo2, imgPepperDemo3, imgPepperDemo4;
        //椒噪声
        MyPepperNoisy(img, imgPepper);
        ImageOpen(imgPepper, "胡椒噪声");
        //算数均值滤波
        blur(imgPepper, imgPepperDemo1, Size(5, 5));
        ImageOpen(imgPepperDemo1, "算数均值滤波");
        //几何均值滤波
        GeometricBlur(imgPepper, imgPepperDemo2, 5);
        ImageOpen(imgPepperDemo2, "几何均值滤波");
        //谐波均值滤波
        HarmonicBlur(imgPepper, imgPepperDemo3, 5);
        ImageOpen(imgPepperDemo3, "谐波均值滤波");
        //逆谐波均值滤波
        InverseHarmonicBlur(imgPepper, imgPepperDemo4, 1.5, 5);
        ImageOpen(imgPepperDemo4, "逆谐波均值滤波");
}

void SaltAndPepperBlurDemo(Mat img) {
        //img:原图
        Mat imgSaltAndPepper;
        Mat imgSaltAndPepperDemo1, imgSaltAndPepperDemo2, imgSaltAndPepperDemo3, imgSaltAndPepperDemo4;
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");
        //算数均值滤波
        blur(imgSaltAndPepper, imgSaltAndPepperDemo1, Size(5, 5));
        ImageOpen(imgSaltAndPepperDemo1, "算数均值滤波");
        //几何均值滤波
        GeometricBlur(imgSaltAndPepper, imgSaltAndPepperDemo2, 5);
        ImageOpen(imgSaltAndPepperDemo2, "几何均值滤波");
        //谐波均值滤波
        HarmonicBlur(imgSaltAndPepper, imgSaltAndPepperDemo3, 5);
        ImageOpen(imgSaltAndPepperDemo3, "谐波均值滤波");
        //逆谐波均值滤波
        InverseHarmonicBlur(imgSaltAndPepper, imgSaltAndPepperDemo4, 1.5, 5);
        ImageOpen(imgSaltAndPepperDemo4, "逆谐波均值滤波");

}

void MedianBlurDemo(Mat img, int size) {
        //img:原图
        //size:滤波器尺寸
        Mat imgSaltAndPepper, imgSaltAndPepperDemo;
        Mat imgPepper, imgPepperDemo;
        Mat imgSalt, imgSaltDemo;
        Mat imgGauss, imgGaussDemo;
        Mat imgMedianBlur;
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");
        //椒噪声
        MyPepperNoisy(img, imgPepper);
        ImageOpen(imgPepper, "胡椒噪声");
        //盐噪声
        MySaltNoisy(img, imgSalt);
        ImageOpen(imgSalt, "盐噪声");
        //高斯噪声
        MyGaussNoisy(img, imgGauss);
        ImageOpen(imgGauss, "高斯噪声");
        //中值滤波
        MedianBlur(imgSaltAndPepper, imgMedianBlur, size);
        string name = "椒盐-中值滤波";
        string sizeName = to_string(size) + "*" + to_string(size);
        ImageOpen(imgMedianBlur, sizeName + name);

        MedianBlur(imgSalt, imgMedianBlur, size);
        name = "盐-中值滤波";
        ImageOpen(imgMedianBlur, sizeName + name);

        MedianBlur(imgPepper, imgMedianBlur, size);
        name = "椒-中值滤波";

        ImageOpen(imgMedianBlur, sizeName + name);

        MedianBlur(imgGauss, imgMedianBlur, size);
        name = "高斯-中值滤波";
        ImageOpen(imgMedianBlur, sizeName + name);
}
void AdaptiveMeanBlurDemo(Mat img, int size, int std) {
        //img:原图
        //size:滤波器尺寸
        Mat imgSaltAndPepper, imgSaltAndPepperDemo;
        Mat imgPepper, imgPepperDemo;
        Mat imgSalt, imgSaltDemo;
        Mat imgGauss, imgGaussDemo;
        Mat imgAdaMeanBlur;
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");
        //椒噪声
        MyPepperNoisy(img, imgPepper);
        ImageOpen(imgPepper, "胡椒噪声");
        //盐噪声
        MySaltNoisy(img, imgSalt);
        ImageOpen(imgSalt, "盐噪声");
        //高斯噪声
        MyGaussNoisy(img, imgGauss);
        ImageOpen(imgGauss, "高斯噪声");
        //自适应均值滤波
        AdaptiveMeanBlur(imgGauss, imgAdaMeanBlur, size, std);
        ImageOpen(imgGauss, "高斯-自适应均值滤波");

        AdaptiveMeanBlur(imgSalt, imgAdaMeanBlur, size, std);
        ImageOpen(imgAdaMeanBlur, "盐-自适应均值滤波");

        AdaptiveMeanBlur(imgPepper, imgAdaMeanBlur, size, std);
        ImageOpen(imgAdaMeanBlur, "胡椒-自适应均值滤波");

        AdaptiveMeanBlur(imgSaltAndPepper, imgAdaMeanBlur, size, std);
        ImageOpen(imgAdaMeanBlur, "椒盐-自适应均值滤波");
}

void AdaptiveMedianBlurDemo(Mat img, int size) {
        //img:原图
        //size:滤波器尺寸
        Mat imgSaltAndPepper, imgSaltAndPepperDemo;
        Mat imgPepper, imgPepperDemo;
        Mat imgSalt, imgSaltDemo;
        Mat imgGauss, imgGaussDemo;
        Mat imgMedianBlur;
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");
        //自适应中值滤波
        AdaptiveMedianBlur(imgSaltAndPepper, imgMedianBlur, size);
        string name = "椒盐-自适应中值滤波";
        string sizeName = " ";
        ImageOpen(imgMedianBlur, sizeName + name);
}

void ColorMeanBlurDemo(Mat img, int size) {
        //img:原图
        //size:滤波器尺寸
        Mat imgSaltAndPepper, imgSaltAndPepperDemo;
        Mat imgPepper, imgPepperDemo;
        Mat imgSalt, imgSaltDemo;
        Mat imgGauss, imgGaussDemo;
        Mat imgMeanBlur;
        Mat imgGeoBlur;

        //椒噪声
        MyPepperNoisy(img, imgPepper, 3);
        ImageOpen(imgPepper, "胡椒噪声");
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper, 3);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");

        //盐噪声
        MySaltNoisy(img, imgSalt, 3);
        ImageOpen(imgSalt, "盐噪声");
        //高斯噪声
        MyGaussNoisy(img, imgGauss, 3, 0, 30);
        ImageOpen(imgGauss, "高斯噪声");

        blur(imgSalt, imgSaltDemo, Size(size, size));
        ImageOpen(imgSaltDemo, "盐-算数均值滤波");

        blur(imgPepper, imgPepperDemo, Size(size, size));
        ImageOpen(imgPepperDemo, "胡椒-算数均值滤波");

        blur(imgGauss, imgGaussDemo, Size(size, size));
        ImageOpen(imgGaussDemo, "高斯-算数均值滤波");

        blur(imgSaltAndPepper, imgSaltAndPepperDemo, Size(size, size));
        ImageOpen(imgSaltAndPepperDemo, "椒盐-算数均值滤波");

}

void ColorGeoDemo(Mat img, int size) {
        //img:原图
        //size:滤波器尺寸
        Mat imgSaltAndPepper, imgSaltAndPepperDemo;
        Mat imgPepper, imgPepperDemo;
        Mat imgSalt, imgSaltDemo;
        Mat imgGauss, imgGaussDemo;
        Mat imgMeanBlur;
        Mat imgGeoBlur;
        //椒噪声
        MyPepperNoisy(img, imgPepper, 3);
        ImageOpen(imgPepper, "胡椒噪声");
        //椒盐噪声
        MySaltAndPepperNoisy(img, imgSaltAndPepper, 3);
        ImageOpen(imgSaltAndPepper, "椒盐噪声");

        //盐噪声
        MySaltNoisy(img, imgSalt, 3);
        ImageOpen(imgSalt, "盐噪声");
        //高斯噪声
        MyGaussNoisy(img, imgGauss, 3, 0, 30);
        ImageOpen(imgGauss, "高斯噪声");

        GeometricBlur(imgSalt, imgSaltDemo, 5, 3);
        ImageOpen(imgSaltDemo, "盐-几何均值滤波");

        GeometricBlur(imgPepper, imgPepperDemo, 5, 3);
        ImageOpen(imgPepperDemo, "胡椒-几何均值滤波");

        GeometricBlur(imgGauss, imgGaussDemo, 5, 3);
        ImageOpen(imgGaussDemo, "高斯-几何均值滤波");

        GeometricBlur(imgSaltAndPepper, imgSaltAndPepperDemo, 5, 3);
        ImageOpen(imgSaltAndPepperDemo, "椒盐-几何均值滤波");
}

void DFTDemo(Mat img, Mat& imgDFT, int flag) {
        //img:原图
        //imgDFT:dft变换

        //DFT变换
        resize(img, img, Size(512, 512)); // 保证图像大小合适

        // DFT变换
        Mat padded;
        int m = getOptimalDFTSize(img.rows);
        int n = getOptimalDFTSize(img.cols);
        copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
        Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
        Mat complexImg;
        merge(planes, 2, complexImg);
        dft(complexImg, complexImg);

        // 创建滤波器
        int radius; // 截止频率
        cout << "请输入截止频率： ";
        cin >> radius;
        Mat temp = createLowHighPassFilter(padded.size(), radius, true); // 创建低通滤波器
        Mat planesFilter[] = { temp, Mat::zeros(temp.size(), CV_32F) };
        Mat lowPassFilter;
        merge(planesFilter, 2, lowPassFilter);

        temp = createLowHighPassFilter(padded.size(), radius, false); // 创建高通滤波器
        Mat highPassFilter;
        merge(planesFilter, 2, highPassFilter);

        // 应用滤波器
        Mat filteredImg;
        if (flag) {
                mulSpectrums(complexImg, lowPassFilter, filteredImg, 0); // 应用低通滤波器
        }
        else {
                mulSpectrums(complexImg, highPassFilter, filteredImg, 0); // 应用高通滤波器
        }

        // IDFT变换
        Mat inverseTransform;
        dft(filteredImg, inverseTransform, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

        // 显示结果
        Mat finalImg;
        normalize(inverseTransform, finalImg, 0, 255, NORM_MINMAX);
        finalImg.convertTo(finalImg, CV_8U);
        imshow("滤波后", finalImg);

}


void DFTButterDemo(Mat img, int flag) {
        resize(img, img, Size(512, 512)); // 保证图像大小合适

        // DFT变换
        Mat padded;
        int m = getOptimalDFTSize(img.rows);
        int n = getOptimalDFTSize(img.cols);
        copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
        Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
        Mat complexImg;
        merge(planes, 2, complexImg);
        dft(complexImg, complexImg);

        // 用户输入
        int d0, filterOrder;
        cout << "输入截止频率: ";
        cin >> d0;
        cout << "滤波器阶数: ";
        cin >> filterOrder;

        // 创建布特沃斯滤波器
        Mat lowPassFilter = createButterworthFilter(padded.size(), d0, filterOrder, true);
        Mat highPassFilter = createButterworthFilter(padded.size(), d0, filterOrder, false);

        // 应用滤波器
        Mat filteredImg;
        // 根据需要选择应用哪个滤波器
        if (flag) {
                mulSpectrums(complexImg, lowPassFilter, filteredImg, 0); // 应用低通滤波器
        }
        else {
                mulSpectrums(complexImg, highPassFilter, filteredImg, 0); // 应用高通滤波器
        }

        // IDFT变换
        Mat inverseTransform;
        dft(filteredImg, inverseTransform, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

        // 显示结果
        Mat finalImg;
        normalize(inverseTransform, finalImg, 0, 255, NORM_MINMAX);
        finalImg.convertTo(finalImg, CV_8U);
        imshow("布特沃滤波后的图像", finalImg);
}



