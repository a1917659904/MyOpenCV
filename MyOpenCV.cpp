//
// Created by wyt on 24-12-2.
//
#include"MyOpenCV.h"

void ImageRead(String imgPath, Mat& image) {
        //ImageName:输入图像路径
        //image：存入的图像名
        image = imread(imgPath);
}

void ImageOpen(Mat &image, String windowName) {
        //image:输出图像名
        //WindowName:输出窗口名
        imshow(windowName, image);
}



void MyThreshold(Mat image, Mat& binary, int thresholdvValue) {
        //imgPath:图像路径
        //binary:返回的图像
        //threshold_value:二值化阈值
        binary.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = binary.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        if (inData[i] > thresholdvValue) {
                                outData[i] = 255;
                        }
                        else {
                                outData[i] = 0;
                        }
                }
        }


}

void MyLogChange(Mat image, Mat& logImg, int normType, double c ) {
        //imgPath:图像路径
        //logImg:返回的图像
        //norm_Type:归一化处理方式
        //c:对数函数系数
        logImg.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = logImg.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        outData[i] = c * log((double)inData[i] + 1);
                }
        }
        normalize(logImg, logImg, 0, 255, normType);
}



void MyExpChange(Mat image, Mat& expImg, int normType, double c) {
        //imgPath:图像路径
        //logImg:返回的图像
        //norm_Type:归一化处理方式
        //c:对数函数系数
        expImg.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = expImg.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        outData[i] = c * exp((double)inData[i]);
                }
        }
        normalize(expImg, expImg, 0, 255, normType);
}

void MyDivedeChange(Mat image, Mat& logImg, int normType, double c) {
        //imgPath:图像路径
        //logImg:返回的图像
        //norm_Type:归一化处理方式
        //c:对数函数系数
        logImg.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = logImg.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        outData[i] = c * log((double)inData[i] + 1);
                }
        }
        normalize(logImg, logImg, 0, 255, normType);
}
void MyGammaChange(Mat image, Mat& logImg, int normType, double gamma, double c) {
        //imgPath:图像路径
        //logImg:返回的图像
        //norm_Type:归一化处理方式
        //gamma:幂次
        //c：幂函数系数
        logImg.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = logImg.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        outData[i] = c * pow((double)inData[i], gamma);
                }
        }
        normalize(logImg, logImg, 0, 255, normType);
}

void MyComplementaryChange(Mat image, Mat& reverseImg) {
        //imgPath:图像路径
        //reverseImg:返回的图像
        reverseImg.create(image.size(), image.type());
        Mat_<Vec3b> im = image;
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        int b = im(i, j)[0];
                        int g = im(i, j)[1];
                        int r = im(i, j)[2];
                        int maxrgb = max(max(b, g), r);
                        int minrgb = min(min(b, g), r);
                        reverseImg.at<Vec3b>(i, j)[0] = maxrgb + minrgb - b;
                        reverseImg.at<Vec3b>(i, j)[1] = maxrgb + minrgb - g;
                        reverseImg.at<Vec3b>(i, j)[2] = maxrgb + minrgb - r;
                }
        }
}
void Draw(double* p, String imgName, int k) {
        //p：传入的直方图数据
        //imgName:输出图像名称
        //k：直方图放大倍数
        int histH = 512;
        int histW = 400;
        int binW = cvRound((double)histW / 256);
        Mat histCanvas = Mat::zeros(histH, histW, CV_8UC3);
        for (int i = 1; i < 256; i++) {
                line(histCanvas, Point(binW * (i - 1), histH - p[i - 1] * histH * k), Point(binW * i, histH - p[i] * histH * k),
                     Scalar(0, 0, 255), 2, 8, 0);
        }
        ImageOpen(histCanvas, imgName);
}
void MyCalcHist(Mat image, double *p) {
        //image：传入的图像
        //p：传出的直方图数据
        Mat_<uchar> im = image;
        int cnt[256] = { 0 };
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        cnt[int(im(i, j))]++;
                }
        }
        int size = image.rows * image.cols;
        for (int i = 0; i < 256; i++) {
                p[i] = double(cnt[i]) / size;
        }
        //Draw(p, "归一化直方图", k);
}

void MyEqualizeHist(Mat image, Mat& img, double* p, int *shadow, double *prob) {
        //image:等待均衡的图像
        // img:均衡后的图像
        // p:传入的直方图数据
        //shadow:传出的均衡前与均衡后的对应关系
        // prob:均衡后的概率分布
        for (int i = 0; i < 256; i++) {
                double sum = p[i];
                for (int j = 0; j < i; j++) {
                        sum += p[j];
                }
                int index = round(255.0 * sum);
                shadow[i] = index;//原始像素值i和变换后的像素值index的映射关系
                prob[index] += p[i];
        }
        img.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = img.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        outData[i] = shadow[inData[i]];
                }
        }

}


void MySplit(Mat image,Mat& imgB, Mat& imgG, Mat& imgR) {
        //image:传入的彩色图像
        // imgB:蓝色分量
        // imgG:绿色分量
        // imgR:红色分量
        //int es = image.elemSize();
        uchar* pointImage = image.data;
        uchar* pointImgR = imgR.data;
        uchar* pointImgG = imgG.data;
        uchar* pointImgB = imgB.data;
        size_t stepImage = image.step;
        size_t stepImageGray = imgR.step;
        //分离通道
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        /*(&imgB.data[i * image.step])[j * es	] = (&image.data[i * image.step])[j * es];
                        (&imgG.data[i * image.step])[j * es] = (&image.data[i * image.step])[j * es + 1];
                        (&imgR.data[i * image.step])[j * es] = (&image.data[i * image.step])[j * es + 2];*/
                        pointImgB[i * stepImageGray + j] = uchar(pointImage[i * stepImage + 3 * j]);
                        pointImgG[i * stepImageGray + j] = uchar(pointImage[i * stepImage + 3 * j + 1]);
                        pointImgR[i * stepImageGray + j] = uchar(pointImage[i * stepImage + 3 * j + 2]);
                }
        }
}
void MyMerge(Mat imgB, Mat imgG, Mat imgR, Mat& image) {
        uchar* pointImage = image.data;
        uchar* pointImgR = imgR.data;
        uchar* pointImgG = imgG.data;
        uchar* pointImgB = imgB.data;
        size_t stepImage = image.step;
        size_t stepImageGray = imgR.step;
        //合并通道
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        pointImage[i * stepImage + 3 * j] = uchar(pointImgB[i * stepImageGray + j]);
                        pointImage[i * stepImage + 3 * j + 1] = uchar(pointImgG[i * stepImageGray + j]);
                        pointImage[i * stepImage + 3 * j + 2] = uchar(pointImgR[i * stepImageGray + j]);

                }
        }
}

void MyColorEqualize(Mat image, Mat& imageE) {
        //image:待均衡的彩色图像
        //Mat imgR(image.rows, image.cols, 16), imgG(image.rows, image.cols, 16), imgB(image.rows, image.cols, 16);
        Mat imgR(image.rows,image.cols,CV_8UC1), imgG(image.rows, image.cols, CV_8UC1), imgB(image.rows, image.cols, CV_8UC1);
        //int es = image.elemSize();
        //分离通道
        MySplit(image, imgB, imgG, imgR);
        ImageOpen(image, "彩色图像");
        ImageOpen(imgR, "红色分量");
        ImageOpen(imgG, "绿色分量");
        ImageOpen(imgB, "蓝色分量");
        double pB[256] = { 0.0 };
        double pG[256] = { 0.0 };
        double pR[256] = { 0.0 };
        double proB[256] = { 0.0 };
        double proG[256] = { 0.0 };
        double proR[256] = { 0.0 };
        Mat imgRE(image.rows, image.cols, CV_8UC1), imgGE(image.rows, image.cols, CV_8UC1), imgBE(image.rows, image.cols, CV_8UC1);
        int shB[256] = { 0 };
        int shG[256] = { 0 };
        int shR[256] = { 0 };
        //计算三原色的直方图，并且分别作均衡处理
        MyCalcHist(imgB, pB);
        MyEqualizeHist(imgB, imgBE, pB, shB, proB);
        MyCalcHist(imgG, pG);
        MyEqualizeHist(imgG, imgGE, pG, shG, proG);
        MyCalcHist(imgR, pR);
        MyEqualizeHist(imgR, imgRE, pR, shR, proR);
        imageE.create(image.rows, image.cols, CV_8UC3);
        //合并均衡处理后的图像
        MyMerge(imgBE, imgGE, imgRE, imageE);
}

double UNIFORM(int rd)
{
        int x;
        double y;
        //srand((unsigned)time(NULL));
        x = (rand() + rd) % 1000;      //x就是由基于系统时钟产生的随机数
        y = (double)(x);    //这个随机数和100求余的结果必然得到一个小与100的整数，然后强制转换成浮点数
        y /= 1000;
        return y;
}
int GaussRand(double mean, double std) {
        //mean:平均值
        //std;标准差
        //n:随机数数量
        //srand((unsigned)time(NULL));
        double u[2] = { UNIFORM(rand()), UNIFORM(756) };
        double A = sqrt((-2) * log(u[0]));
        double B = 2 * PI * u[1];
        double C = A * cos(B);
        int r = (mean + C * std);
        return r;
}
void MyGaussNoisy(Mat image, Mat& imgNoisy, double mean, double std) {
        //image:原图
        //imgNoisy:添加高斯噪声后的图
        //mean:高斯噪声的均值
        //std:高斯噪声的方差
        imgNoisy.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = imgNoisy.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        int r = GaussRand(mean, std);

                        if (r + inData[i] > 255 || r + inData[i] < 0) {
                                outData[i] = inData[i];
                        }
                        else {
                                outData[i] = inData[i] + r;
                        }
                }
        }
}


void MyGaussNoisy(Mat image, Mat& imgNoisy, int channel,  double mean, double std) {
        //image:原图
        //imgNoisy:添加高斯噪声后的图
        // channel:通道数
        //mean:高斯噪声的均值
        //std:高斯噪声的方差
        imgNoisy.create(image.size(), image.type());
        uchar* pointImage = image.data;
        uchar* pointImgNoisy = imgNoisy.data;
        size_t stepImage = image.step;
        size_t stepImageNoisy = imgNoisy.step;
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {

                        for (int k = 0; k < channel; k++) {
                                int r = GaussRand(mean, std);
                                if (r + pointImage[i * stepImage + channel * j + k] > 255 || r + pointImage[i * stepImage + channel * j + k] < 0) {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = pointImage[i * stepImage + channel * j + k];
                                }
                                else {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = r + pointImage[i * stepImage + channel * j + k];
                                }
                        }
                }
        }
}
void MySaltNoisy(Mat image, Mat& imgNoisy) {
        //image:原图
        //imgNoisy:添加噪声后的图

        imgNoisy.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = imgNoisy.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        int r = GaussRand(0, 50);
                        if (r % 100 == 0) {
                                outData[i] = 255;
                        }
                        else {
                                outData[i] = inData[i];
                        }

                }
        }
}


void MySaltNoisy(Mat image, Mat& imgNoisy, int channel) {
        //image:原图
        //imgNoisy:添加噪声后的图
        // channel:通道数

        imgNoisy.create(image.size(), image.type());
        uchar* pointImage = image.data;
        uchar* pointImgNoisy = imgNoisy.data;
        size_t stepImage = image.step;
        size_t stepImageNoisy = imgNoisy.step;
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        for (int k = 0; k < channel; k++) {
                                int r = GaussRand(0, 50);
                                if (r % 100 == 0) {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = 255;
                                }
                                else {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = pointImage[i * stepImage + channel * j + k];
                                }
                        }
                }
        }
}
void MyPepperNoisy(Mat image, Mat& imgNoisy) {
        //image:原图
        //imgNoisy:添加噪声后的图
        imgNoisy.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = imgNoisy.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        int r = GaussRand(0, 50);
                        if (r % 100 == 0) {
                                outData[i] = 0;
                        }
                        else {
                                outData[i] = inData[i];
                        }
                }
        }
}


void MyPepperNoisy(Mat image, Mat& imgNoisy, int channel) {
        //image:原图
        //imgNoisy:添加噪声后的图
        //channel:通道数
        imgNoisy.create(image.size(), image.type());
        uchar* pointImage = image.data;
        uchar* pointImgNoisy = imgNoisy.data;
        size_t stepImage = image.step;
        size_t stepImageNoisy = imgNoisy.step;
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        for (int k = 0; k < channel; k++) {
                                int r = GaussRand(0, 50);
                                if (r % 100 == 0) {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = 0;
                                }
                                else {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = pointImage[i * stepImage + channel * j + k];
                                }
                        }
                }
        }
}
void MySaltAndPepperNoisy(Mat image, Mat& imgNoisy) {
        // image:原图
        // imgNoisy:添加噪声后的图
        imgNoisy.create(image.size(), image.type());
        int nr = image.rows;
        int nl = image.cols * image.channels();
        for (int k = 0; k < nr; k++) {
                const uchar* inData = image.ptr<uchar>(k);
                uchar* outData = imgNoisy.ptr<uchar>(k);
                for (int i = 0; i < nl; i++) {
                        int r = GaussRand(0, 50);
                        if (r % 100 == 0) {
                                outData[i] = 255;
                        }
                        else if (r % 32 == 1) {
                                outData[i] = 0;
                        }
                        else {
                                outData[i] = inData[i];
                        }
                }
        }
}


void MySaltAndPepperNoisy(Mat image, Mat& imgNoisy, int channel) {
        //image:原图
        //imgNoisy:添加噪声后的图
        //channel:通道数
        imgNoisy.create(image.size(), image.type());
        uchar* pointImage = image.data;
        uchar* pointImgNoisy = imgNoisy.data;
        size_t stepImage = image.step;
        size_t stepImageNoisy = imgNoisy.step;
        for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                        for (int k = 0; k < channel; k++) {
                                int r = GaussRand(0, 50);
                                if (r % 100 == 0) {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = 0;
                                }
                                else if (r % 100 == 1) {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = 255;
                                }
                                else {
                                        pointImgNoisy[i * stepImageNoisy + channel * j + k] = pointImage[i * stepImage + channel * j + k];
                                }
                        }
                }
        }
}

void GeometricBlur(Mat img, Mat& imgBlur, int size) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        double mul = 1.0;
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        mul *= double(pointImg[(i + k) * stepImg + j + l]);
                                }
                        }
                        pointImageBlur[i * stepImgBlur + j] = pow(mul, 1.0 / (size * size));
                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}


void GeometricBlur(Mat img, Mat& imgBlur, int size, int channel) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        //channel:通道数
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<double>mul(channel, 1.0);
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        for (int t = 0; t < channel; t++) {
                                                mul[t] *= double(pointImg[(i + k) * stepImg + channel * (j + l) + t]);
                                        }
                                }
                        }
                        for(int t = 0; t < channel; t++) {
                                pointImageBlur[i * stepImgBlur + j * channel + t] = uchar(pow(mul[t], 1.0 / (size * size)));
                        }

                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

void HarmonicBlur(Mat img, Mat & imgBlur, int size) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        img += 1;
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<double>neighbors;
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        neighbors.push_back(double(pointImg[(i + k) * stepImg + j + l]));
                                }
                        }
                        double sum = 0.0;
                        for (int m = 0; m < neighbors.size(); m++) {
                                sum += 1.0 / neighbors[m];
                        }
                        pointImageBlur[i * stepImgBlur + j] = neighbors.size() / sum;
                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

void InverseHarmonicBlur(Mat img, Mat& imgBlur, double Q, int size) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<double>neighborsQ1;
                        vector<double>neighborsQ;
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        neighborsQ1.push_back(pow(double(pointImg[(i + k) * stepImg + j + l]), Q + 1));
                                        neighborsQ.push_back(pow(double(pointImg[(i + k) * stepImg + j + l]), Q));
                                }
                        }
                        double sumQ1 = 0.0;
                        double sumQ = 0.0;
                        for (int m = 0; m < neighborsQ1.size(); m++) {
                                sumQ1 += neighborsQ1[m];
                                sumQ += neighborsQ[m];
                        }
                        pointImageBlur[i * stepImgBlur + j] = sumQ1 / sumQ;
                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

void MedianBlur(Mat img, Mat& imgBlur, int size) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<int>neighbors;
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        neighbors.push_back(pointImg[(i + k) * stepImg + j + l]);
                                }
                        }
                        sort(neighbors.begin(), neighbors.end());
                        pointImageBlur[i * stepImgBlur + j] = neighbors[(size * size) / 2];
                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

void AdaptiveMeanBlur(Mat img, Mat& imgBlur, int size, int std) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        //std:方差
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<double>neighbors;
                        double sum = 0.0;
                        for (int k = -radius; k <= radius; k++) {
                                for (int l = -radius; l <= radius; l++) {
                                        neighbors.push_back(double(pointImg[(i + k) * stepImg + j + l]));
                                        sum += double(pointImg[(i + k) * stepImg + j + l]);
                                }
                        }


                        double mean = sum / neighbors.size();
                        if (pointImg[i * stepImg + j] > mean * 1.3|| pointImg[i * stepImg + j] < mean * 0.7) {
                                double std2 = 0;
                                int sub = 0;
                                for (int k = 0; k < neighbors.size(); k++) {
                                        sub += (neighbors[k] - mean) * (neighbors[k] - mean);
                                }
                                std2 = sqrt(sub / neighbors.size());
                                if (std2 < std) {
                                        pointImageBlur[i * stepImgBlur + j] = uchar(mean);
                                }
                        }

                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

void AdaptiveMedianBlur(Mat img, Mat& imgBlur, int size) {
        //img:原图
        //imgBlur:输出图像
        //Size:滤波器尺寸
        img.copyTo(imgBlur);
        imgBlur.create(img.rows, img.cols, img.type());
        int radius = size / 2;
        uchar* pointImg = img.data;
        uchar* pointImageBlur = imgBlur.data;
        size_t stepImg = img.step;
        size_t stepImgBlur = imgBlur.step;
        for (int i = radius; i < img.rows - radius; i++) {
                for (int j = radius; j < img.cols - radius; j++) {
                        vector<int>neighbors;
                        if (pointImageBlur[i * stepImgBlur + j] == 255 || pointImageBlur[i * stepImgBlur + j] == 0)
                        {
                                for (int k = -radius; k <= radius; k++) {
                                        for (int l = -radius; l <= radius; l++) {
                                                neighbors.push_back(pointImg[(i + k) * stepImg + j + l]);
                                        }
                                }
                                sort(neighbors.begin(), neighbors.end());
                                pointImageBlur[i * stepImgBlur + j] = neighbors[(size * size) / 2];
                        }
                }
        }
        normalize(imgBlur, imgBlur, 0, 255, 32);
}

Mat createLowHighPassFilter(const Size& sz, int radius, bool isLowPass) {
        Mat filter = Mat::zeros(sz, CV_32F);
        Point center = Point(sz.width / 2, sz.height / 2);
        for (int i = 0; i < sz.height; i++) {
                for (int j = 0; j < sz.width; j++) {
                        float distance = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
                        if (isLowPass) {
                                filter.at<float>(i, j) = (distance <= radius) ? 1.0 : 0.0;
                        }
                        else {
                                filter.at<float>(i, j) = (distance >= radius) ? 1.0 : 0.0;
                        }
                }
        }
        return filter;
}

Mat createButterworthFilter(const Size& sz, int d0, int n, bool isLowPass) {
        Mat filter = Mat::zeros(sz, CV_32F);
        Point center = Point(sz.width / 2, sz.height / 2);

        for (int i = 0; i < sz.height; i++) {
                for (int j = 0; j < sz.width; j++) {
                        float distance = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
                        float value = 1 / (1 + pow(distance / d0, 2 * n));

                        filter.at<float>(i, j) = isLowPass ? value : (1 - value);
                }
        }

        // 将滤波器从单通道转换为双通道
        Mat mergeImg[] = { filter, filter };
        merge(mergeImg, 2, filter);

        return filter;
}

