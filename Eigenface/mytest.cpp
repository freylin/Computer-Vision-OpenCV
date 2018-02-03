#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "cv.h"
#include <fstream>
#include <sstream>
#include <windows.h>  
#include <math.h>

using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // �����ͷ���һ����һ�����ͼ�����:
    Mat dst;
    switch(src.channels()) {
    case 1:
        //cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_32FC1);
		cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Mat cut(string path)
{
    string txtpath = path , line, sx4, sy4, sx3, sy3;
	// �۾��ļ�·��
	txtpath.erase(txtpath.find(".pgm"), 4);
	txtpath = txtpath+".txt";
	std::ifstream file((txtpath).c_str(), ifstream::in);
	getline(file, line);
	stringstream liness(line);
	//�۾�λ��
	getline(liness, sx4, ' ');
	getline(liness, sy4, ' ');
	getline(liness, sx3, ' ');
	getline(liness, sy3);
	int x3 = atoi(sx3.c_str()), y3 = atoi(sy3.c_str()), x4 = atoi(sx4.c_str()), y4 = atoi(sy4.c_str());
	
	//��תͼƬ��ʹ�۾���һ��ˮƽ����
	Mat m = imread(path, 0);
	Point2f center = cv::Point2f(x3, y3);  // ��ת����Ϊ���� 
	double angle = atan((double)(y4-y3)/(double)(x4-x3)) /3.1415926*180.0;  // ��ʱ����ת�Ƕ�
	x4 = x3 + sqrt((y4-y3)*(y4-y3)+(x4-x3)*(x4-x3));
	Mat rotateMat;   
	rotateMat = getRotationMatrix2D(center, angle, 1.0);  
	warpAffine(m, m, rotateMat, m.size());  

	int w = 120, h = 150 , w2 = m.cols, h2 = m.rows;
	double d = (60.0/(1.0*(x4-x3)));
	resize(m,m,Size(w2 * d, h2 * d));
	Size size = Size(w, h);
	int x5 = x3*d-(w-60)/2, y5 = y3*d - h/3;
	Mat m2 = Mat::zeros( size, 0 );
	for (int i  = y5; i < y5 + h; i++)
		if (i < h2 && i >=0)
			for (int j = x5; j < x5 + w; j++)
			if (j < w2 && j >=0)
				{
					m2.at<uchar>(i - y5 ,j - x5) = m.at<uchar>(i,j);
				} 
	Mat m3 = Mat::zeros( Size(w*2, h*2), 0 );
	resize(m2,m3, Size(w*2, h*2));
	return norm_0_255(m3);
}

// ��һ��ͼ�������ת��ΪRow Matrix�е�һ��
Mat asRowMatrix(Mat src, int rtype, double alpha = 1, double beta = 0) {
    size_t d = src.total();
    Mat data(1, d, rtype);
    // ��ͼ�����ݸ��Ƶ����������
    Mat x0 = data.row(0);
    // ��һ��ͼ��ӳ�䵽���ؾ����һ����:
    if(src.isContinuous()) {
        src.reshape(1, 1).convertTo(x0, rtype, alpha, beta);
    } else {
        src.clone().reshape(1, 1).convertTo(x0, rtype, alpha, beta);
    }
    return data;
}

double dis(Mat a, Mat b)
{
	double d = 0.0;
	for (int i = 0; i < a.rows; i++)
	{
		d += (a.at<float>(i, 0) - b.at<float>(i, 0)) * (a.at<float>(i, 0) - b.at<float>(i, 0));
	}
	return sqrt(d);
}

int main(int argc, const char*argv[]) {
	string exepath;
	exepath = argv[0];
	exepath.erase(exepath.find("mytest.exe"), 13);

	FileStorage fs(argv[2], FileStorage::READ);
	Mat eigenvectors;
	// ��ȡ���������� MN*k
	fs["eigenvectors"] >> eigenvectors;
	Mat Samples;
	// k*K
	fs["Samples"] >> Samples;
	Mat mean;
	fs["mean"] >> mean;
	// ��ʶ��ͼƬ 
	Mat image = cut(argv[1]);
	IplImage *Image;
	Image = cvLoadImage(argv[1]);
	imshow("test face", image);

	Mat f = asRowMatrix(image, CV_32FC1);
	// k*1
	Mat y = eigenvectors.t()*f.t();
	double mindis = 999999999;
	int label = -1;
	for (int i = 0; i < Samples.cols; i++)
	{
		double d = dis(Samples.col(i), y);
		if (d < mindis) 
		{
			mindis = d;
			label = i;
		}
	}

	char Slabel[10];
	itoa(label,Slabel,10);
	string path = "face\\BioID_";
	if (label < 10) path = path + "000";
	else if (label < 100) path = path +"00";
	else if (label < 1000) path = path + "0";
	path = path + Slabel;
	//�����Ƶ�����
	Mat image2 = imread(exepath+path+".pgm");
	imshow("the most similar image", image2);

	CvFont font; 
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, 1.0f, 1.0f, 0, 1, 8);
	Point p = cvPoint(Image->width/20, Image->height-10);
	//������Ļ
	path.erase(0, 5);
	cvPutText(Image, path.c_str(), p, &font,CV_RGB(238,221,130) );	

	cvShowImage("source image with test result", Image);
	waitKey(0);
    return 0;
}