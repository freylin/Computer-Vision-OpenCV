#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "cv.h"

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, dst_norm;
int thresh = 87;
int max_thresh = 255;

char* source_window = "SourceImage";
char* corner_response_window = "CornerResponse";
char* LambdaMin_window = "LambdaMin";
char* LambdaMax_window = "LambdaMax";
char* Final_window = "FinalImage";
string path;
char* pic;

/// Function header
void HarrisDetector(double k, int Aperture_size);
void Show(char * name, Mat m);
void CornerResponse( int, void* );

int main( int argc, char** argv )
{
	/// Load source image and convert it to gray
	pic = argv[1];
	src = imread( pic, 1 );
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	//图片存储路径
	path = argv[1];
	for (int i = path.length()-1; i >= 0; i--)
	if (argv[1][i] == '\\')
	{
		path.erase(i+1, path.length()-1-i);
		break;
	}
	//显示原图
	Show( source_window, src );

	double k = 0.04;
	int Aperture_size = 3;
	if (argc > 2) k = atof(argv[2]);
	if (argc == 4) Aperture_size = atoi(argv[3]);
	HarrisDetector(k, Aperture_size);

	waitKey(0);
	return(0);
}

//初始化矩阵
void Init(Mat* m)
{
	*m = Mat::zeros( src.size(), CV_32F );
}

//判断局部最大值
bool LocalMax(int x, int y, Mat m)
{
	//区域尺寸
	int size = 5;
	for (int i = -size; i <= size; i++)
	{
		int xx = x + i;
		if (xx >= 0 && xx < m.rows)
			for (int j = -size; j <= size; j++)
			{
				int yy = y + j;
				if (yy >= 0 && yy < m.cols)
					if (m.at<float>(x,y) < m.at<float>(xx,yy))
						return false;
			}
	}
	return true;
}

void Norm(Mat *m1, Mat *m2, Mat *m3)
{
	//归化到0~255
	normalize(*m1, *m2, 0, 255, NORM_MINMAX, CV_32F, Mat() );
	//转换成8位无符号整型 
	convertScaleAbs(*m2, *m3);
}

void Show(char * name, Mat m)
{
	//创建窗口
	namedWindow( name, WINDOW_AUTOSIZE );
	//显示图片
	imshow( name, m );
	//保存图片
	imwrite(path+name+".jpg", m );
}

void CornerResponse( int, void* )
{
	Mat src2 = imread( pic, 1 );
	
	//在值较大的角点处做标记
	for( int i = 0; i < dst_norm.rows ; i++ )
      for( int j = 0; j < dst_norm.cols; j++ )
		if( dst_norm.at<float>(i,j) + 0.5 > thresh )
		{
			//判断R是否是局部的极大值,若是，则在原图上标记
			if (LocalMax(i,j,dst_norm)) circle(src2, Point(j, i), 2, cvScalar(0, 0, 255), -1, 8, 0 );
		}
	//显示图片
	imshow( Final_window, src2 );
	//保存图片
	imwrite(path+Final_window+".jpg", src2 );
}

void HarrisDetector(double k, int Aperture_size)
{
	/// Detector parameters
	int blockSize = 3;
	int depth = src_gray.depth();
    double scale = (double)(1 << (Aperture_size - 1)) * blockSize;
    if( depth == CV_8U )
        scale *= 255.;
    scale = 1./scale;

	Mat lx, ly;
	//用Sobel算子求一阶差分
    Sobel( src_gray, lx, CV_32F, 1, 0, Aperture_size, scale, 0, BORDER_DEFAULT );
    Sobel( src_gray, ly, CV_32F, 0, 1, Aperture_size, scale, 0, BORDER_DEFAULT );

	Mat l[3], lam1, lam2;
	Mat dst, dst_norm_scaled;
	for (int i = 0; i < 3; i++) Init(&l[i]);
	Init(&lam1);
	Init(&lam2);
	Init(&dst);

	//得到矩阵元素
	for (int i = 0; i < lx.rows; i++)
		for (int j = 0; j < lx.cols; j++)
		{
			l[0].at<float>(i,j) = lx.at<float>(i,j) * lx.at<float>(i,j);
			l[1].at<float>(i,j) = ly.at<float>(i,j) * ly.at<float>(i,j);
			l[2].at<float>(i,j) = lx.at<float>(i,j) * ly.at<float>(i,j);
		}
	
	//对图形矩阵分别进行高斯滤波
	for (int i = 0; i < 3; i++)
		GaussianBlur(l[i], l[i], Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
	
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{
			//求解特征值所用的delta值
			float del = sqrt(
				pow( l[0].at<float>(i,j) + l[1].at<float>(i,j), 2) -
				4.0 * (l[0].at<float>(i,j) * l[1].at<float>(i,j) - 
					pow(l[2].at<float>(i,j), 2) )
				);
			//求解两个特征值lambda
			lam1.at<float>(i,j) = 0.5 * ( l[0].at<float>(i,j) + l[1].at<float>(i,j) + del );
			lam2.at<float>(i,j) = 0.5 * ( l[0].at<float>(i,j) + l[1].at<float>(i,j) - del );
			//利用特征值求解R值
			dst.at<float>(i,j) = (lam1.at<float>(i,j) * lam2.at<float>(i,j)) 
				- k * pow(lam1.at<float>(i,j) + lam2.at<float>(i,j), 2);
		}
	
	//Normalizing，将矩阵的值归化到0~255，并转换成整型
	Norm(&lam1, &lam1, &lam1);
	Norm(&lam2, &lam2, &lam2);
	Norm(&dst, &dst_norm, &dst_norm_scaled);

	/// Showing the results
	Show( LambdaMax_window, lam1 );
	Show( LambdaMin_window, lam2 );
	Show( corner_response_window, dst_norm_scaled );

	//创建交互进度条，用于调整阈值
	namedWindow( Final_window, WINDOW_AUTOSIZE );
	createTrackbar( "Threshold: ", Final_window, &thresh, max_thresh, CornerResponse );
	CornerResponse(0, 0);

}
