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
	//ͼƬ�洢·��
	path = argv[1];
	for (int i = path.length()-1; i >= 0; i--)
	if (argv[1][i] == '\\')
	{
		path.erase(i+1, path.length()-1-i);
		break;
	}
	//��ʾԭͼ
	Show( source_window, src );

	double k = 0.04;
	int Aperture_size = 3;
	if (argc > 2) k = atof(argv[2]);
	if (argc == 4) Aperture_size = atoi(argv[3]);
	HarrisDetector(k, Aperture_size);

	waitKey(0);
	return(0);
}

//��ʼ������
void Init(Mat* m)
{
	*m = Mat::zeros( src.size(), CV_32F );
}

//�жϾֲ����ֵ
bool LocalMax(int x, int y, Mat m)
{
	//����ߴ�
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
	//�黯��0~255
	normalize(*m1, *m2, 0, 255, NORM_MINMAX, CV_32F, Mat() );
	//ת����8λ�޷������� 
	convertScaleAbs(*m2, *m3);
}

void Show(char * name, Mat m)
{
	//��������
	namedWindow( name, WINDOW_AUTOSIZE );
	//��ʾͼƬ
	imshow( name, m );
	//����ͼƬ
	imwrite(path+name+".jpg", m );
}

void CornerResponse( int, void* )
{
	Mat src2 = imread( pic, 1 );
	
	//��ֵ�ϴ�Ľǵ㴦�����
	for( int i = 0; i < dst_norm.rows ; i++ )
      for( int j = 0; j < dst_norm.cols; j++ )
		if( dst_norm.at<float>(i,j) + 0.5 > thresh )
		{
			//�ж�R�Ƿ��Ǿֲ��ļ���ֵ,���ǣ�����ԭͼ�ϱ��
			if (LocalMax(i,j,dst_norm)) circle(src2, Point(j, i), 2, cvScalar(0, 0, 255), -1, 8, 0 );
		}
	//��ʾͼƬ
	imshow( Final_window, src2 );
	//����ͼƬ
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
	//��Sobel������һ�ײ��
    Sobel( src_gray, lx, CV_32F, 1, 0, Aperture_size, scale, 0, BORDER_DEFAULT );
    Sobel( src_gray, ly, CV_32F, 0, 1, Aperture_size, scale, 0, BORDER_DEFAULT );

	Mat l[3], lam1, lam2;
	Mat dst, dst_norm_scaled;
	for (int i = 0; i < 3; i++) Init(&l[i]);
	Init(&lam1);
	Init(&lam2);
	Init(&dst);

	//�õ�����Ԫ��
	for (int i = 0; i < lx.rows; i++)
		for (int j = 0; j < lx.cols; j++)
		{
			l[0].at<float>(i,j) = lx.at<float>(i,j) * lx.at<float>(i,j);
			l[1].at<float>(i,j) = ly.at<float>(i,j) * ly.at<float>(i,j);
			l[2].at<float>(i,j) = lx.at<float>(i,j) * ly.at<float>(i,j);
		}
	
	//��ͼ�ξ���ֱ���и�˹�˲�
	for (int i = 0; i < 3; i++)
		GaussianBlur(l[i], l[i], Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
	
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{
			//�������ֵ���õ�deltaֵ
			float del = sqrt(
				pow( l[0].at<float>(i,j) + l[1].at<float>(i,j), 2) -
				4.0 * (l[0].at<float>(i,j) * l[1].at<float>(i,j) - 
					pow(l[2].at<float>(i,j), 2) )
				);
			//�����������ֵlambda
			lam1.at<float>(i,j) = 0.5 * ( l[0].at<float>(i,j) + l[1].at<float>(i,j) + del );
			lam2.at<float>(i,j) = 0.5 * ( l[0].at<float>(i,j) + l[1].at<float>(i,j) - del );
			//��������ֵ���Rֵ
			dst.at<float>(i,j) = (lam1.at<float>(i,j) * lam2.at<float>(i,j)) 
				- k * pow(lam1.at<float>(i,j) + lam2.at<float>(i,j), 2);
		}
	
	//Normalizing���������ֵ�黯��0~255����ת��������
	Norm(&lam1, &lam1, &lam1);
	Norm(&lam2, &lam2, &lam2);
	Norm(&dst, &dst_norm, &dst_norm_scaled);

	/// Showing the results
	Show( LambdaMax_window, lam1 );
	Show( LambdaMin_window, lam2 );
	Show( corner_response_window, dst_norm_scaled );

	//�������������������ڵ�����ֵ
	namedWindow( Final_window, WINDOW_AUTOSIZE );
	createTrackbar( "Threshold: ", Final_window, &thresh, max_thresh, CornerResponse );
	CornerResponse(0, 0);

}