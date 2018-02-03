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

Mat cut(string path, string l)
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
	Point2f center = cv::Point2f(x3, y3);  // ��ת����Ϊ��ߵ��۾����˵����ۣ� 
	double angle = atan((double)(y4-y3)/(double)(x4-x3))/3.1415926*180.0;  // ��ʱ����ת�Ƕ�
	x4 = x3 + sqrt((y4-y3)*(y4-y3)+(x4-x3)*(x4-x3));
	Mat rotateMat; 
	//��ת����
	rotateMat = getRotationMatrix2D(center, angle, 1.0);  
	//ӳ�䣬�����ת
	warpAffine(m, m, rotateMat, m.size());  

	//�趨������ȡģ��
	int w = 120, h = 150 , w2 = m.cols, h2 = m.rows;
	double d = (60.0/(1.0*(x4-x3)));
	resize(m,m,Size(w2 * d, h2 * d));
	Size size = Size(w, h);
	int x5 = x3*d-(w-60)/2, y5 = y3*d - h/3;
	//���н�ȡ
	Mat m2 = Mat::zeros( size, 0 );
	for (int i  = y5; i < y5 + h; i++)
		if (i < h2 && i >= 0)
			for (int j = x5; j < x5 + w; j++)
			if (j < w2 && j >= 0)
				{
					m2.at<uchar>(i - y5 ,j - x5) = m.at<uchar>(i,j);
				} 
	Mat m3 = Mat::zeros( Size(w*2, h*2), 0 );
	//����
	resize(m2,m3, Size(w*2, h*2));
	//��һ����������ͼ��
	return norm_0_255(m3);
}

//ʹ��CSV�ļ�ȥ��ͼ��ͱ�ǩ
static void read_csv(const string& filename, string exepath, vector<Mat>& images, vector<int>& labels, char separator =',') {
	std::ifstream file(filename.c_str(), ifstream::in);
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty()&&!classlabel.empty()) {
			path = exepath+path;
            // images.push_back(imread(path, 0));
			images.push_back( cut(path, classlabel) );
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

// ��һ��ͼ�������ת��ΪRow Matrix�е�һ��
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    size_t n = src.size();
    size_t d = src[0].total();
    // �������ؾ���
    Mat data(n, d, rtype);
    // ��ͼ�����ݸ��Ƶ����������
    for(int i = 0; i < n; i++) {
        // ��÷��ؾ����еĵ�ǰ�о���:
        Mat xi = data.row(i);
        // ��һ��ͼ��ӳ�䵽���ؾ����һ����:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

Mat get_mean(Mat src)
{
	Mat mean(1,src.cols, CV_32FC1 );
	for (int i = 0; i < src.rows; i++)
		mean = mean + src.row(i);
	mean = mean / src.rows;
	return mean;
}

Mat get_S(Mat src, Mat mean, Mat* AT)
{
	Mat x(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++)
	{
		x.row(i) = src.row(i) - mean;
	}
	*AT = x.t();
	return x*x.t()/src.rows;
}

int main(int argc, const char*argv[]) {
	string exepath;
	exepath = argv[0];
	float Kper = atof(argv[1]);

	exepath.erase(exepath.find("mytrain.exe"), 13);
    //��ȡCSV�ļ�·��.
    string fn_csv = exepath+"face.csv";

    // 2�����������ͼ�����ݺͶ�Ӧ�ı�ǩ
    vector<Mat> images;
    vector<int> labels;
    // ��ȡ����
    read_csv(fn_csv, exepath, images, labels);
    // ��ѵ�����ݶ��뵽���ݼ�����
    Mat data = asRowMatrix(images, CV_32FC1);

    // ƽ��ֵ
    Mat mean = get_mean(data);

	// Э�������
	Mat S, eigenvalues, eigenvectors, AT;
	S = get_S(data, mean, &AT);
	eigen(S, eigenvalues, eigenvectors);

	//���������� MN*k
	eigenvectors = AT * (eigenvectors.t());
	normalize(eigenvectors, eigenvectors, 0,255, NORM_MINMAX, CV_32FC1);
	Mat e10(eigenvectors.rows, 1, CV_32FC1);
	for (int i = 0; i < 10; i++)
	{	
		e10 = e10 + eigenvectors.col(i);
	}
	// ǰʮ��������
	imshow("eigen10",norm_0_255(e10.reshape(1, images[0].rows)));

    // ƽ����:
    imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

	//���������ٷֱ�ѡȡ������������
	int k = 0;
	double total_e = 0.0, ce = 0.0;
	for (int i = 0 ; i < eigenvalues.rows; i++)
		total_e += eigenvalues.at<float>(i, 0);
	while (ce/total_e < Kper)
	{
		ce += eigenvalues.at<float>(k, 0);
		k++;
	}
	//ѡȡ��ǰk���������� MN*k
	Mat eigenk(eigenvectors.rows, k, CV_32FC1);
	for (int i = 0; i < k; i++)
		eigenk.col(i) = eigenvectors.col(i) * 1.0;
    //����ѵ�����
    FileStorage fs(argv[2], FileStorage::WRITE);
	fs<<"eigenvectors"<<eigenk;
    fs<<"Samples"<<eigenk.t() * data.t();
	fs<<"mean"<<norm_0_255(mean.reshape(1, images[0].rows));
    fs.release();

	waitKey(0);
    return 0;
}