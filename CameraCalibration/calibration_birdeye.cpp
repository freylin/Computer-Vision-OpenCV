#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
//ȫ�ֱ���
int n_boards = 0;
//���̳ߴ�
int board_w;
int board_h;
int board_n;
CvSize board_sz;
//����·��
string exepath;
//����ڲκͻ���ϵ��
CvMat* intrinsic  = cvCreateMat(3,3,CV_32FC1);
CvMat* distortion = cvCreateMat(4,1,CV_32FC1);

//����
void undistortion(IplImage* image, IplImage* mapx, IplImage* mapy)
{
	IplImage *t = cvCloneImage(image);
	cvRemap( t, image, mapx, mapy );
	cvReleaseImage(&t);
}

//���������ؽǵ�����
void CornerSubPix(IplImage* image, CvPoint2D32f* corners, int* corner_count)
{
	//�����������ؾ���ĻҶ�ͼ
	IplImage* gray_image = cvCreateImage(cvGetSize(image),8,1);
	cvCvtColor(image, gray_image, CV_BGR2GRAY);
	//��ȡ�����ؾ��Ƚǵ�����
	cvFindCornerSubPix(gray_image, corners, *corner_count, 
		cvSize(board_w,board_h),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
}

//�궨
void calibration(string txtpath)
{
	FILE *fptr = fopen(txtpath.c_str(),"r");
	char names[2048];
	//ͼƬ����
	while(fscanf(fptr,"%s ",names)==1){
		n_boards++;//ͳ���ۼ�
	}
	rewind(fptr);

	cvNamedWindow( "Calibration" );
	//�ڴ���䣬�ֱ�洢����������ꡢ��������ͽǵ�����
	CvMat* image_points      = cvCreateMat(n_boards*board_n,2,CV_32FC1);
	CvMat* object_points     = cvCreateMat(n_boards*board_n,3,CV_32FC1);
	CvMat* point_counts      = cvCreateMat(n_boards,1,CV_32SC1);

	IplImage* image = 0;
	//�ǵ�����
	CvPoint2D32f* corners = new CvPoint2D32f[ board_n ];
	int corner_count;
	int successes = 0;
	int step;

	for( int frame=0; frame<n_boards; frame++ ) {
		fscanf(fptr,"%s ",names);

		if(image){
			cvReleaseImage(&image);
			image = 0;
		}
		//��ȡͼƬ·��
		string imgpath = exepath+names;
		//��ȡͼƬ
		image = cvLoadImage(imgpath.c_str());
			
		//����ǵ����������
		int found = cvFindChessboardCorners(
			image,
			board_sz,
			corners,
			&corner_count, 
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
		);
		//���������ؽǵ�����
		CornerSubPix(image, corners, &corner_count);
	  
		//�����ǵ�
		cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
		cvShowImage( "Calibration", image );

		//���ǵ�ȫ���ҵ������������
		if( corner_count == board_n ) {
			step = successes*board_n;
			//�洢�����������������
			for( int i=step, j=0; j<board_n; ++i,++j ) {
				CV_MAT_ELEM(*image_points, float,i,0) = corners[j].x;
				CV_MAT_ELEM(*image_points, float,i,1) = corners[j].y;
				CV_MAT_ELEM(*object_points,float,i,0) = j/board_w;
				CV_MAT_ELEM(*object_points,float,i,1) = j%board_w;
				CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
			}
			//�洢�ǵ�����
			CV_MAT_ELEM(*point_counts, int,successes,0) = board_n;		
			successes++;
		}

		int c = cvWaitKey(20);
		//��ͣ��
		if(c == 'p') {
			c = 0;
			while(c != 'p' && c != 27){
				c = cvWaitKey(250);
			}
		}
		//�˳���
		if(c == 27) return;
	}
	printf("successes = %d, n_boards=%d\n",successes,n_boards);
	//����õ�������ֵ��������Ӧ��С�ľ�����
	CvMat* object_points2     = cvCreateMat(successes*board_n,3,CV_32FC1);
	CvMat* image_points2      = cvCreateMat(successes*board_n,2,CV_32FC1);
	CvMat* point_counts2      = cvCreateMat(successes,1,CV_32SC1);
	for(int i = 0; i<successes*board_n; ++i){
		CV_MAT_ELEM(*image_points2, float,i,0) 	=	CV_MAT_ELEM(*image_points, float,i,0);
		CV_MAT_ELEM(*image_points2, float,i,1) 	= 	CV_MAT_ELEM(*image_points, float,i,1);
		CV_MAT_ELEM(*object_points2,float,i,0) = CV_MAT_ELEM(*object_points,float,i,0) ;
		CV_MAT_ELEM(*object_points2,float,i,1) = CV_MAT_ELEM(*object_points,float,i,1) ;
		CV_MAT_ELEM(*object_points2,float,i,2) = CV_MAT_ELEM(*object_points,float,i,2) ;
	} 
	for(int i=0; i<successes; ++i){
 		CV_MAT_ELEM(*point_counts2,int,i, 0) = CV_MAT_ELEM(*point_counts, int,i,0);
	}
	//�ͷ�ԭ�����ڴ�
	cvReleaseMat(&object_points);
	cvReleaseMat(&image_points);
	cvReleaseMat(&point_counts);

    //��ʼ���ڲξ���ʹ�ý���fx��fy�ĵı�ֵΪ1
    CV_MAT_ELEM( *intrinsic, float, 0, 0 ) = 1.0f;
    CV_MAT_ELEM( *intrinsic, float, 1, 1 ) = 1.0f;
	printf("cvCalibrateCamera2\n");
	//���б궨����ȡ�ڲ�������ͻ���ϵ������
    cvCalibrateCamera2(
		object_points2,
		image_points2,
		point_counts2,
		cvGetSize( image ),
		intrinsic,
		distortion,
		NULL,
		NULL,
		0
    );
	//�����ڲξ���ͻ������
	string xml1path =  exepath + "Intrinsics.xml";
	string xml2path =  exepath + "Distortion.xml";
	cvSave(xml1path.c_str(),intrinsic);
	cvSave(xml2path.c_str(),distortion);

	//����ӳ����󣬱�ʾ�㱻ӳ���xֵ��yֵ
	IplImage* mapx = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
	IplImage* mapy = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
	printf("cvInitUndistortMap\n");
	cvInitUndistortMap(
		intrinsic,
		distortion,
		mapx,
		mapy
	);
	rewind(fptr);
	cvNamedWindow( "Undistort" );
	printf("Press any key to step through the images, ESC to quit\n");
	//��ʾԭͼ��������ͼƬ���жԱ�
	while(fscanf(fptr,"%s ",names)==1){
		if(image){
			cvReleaseImage(&image);
			image = 0;
		}  
		string imgpath = exepath+names;
		image = cvLoadImage(imgpath.c_str());
		//��ʾԭͼ
		cvShowImage( "Calibration", image );
		//����ͼƬ
		undistortion(image, mapx, mapy);
		cvShowImage("Undistort", image);
		//esc�˳�
		if((cvWaitKey()&0x7F) == 27) break;  
	}
}

int main(int argc, char* argv[]) {
	//����ı���
	board_w = atoi(argv[1]);
	board_h = atoi(argv[2]);
	board_n  = board_w * board_h;
	board_sz = cvSize( board_w, board_h );
	string txtpath = argv[3];
	exepath = argv[3];
	exepath.erase(exepath.find("chessboards.txt"), 15);
	//��������궨��ͼ�����
	calibration(txtpath);

	//����ͼƬ
	IplImage *image = cvLoadImage(argv[4]);

	//���ݱ궨ʱ��ȡ���ڲ�������ͻ���ϵ���������ͼƬ
    IplImage* mapx = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
    IplImage* mapy = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
    cvInitUndistortMap(
      intrinsic,
      distortion,
      mapx,
      mapy
    );
	//����ͼƬ
	undistortion(image, mapx, mapy);

	//��ȡƽ���ϵ����̸�ǵ�
	cvNamedWindow("Checkers");
    CvPoint2D32f* corners = new CvPoint2D32f[ board_n ];
    int corner_count = 0;
    int found = cvFindChessboardCorners(
        image,
        board_sz,
        corners,
        &corner_count, 
        CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
    );
	//���ҵ�ȫ���ǵ�ſɽ���ͶӰ�Ȳ������������������Ϣ
	if(!found){
		printf("Couldn't aquire checkerboard on %s, only found %d of %d corners\n",
				argv[4],corner_count,board_n);
	}
	
	//���������ؽǵ�����
	CornerSubPix(image, corners, &corner_count);

	//��ȡ�����������������
	//�ĸ��������������λ��: (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
	//���������Ӧ����������: corners[r*board_w + c]
	CvPoint2D32f objPts[4], imgPts[4];
	objPts[0].x = 0;         objPts[0].y = 0; 
	objPts[1].x = board_w-1; objPts[1].y = 0; 
	objPts[2].x = 0;         objPts[2].y = board_h-1;
	objPts[3].x = board_w-1; objPts[3].y = board_h-1; 
	imgPts[0] = corners[0];
	imgPts[1] = corners[board_w-1];
	imgPts[2] = corners[(board_h-1)*board_w];
	imgPts[3] = corners[(board_h-1)*board_w + board_w-1];

	//���ĸ����������̺�Ƶ�˳����ԲȦ��ǳ���
	cvCircle(image,cvPointFrom32f(imgPts[0]),9,CV_RGB(0,0,255),3);
	cvCircle(image,cvPointFrom32f(imgPts[1]),9,CV_RGB(0,255,0),3);
	cvCircle(image,cvPointFrom32f(imgPts[2]),9,CV_RGB(255,0,0),3);
	cvCircle(image,cvPointFrom32f(imgPts[3]),9,CV_RGB(255,255,0),3);

	//��ǽǵ㲢��ʾͼ��
	cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
    cvShowImage( "Checkers", image );

	//��ȡ��Ӧ�Ծ���H
	CvMat *H = cvCreateMat( 3, 3, CV_32F);
	cvGetPerspectiveTransform(objPts,imgPts,H);

	//���û������ӽǵĸ߶�
	float Z = 25;//��ʼ�߶�Ϊ25
	int key = 0;
	IplImage *birds_image = cvCloneImage(image);
	cvNamedWindow("Birds_Eye");
	//esc�˳�
    while(key != 27) {
	   CV_MAT_ELEM(*H,float,2,2) = Z;
	   //ʹ�õ�Ӧ�Ծ���ͶӰͼƬ
	   cvWarpPerspective(image,birds_image,H,
			CV_INTER_LINEAR+CV_WARP_INVERSE_MAP+CV_WARP_FILL_OUTLIERS );
	   //��ʾ��ȡ�����ͼ
	   cvShowImage("Birds_Eye", birds_image);
	   key = cvWaitKey();
	   //�û���ͨ��u��d�����и߶ȵĵ���
	   if(key == 'u') Z += 0.5;
	   if(key == 'd') Z -= 0.5;
	}
	//���浥Ӧ�Ծ���
	cvSave("H.xml",H);
	return 0;
}