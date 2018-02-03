#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
//全局变量
int n_boards = 0;
//棋盘尺寸
int board_w;
int board_h;
int board_n;
CvSize board_sz;
//运行路径
string exepath;
//相机内参和畸变系数
CvMat* intrinsic  = cvCreateMat(3,3,CV_32FC1);
CvMat* distortion = cvCreateMat(4,1,CV_32FC1);

//矫正
void undistortion(IplImage* image, IplImage* mapx, IplImage* mapy)
{
	IplImage *t = cvCloneImage(image);
	cvRemap( t, image, mapx, mapy );
	cvReleaseImage(&t);
}

//计算亚像素角点坐标
void CornerSubPix(IplImage* image, CvPoint2D32f* corners, int* corner_count)
{
	//用于求亚像素矩阵的灰度图
	IplImage* gray_image = cvCreateImage(cvGetSize(image),8,1);
	cvCvtColor(image, gray_image, CV_BGR2GRAY);
	//获取亚像素精度角点坐标
	cvFindCornerSubPix(gray_image, corners, *corner_count, 
		cvSize(board_w,board_h),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
}

//标定
void calibration(string txtpath)
{
	FILE *fptr = fopen(txtpath.c_str(),"r");
	char names[2048];
	//图片数量
	while(fscanf(fptr,"%s ",names)==1){
		n_boards++;//统计累加
	}
	rewind(fptr);

	cvNamedWindow( "Calibration" );
	//内存分配，分别存储点的像素坐标、物理坐标和角点数量
	CvMat* image_points      = cvCreateMat(n_boards*board_n,2,CV_32FC1);
	CvMat* object_points     = cvCreateMat(n_boards*board_n,3,CV_32FC1);
	CvMat* point_counts      = cvCreateMat(n_boards,1,CV_32SC1);

	IplImage* image = 0;
	//角点坐标
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
		//获取图片路径
		string imgpath = exepath+names;
		//读取图片
		image = cvLoadImage(imgpath.c_str());
			
		//计算角点坐标和数量
		int found = cvFindChessboardCorners(
			image,
			board_sz,
			corners,
			&corner_count, 
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
		);
		//计算亚像素角点坐标
		CornerSubPix(image, corners, &corner_count);
	  
		//画出角点
		cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
		cvShowImage( "Calibration", image );

		//若角点全部找到，则加入数据
		if( corner_count == board_n ) {
			step = successes*board_n;
			//存储像素坐标和物理坐标
			for( int i=step, j=0; j<board_n; ++i,++j ) {
				CV_MAT_ELEM(*image_points, float,i,0) = corners[j].x;
				CV_MAT_ELEM(*image_points, float,i,1) = corners[j].y;
				CV_MAT_ELEM(*object_points,float,i,0) = j/board_w;
				CV_MAT_ELEM(*object_points,float,i,1) = j%board_w;
				CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
			}
			//存储角点数量
			CV_MAT_ELEM(*point_counts, int,successes,0) = board_n;		
			successes++;
		}

		int c = cvWaitKey(20);
		//暂停键
		if(c == 'p') {
			c = 0;
			while(c != 'p' && c != 27){
				c = cvWaitKey(250);
			}
		}
		//退出键
		if(c == 27) return;
	}
	printf("successes = %d, n_boards=%d\n",successes,n_boards);
	//将获得的坐标规和值等整到相应大小的矩阵中
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
	//释放原矩阵内存
	cvReleaseMat(&object_points);
	cvReleaseMat(&image_points);
	cvReleaseMat(&point_counts);

    //初始化内参矩阵，使得焦距fx和fy的的比值为1
    CV_MAT_ELEM( *intrinsic, float, 0, 0 ) = 1.0f;
    CV_MAT_ELEM( *intrinsic, float, 1, 1 ) = 1.0f;
	printf("cvCalibrateCamera2\n");
	//进行标定，获取内参数矩阵和畸变系数矩阵
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
	//保存内参矩阵和畸变矩阵
	string xml1path =  exepath + "Intrinsics.xml";
	string xml2path =  exepath + "Distortion.xml";
	cvSave(xml1path.c_str(),intrinsic);
	cvSave(xml2path.c_str(),distortion);

	//畸变映射矩阵，表示点被映射的x值和y值
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
	//显示原图与矫正后的图片进行对比
	while(fscanf(fptr,"%s ",names)==1){
		if(image){
			cvReleaseImage(&image);
			image = 0;
		}  
		string imgpath = exepath+names;
		image = cvLoadImage(imgpath.c_str());
		//显示原图
		cvShowImage( "Calibration", image );
		//矫正图片
		undistortion(image, mapx, mapy);
		cvShowImage("Undistort", image);
		//esc退出
		if((cvWaitKey()&0x7F) == 27) break;  
	}
}

int main(int argc, char* argv[]) {
	//输入的变量
	board_w = atoi(argv[1]);
	board_h = atoi(argv[2]);
	board_n  = board_w * board_h;
	board_sz = cvSize( board_w, board_h );
	string txtpath = argv[3];
	exepath = argv[3];
	exepath.erase(exepath.find("chessboards.txt"), 15);
	//进行相机标定和图像矫正
	calibration(txtpath);

	//待测图片
	IplImage *image = cvLoadImage(argv[4]);

	//根据标定时获取的内参数矩阵和畸变系数矩阵矫正图片
    IplImage* mapx = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
    IplImage* mapy = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
    cvInitUndistortMap(
      intrinsic,
      distortion,
      mapx,
      mapy
    );
	//矫正图片
	undistortion(image, mapx, mapy);

	//获取平面上的棋盘格角点
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
	//需找到全部角点才可进行投影等操作，否则输出错误信息
	if(!found){
		printf("Couldn't aquire checkerboard on %s, only found %d of %d corners\n",
				argv[4],corner_count,board_n);
	}
	
	//计算亚像素角点坐标
	CornerSubPix(image, corners, &corner_count);

	//获取像素坐标和物理坐标
	//四个顶点的物理坐标位于: (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
	//物理坐标对应的像素坐标: corners[r*board_w + c]
	CvPoint2D32f objPts[4], imgPts[4];
	objPts[0].x = 0;         objPts[0].y = 0; 
	objPts[1].x = board_w-1; objPts[1].y = 0; 
	objPts[2].x = 0;         objPts[2].y = board_h-1;
	objPts[3].x = board_w-1; objPts[3].y = board_h-1; 
	imgPts[0] = corners[0];
	imgPts[1] = corners[board_w-1];
	imgPts[2] = corners[(board_h-1)*board_w];
	imgPts[3] = corners[(board_h-1)*board_w + board_w-1];

	//将四个顶点以蓝绿红黄的顺序用圆圈标记出来
	cvCircle(image,cvPointFrom32f(imgPts[0]),9,CV_RGB(0,0,255),3);
	cvCircle(image,cvPointFrom32f(imgPts[1]),9,CV_RGB(0,255,0),3);
	cvCircle(image,cvPointFrom32f(imgPts[2]),9,CV_RGB(255,0,0),3);
	cvCircle(image,cvPointFrom32f(imgPts[3]),9,CV_RGB(255,255,0),3);

	//标记角点并显示图像
	cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
    cvShowImage( "Checkers", image );

	//获取单应性矩阵H
	CvMat *H = cvCreateMat( 3, 3, CV_32F);
	cvGetPerspectiveTransform(objPts,imgPts,H);

	//让用户调整视角的高度
	float Z = 25;//初始高度为25
	int key = 0;
	IplImage *birds_image = cvCloneImage(image);
	cvNamedWindow("Birds_Eye");
	//esc退出
    while(key != 27) {
	   CV_MAT_ELEM(*H,float,2,2) = Z;
	   //使用单应性矩阵投影图片
	   cvWarpPerspective(image,birds_image,H,
			CV_INTER_LINEAR+CV_WARP_INVERSE_MAP+CV_WARP_FILL_OUTLIERS );
	   //显示获取的鸟瞰图
	   cvShowImage("Birds_Eye", birds_image);
	   key = cvWaitKey();
	   //用户可通过u和d键进行高度的调整
	   if(key == 'u') Z += 0.5;
	   if(key == 'd') Z -= 0.5;
	}
	//保存单应性矩阵
	cvSave("H.xml",H);
	return 0;
}
