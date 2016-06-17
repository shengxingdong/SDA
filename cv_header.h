#ifndef __cv_header__H__
#define __cv_header__H__

#include <opencv2/opencv.hpp>

#ifdef WIN32

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef WIN64
#define OPENCV_LIB_DIR "D:/software/OpenCV/OpenCV310/install/x64/vc12/lib/"
#else
#define OPENCV_LIB_DIR "D:/software/OpenCV/OpenCV310/install/x86/vc12/lib/"
#endif

#ifdef _DEBUG
#define cvLIB(name) OPENCV_LIB_DIR "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) OPENCV_LIB_DIR "opencv_" name CV_VERSION_ID
#endif

#if 0
#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgcodecs") )
#pragma comment( lib, cvLIB("imgproc") ) 
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("flann") )
#pragma comment( lib, cvLIB("features2d") )
#pragma comment( lib, cvLIB("calib3d") )
#pragma comment( lib, cvLIB("ml") )
#pragma comment( lib, cvLIB("objdetect") )
#pragma comment( lib, cvLIB("ts") )
#pragma comment( lib, cvLIB("video") )
#pragma comment( lib, cvLIB("dnn") )
#else
#pragma comment( lib, cvLIB("world") )
//#pragma comment( lib, cvLIB("ts") )
#endif//lib

#include <tchar.h>

#endif//win32

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;
using namespace cv;

#endif
