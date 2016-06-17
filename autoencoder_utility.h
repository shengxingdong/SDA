#ifndef _auto_encoder_utility_h_
#define _auto_encoder_utility_h_

#include "cv_header.h"


Mat concatenateMat(vector<Mat> &vec);
void read_Mnist(string filename, vector<Mat> &vec);
void read_Mnist_Label(string filename, Mat &mat);


void readMnistData(Mat &x, Mat &y, string xpath, string ypath, int number_of_images);

#endif