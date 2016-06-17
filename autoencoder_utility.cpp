#include "autoencoder_utility.h"

#define ATD at<double>

#define RZW 28
#define RZH 28
#define SAMPLE_DIV 1

Mat concatenateMat(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
	Mat res = Mat::zeros(vec.size() / SAMPLE_DIV, RZW * RZH, CV_64FC1);
	for (int i = 0; i<vec.size(); i += SAMPLE_DIV){
        Mat img;
        vec[i].convertTo(img, CV_64FC1);
		resize(img, img, Size(RZW, RZH), 0, 0, 0);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, 1);
		Rect roi = cv::Rect(0, i / SAMPLE_DIV, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    //divide(res, 255.0, res);
    return res;
}

int ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<Mat> &vec){
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i){
            Mat tpmat = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tpmat.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tpmat);
        }
    }
}

void read_Mnist_Label(string filename, Mat &mat)
{
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            mat.ATD(0, i) = (double)temp;
        }
    }
}


void readMnistData(Mat &x, Mat &y, string xpath, string ypath, int number_of_images)
{

    //read MNIST iamge into OpenCV Mat vector
	int image_size = RZW * RZH;
    vector<Mat> vec;
    //vec.resize(number_of_images, cv::Mat(28, 28, CV_8UC1));
    read_Mnist(xpath, vec);
    //read MNIST label into double vector
    Mat label = Mat::zeros(1, number_of_images, CV_64FC1);
	read_Mnist_Label(ypath, label);
    x = concatenateMat(vec);


	y = Mat::zeros(number_of_images / SAMPLE_DIV, 10, CV_64FC1);
	for (int i = 0; i < number_of_images; i += SAMPLE_DIV){
		int val = label.at<double>(0, i);
		y.at<double>(i / SAMPLE_DIV, val) = 1;
	}

#if 1
	for (int r = 0; r < x.rows; r++){
		for (int c = 0; c < x.cols; c++){
			if (x.at<double>(r, c) > 0.1){
				x.at<double>(r, c) = 1;
			}
		}
	}
#endif
}