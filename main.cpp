/*
 * SdA.cpp (Stacked Denoising Autoencoders)
 *
 * @author  yusugomori (http://yusugomori.com)
 * @usage   $ g++ SdA.cpp
 *
 */

#include "autoencoder_utility.h"
//#include "SdA.h"
#include "SdA.cpp"

#include <iostream>
#include <math.h>
using namespace std;

void test_sda() {
	srand(0);

	double pretrain_lr = 0.1;
	double corruption_level = 0.3;
	int pretraining_epochs = 1000;
	double finetune_lr = 0.1;
	int finetune_epochs = 500;

	int train_N = 10;
	int test_N = 4;
	int n_ins = 28;
	int n_outs = 2;
	int hidden_layer_sizes[] = { 15, 15 };
	int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

	// training data
	int train_X[10][28] = {
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1 }
	};

	int train_Y[10][2] = {
		{ 1, 0 },
		{ 1, 0 },
		{ 1, 0 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 0, 1 },
		{ 0, 1 },
		{ 0, 1 }
	};

#if 1
	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 28; j++){
			train_X[i][j] *= 255;
		}
	}

#endif

	// construct SdA
	SdA<int> sda(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

	// pretrain
	sda.pretrain(*train_X, pretrain_lr, corruption_level, pretraining_epochs);

	// finetune
	sda.finetune(*train_X, *train_Y, finetune_lr, finetune_epochs);


	// test data
	int test_X[4][28] = {
		{ 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 }
	};

#if 1
	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 28; j++){
			test_X[i][j] *= 255;
		}
	}
#endif

	double test_Y[4][28];

	// test
	for (int i = 0; i < test_N; i++) {
		sda.predict(test_X[i], test_Y[i]);
		for (int j = 0; j < n_outs; j++) {
			printf("%.5f ", test_Y[i][j]);
		}
		cout << endl;
	}

}


void test_sda_mnist()
{

	//loading data
	//read train and label data
	Mat trainX, trainY;
	readMnistData(trainX, trainY, "../mnist/train-images-idx3-ubyte", "../mnist/train-labels-idx1-ubyte", 60000);
	cout << "Read trainX successfully, including " << trainX.cols << " features and " << trainX.rows << " samples." << endl;
	cout << "Read trainY successfully, including " << trainY.cols << " labels and " << trainY.rows << " samples." << endl;

	// pre-processing data. 
	//Scalar mean, stddev;
	//meanStdDev(trainX, mean, stddev);
	//Mat normX = trainX - mean[0];
	//normX.copyTo(trainX);

	Mat testX, testY;
	readMnistData(testX, testY, "../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte", 10000);
	cout << "Read testX successfully, including " << testX.cols << " features and " << testX.rows << " samples." << endl;
	cout << "Read testY successfully, including " << testY.cols << " labels and " << testY.rows << " samples." << endl;

	Mat trainXInt, trainYInt;
	trainX.convertTo(trainXInt, CV_32FC1);
	trainY.convertTo(trainYInt, CV_32SC1);

	Mat testXInt, testYInt;
	testX.convertTo(testXInt, CV_32FC1);
	testY.convertTo(testYInt, CV_32SC1);

	//for (int i = 0; i < trainX.cols; i++){
	//	cout << trainXInt.at<int>(0, i) << ", "; 
	//	if ((i + 1) % 16 == 0)cout << endl;
	//}
	//cout << endl;
	//for (int i = 0; i < trainY.cols; i++){
	//	cout << trainYInt.at<int>(0, i) << ", ";
	//}
	//return;

	float *train_X = (float*)trainXInt.data;
	int *train_Y = (int*)trainYInt.data;
	//
	srand(0);
	double pretrain_lr = 0.1;
	double corruption_level = 0.3;
	int pretraining_epochs = 1000;
	double finetune_lr = 0.1;
	int finetune_epochs = 1000;

	int train_N = trainX.rows;
	int test_N = testY.rows;
	int n_ins = trainX.cols;
	int n_outs = 10;
	int hidden_layer_sizes[] = { 100, 50 };
	int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

	// construct SdA
	SdA<float> sda(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

	// pretrain
	cout << "pretrainning... \n";
	sda.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs);

	// finetune
	cout << "finetuning... \n";
	sda.finetune(train_X, train_Y, finetune_lr, finetune_epochs);

	// test
	double *predict_y = new double[n_outs];
	int correct_cnt = 0;
	for (int i = 0; i < test_N; i++) {
		float *test_X = ((float*)testXInt.data) + i * testXInt.cols;
		
		sda.predict(test_X, predict_y);
		int max_j = 0;
		double max_val = 0;
		for (int j = 0; j < n_outs; j++) {
			if (predict_y[j] > max_val){
				max_val = predict_y[j];
				max_j = j;
			}

			//if (i < 10)printf("%.5f ", predict_y[j]);
		}
		
		int gt_j = 0;
		for (int j = 0; j < n_outs; j++) {
			if (testYInt.at<int>(i, j) == 1)gt_j = j;

			//if (i < 10)printf("%d ", testYInt.at<int>(i, j));
		}

		if (gt_j == max_j)correct_cnt++;
	}

	cout << "\nprecision: " << 100.0 * correct_cnt / test_N << endl;

	delete[]predict_y;
}

int main() {
	//test_sda();
	test_sda_mnist();
	return 0;
}
