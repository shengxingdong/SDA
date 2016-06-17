#include "SdA.h"
#include <iostream>
#include <math.h>
using namespace std;

double uniform(double min, double max) {
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

template<typename T>
T binomial(T n, double p) {
	if (p < 0 || p > 1) return 0;

	double r;
#if 0
	int c = 0;
	for (int i = 0; i < n; i++) {
		r = rand() / (RAND_MAX + 1.0);
		if (r < p) c++;
	}
#else
	T c = n;
	r = rand() / (RAND_MAX + 1.0);
	if (r < p) {
		return n;
	}else{
		return 0;
	}
	
#endif

	return c;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}


// SdA
template<typename T>
SdA<T>::SdA(int size, int n_i, int *hls, int n_o, int n_l) {
	int input_size;
	N = size;
	n_ins = n_i;
	hidden_layer_sizes = hls;
	n_outs = n_o;
	n_layers = n_l;

	sigmoid_layers = new HiddenLayer<T>*[n_layers];
	dA_layers = new dA<T>*[n_layers];

	// construct multi-layer
	for (int i = 0; i < n_layers; i++) {
		if (i == 0) {
			input_size = n_ins;
		}
		else {
			input_size = hidden_layer_sizes[i - 1];
		}

		// construct sigmoid_layer
		sigmoid_layers[i] = new HiddenLayer<T>(N, input_size, hidden_layer_sizes[i], NULL, NULL);

		// construct dA_layer
		dA_layers[i] = new dA<T>(N, input_size, hidden_layer_sizes[i], \
			sigmoid_layers[i]->W, sigmoid_layers[i]->b, NULL);
	}

	// layer for output using LogisticRegression
	log_layer = new LogisticRegression<T>(N, hidden_layer_sizes[n_layers - 1], n_outs);
}

template<typename T>
SdA<T>::~SdA() {
	delete log_layer;

	for (int i = 0; i < n_layers; i++) {
		delete sigmoid_layers[i];
		delete dA_layers[i];
	}
	delete[] sigmoid_layers;
	delete[] dA_layers;
}

template<typename T>
void SdA<T>::pretrain(T *input, double lr, double corruption_level, int epochs) {
	T *layer_input = NULL;
	int prev_layer_input_size;
	T *prev_layer_input = NULL;

	T *train_X = new T[n_ins];

	for (int i = 0; i < n_layers; i++) {  // layer-wise
		printf("\nlayer :%d\n", i);
		for (int epoch = 0; epoch < epochs; epoch++) {  // training epochs
			printf("epoch :%d/%d\r", epoch, epochs);
			for (int n = 0; n < N; n++) { // input x1...xN
				// initial input
				for (int m = 0; m < n_ins; m++) train_X[m] = input[n * n_ins + m];

				// layer input
				for (int l = 0; l <= i; l++) {

					if (l == 0) {
						delete[] layer_input; layer_input = NULL;
						layer_input = new T[n_ins];
						for (int j = 0; j < n_ins; j++) layer_input[j] = train_X[j];
					}
					else {
						if (l == 1) prev_layer_input_size = n_ins;
						else prev_layer_input_size = hidden_layer_sizes[l - 2];

						delete[] prev_layer_input; prev_layer_input = NULL;
						prev_layer_input = new T[prev_layer_input_size];

						for (int j = 0; j < prev_layer_input_size; j++) prev_layer_input[j] = layer_input[j]; 

						delete[] layer_input; layer_input = NULL;
						layer_input = new T[hidden_layer_sizes[l - 1]];

						sigmoid_layers[l - 1]->sample_h_given_v(prev_layer_input, layer_input);
						
					}
				}
				dA_layers[i]->train(layer_input, lr, corruption_level);
				delete[] layer_input; layer_input = NULL;
			}
		}
	}

	delete[] train_X;
	delete[] layer_input;
}

template<typename T>
void SdA<T>::finetune(T *input, int *label, double lr, int epochs) {
	T *layer_input = NULL;
	int prev_layer_input_size;
	T *prev_layer_input = NULL;

	T *train_X = new T[n_ins];
	int *train_Y = new int[n_outs];

	for (int epoch = 0; epoch < epochs; epoch++) {
		printf("epoch :%d/%d\r", epoch, epochs);
		for (int n = 0; n < N; n++) { // input x1...xN
			// initial input
			for (int m = 0; m < n_ins; m++)  train_X[m] = input[n * n_ins + m];
			for (int m = 0; m < n_outs; m++) train_Y[m] = label[n * n_outs + m];

			// layer input
			for (int i = 0; i < n_layers; i++) {
				if (i == 0) {
					delete[] prev_layer_input; prev_layer_input = NULL;
					prev_layer_input = new T[n_ins];
					for (int j = 0; j < n_ins; j++) prev_layer_input[j] = train_X[j];
				}
				else {
					delete[] prev_layer_input; prev_layer_input = NULL;
					prev_layer_input = new T[hidden_layer_sizes[i - 1]];
					for (int j = 0; j < hidden_layer_sizes[i - 1]; j++) prev_layer_input[j] = layer_input[j];
				}

				delete[] layer_input; layer_input = NULL;
				layer_input = new T[hidden_layer_sizes[i]];
				sigmoid_layers[i]->sample_h_given_v(prev_layer_input, layer_input);
				
			}
			log_layer->train(layer_input, train_Y, lr);
		}
		// lr *= 0.95;
	}

	delete[] layer_input;
	delete[] train_X;
	delete[] train_Y;
}

template<typename T>
void SdA<T>::predict(T *x, double *y) {
	double *layer_input;
	int prev_layer_input_size;
	double *prev_layer_input;

	double linear_output;

	prev_layer_input = new double[n_ins];
	for (int j = 0; j < n_ins; j++) prev_layer_input[j] = x[j];

	// layer activation
	for (int i = 0; i < n_layers; i++) {
		layer_input = new double[sigmoid_layers[i]->n_out];

		for (int k = 0; k < sigmoid_layers[i]->n_out; k++) {
			linear_output = 0.0;

			for (int j = 0; j < sigmoid_layers[i]->n_in; j++) {
				linear_output += sigmoid_layers[i]->W[k][j] * prev_layer_input[j];
			}
			linear_output += sigmoid_layers[i]->b[k];
			layer_input[k] = sigmoid(linear_output);
		}
		delete[] prev_layer_input;

		if (i < n_layers - 1) {
			prev_layer_input = new double[sigmoid_layers[i]->n_out];
			for (int j = 0; j < sigmoid_layers[i]->n_out; j++) prev_layer_input[j] = layer_input[j];
			delete[] layer_input;
		}
	}

	for (int i = 0; i < log_layer->n_out; i++) {
		y[i] = 0;
		for (int j = 0; j < log_layer->n_in; j++) {
			y[i] += log_layer->W[i][j] * layer_input[j];
		}
		y[i] += log_layer->b[i];
	}

	log_layer->softmax(y);


	delete[] layer_input;
}


// HiddenLayer
template<typename T>
HiddenLayer<T>::HiddenLayer(int size, int in, int out, double **w, double *bp) {
	N = size;
	n_in = in;
	n_out = out;

	if (w == NULL) {
		W = new double*[n_out];
		for (int i = 0; i < n_out; i++) W[i] = new double[n_in];
		double a = 1.0 / n_in;

		for (int i = 0; i < n_out; i++) {
			for (int j = 0; j < n_in; j++) {
				W[i][j] = uniform(-a, a);
			}
		}
	}
	else {
		W = w;
	}

	if (bp == NULL) {
		b = new double[n_out];
	}
	else {
		b = bp;
	}
}

template<typename T>
HiddenLayer<T>::~HiddenLayer() {
	for (int i = 0; i < n_out; i++) delete W[i];
	delete[] W;
	delete[] b;
}

template<typename T>
double HiddenLayer<T>::output(T *input, double *w, double b) {
	double linear_output = 0.0;
	for (int j = 0; j < n_in; j++) {
		linear_output += w[j] * input[j];
	}
	linear_output += b;
	return sigmoid(linear_output);
}

template<typename T>
void HiddenLayer<T>::sample_h_given_v(T *input, T *sample) {
	for (int i = 0; i < n_out; i++) {
		sample[i] = binomial(1, output(input, W[i], b[i]));//binomial(1, output(input, W[i], b[i]));
	}
}


// dA
template<typename T>
dA<T>::dA(int size, int n_v, int n_h, double **w, double *hb, double *vb) {
	N = size;
	n_visible = n_v;
	n_hidden = n_h;

	if (w == NULL) {
		W = new double*[n_hidden];
		for (int i = 0; i < n_hidden; i++) W[i] = new double[n_visible];
		double a = 1.0 / n_visible;

		for (int i = 0; i < n_hidden; i++) {
			for (int j = 0; j < n_visible; j++) {
				W[i][j] = uniform(-a, a);
			}
		}
	}
	else {
		W = w;
	}

	if (hb == NULL) {
		hbias = new double[n_hidden];
		for (int i = 0; i < n_hidden; i++) hbias[i] = 0;
	}
	else {
		hbias = hb;
	}

	if (vb == NULL) {
		vbias = new double[n_visible];
		for (int i = 0; i < n_visible; i++) vbias[i] = 0;
	}
	else {
		vbias = vb;
	}
}

template<typename T>
dA<T>::~dA() {
	// for(int i=0; i<n_hidden; i++) delete[] W[i];
	// delete[] W;
	// delete[] hbias;
	delete[] vbias;
}

template<typename T>
void dA<T>::get_corrupted_input(T *x, T *tilde_x, double p) {
	for (int i = 0; i < n_visible; i++) {
		if (x[i] == 0) {
			tilde_x[i] = 0;
		}
		else {
			tilde_x[i] = binomial(x[i], p); //binomial(1, p);
		}
	}
}

// Encode
template<typename T>
void dA<T>::get_hidden_values(T *x, double *y) {
	for (int i = 0; i < n_hidden; i++) {
		y[i] = 0;
		for (int j = 0; j < n_visible; j++) {
			y[i] += W[i][j] * x[j];
		}
		y[i] += hbias[i];
		y[i] = sigmoid(y[i]);
	}
}

// Decode
template<typename T>
void dA<T>::get_reconstructed_input(double *y, double *z) {
	for (int i = 0; i < n_visible; i++) {
		z[i] = 0;
		for (int j = 0; j < n_hidden; j++) {
			z[i] += W[j][i] * y[j];
		}
		z[i] += vbias[i];
		z[i] = sigmoid(z[i]);
	}
}

template<typename T>
void dA<T>::train(T *x, double lr, double corruption_level) {
	T *tilde_x = new T[n_visible];
	double *y = new double[n_hidden];
	double *z = new double[n_visible];

	double *L_vbias = new double[n_visible];
	double *L_hbias = new double[n_hidden];

	double p = 1 - corruption_level;

	get_corrupted_input(x, tilde_x, p);
	get_hidden_values(tilde_x, y);
	get_reconstructed_input(y, z);

	// vbias
	for (int i = 0; i < n_visible; i++) {
		L_vbias[i] = x[i] - z[i];
		vbias[i] += lr * L_vbias[i] / N;
	}

	// hbias
	for (int i = 0; i < n_hidden; i++) {
		L_hbias[i] = 0;
		for (int j = 0; j < n_visible; j++) {
			L_hbias[i] += W[i][j] * L_vbias[j];
		}
		L_hbias[i] *= y[i] * (1 - y[i]);

		hbias[i] += lr * L_hbias[i] / N;
	}

	// W
	for (int i = 0; i < n_hidden; i++) {
		for (int j = 0; j < n_visible; j++) {
			W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
		}
	}

	delete[] L_hbias;
	delete[] L_vbias;
	delete[] z;
	delete[] y;
	delete[] tilde_x;
}

template<typename T>
void dA<T>::reconstruct(T *x, double *z) {
	double *y = new double[n_hidden];

	get_hidden_values(x, y);
	get_reconstructed_input(y, z);

	delete[] y;
}


// LogisticRegression
template<typename T>
LogisticRegression<T>::LogisticRegression(int size, int in, int out) {
	N = size;
	n_in = in;
	n_out = out;

	W = new double*[n_out];
	for (int i = 0; i < n_out; i++) W[i] = new double[n_in];
	b = new double[n_out];

	for (int i = 0; i < n_out; i++) {
		for (int j = 0; j < n_in; j++) {
			W[i][j] = 0;
		}
		b[i] = 0;
	}
}

template<typename T>
LogisticRegression<T>::~LogisticRegression() {
	for (int i = 0; i < n_out; i++) delete[] W[i];
	delete[] W;
	delete[] b;
}

template<typename T>
void LogisticRegression<T>::train(T *x, int *y, double lr) {
	double *p_y_given_x = new double[n_out];
	double *dy = new double[n_out];

	for (int i = 0; i < n_out; i++) {
		p_y_given_x[i] = 0;
		for (int j = 0; j < n_in; j++) {
			p_y_given_x[i] += W[i][j] * x[j];
		}
		p_y_given_x[i] += b[i];
	}
	softmax(p_y_given_x);

	for (int i = 0; i < n_out; i++) {
		dy[i] = y[i] - p_y_given_x[i];

		for (int j = 0; j < n_in; j++) {
			W[i][j] += lr * dy[i] * x[j] / N;
		}

		b[i] += lr * dy[i] / N;
	}

	delete[] p_y_given_x;
	delete[] dy;
}

template<typename T>
void LogisticRegression<T>::softmax(double *x) {
	double max = 0.0;
	double sum = 0.0;

	for (int i = 0; i < n_out; i++) if (max < x[i]) max = x[i];
	for (int i = 0; i < n_out; i++) {
		x[i] = exp(x[i] - max);
		sum += x[i];
	}

	for (int i = 0; i < n_out; i++) x[i] /= sum;
}

template<typename T>
void LogisticRegression<T>::predict(T *x, double *y) {
	for (int i = 0; i < n_out; i++) {
		y[i] = 0;
		for (int j = 0; j < n_in; j++) {
			y[i] += W[i][j] * x[j];
		}
		y[i] += b[i];
	}

	softmax(y);
}

