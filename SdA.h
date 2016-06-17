#ifndef __sda__H___
#define __sda__H___

/*
 * SdA.cpp (Stacked Denoising Autoencoders)
 *
 * @author  yusugomori (http://yusugomori.com)
 * @usage   $ g++ SdA.cpp
 *
 */


/* HiddenLayer.h */
template<typename T>
class HiddenLayer {

public:
	int N;
	int n_in;
	int n_out;
	double **W;
	double *b;
	HiddenLayer(int size, int in, int out, double **w, double *bp);
	~HiddenLayer();
	double output(T *input, double *w, double b);
	void sample_h_given_v(T *input, T *sample);
};

/* dA.h */
template<typename T>
class dA {

public:
	int N;
	int n_visible;
	int n_hidden;
	double **W;
	double *hbias;
	double *vbias;
	dA(int size, int n_v, int n_h, double **w, double *hb, double *vb);
	~dA();
	void get_corrupted_input(T *x, T *tilde_x, double p);
	void get_hidden_values(T *x, double *y);
	void get_reconstructed_input(double *y, double *z);
	void train(T *x, double lr, double corruption_level);
	void reconstruct(T *x, double *z);
};

/* LogisticRegression.h */
template<typename T>
class LogisticRegression {

public:
	int N;  // num of inputs
	int n_in;
	int n_out;
	double **W;
	double *b;
	LogisticRegression(int size, int in, int out);
	~LogisticRegression();
	void train(T *x, int *y, double lr);
	void softmax(double *x);
	void predict(T *x, double *y);
};

/* SdA.h */
template<typename T>
class SdA {

public:
	int N;
	int n_ins;
	int *hidden_layer_sizes;
	int n_outs;
	int n_layers;
	HiddenLayer<T> **sigmoid_layers;
	dA<T> **dA_layers;
	LogisticRegression<T> *log_layer;
	SdA(int size, int n_i, int *hls, int n_o, int n_l);
	~SdA();
	void pretrain(T *input, double lr, double corruption_level, int epochs);
	void finetune(T *input, int *label, double lr, int epochs);
	void predict(T *x, double *y);
};

#endif
