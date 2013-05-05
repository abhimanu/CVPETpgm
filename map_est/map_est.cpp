//#include <glog/logging.h>   // Use Google logging library
#include <gflags/gflags.h>  // Use Google commandline library

#include <iostream>
#include <cassert>
using std::cerr;
using std::cout;
using std::endl;

#include <armadillo> // Matrix operation library

#include <fstream>
using std::ifstream;

#include <cstdlib> // for exit function

using std::string;

#include <vector>
using std::vector;

#include <time.h>

DEFINE_int32(n_init, 1, "number of random initialization in gradient descent.");
DEFINE_string(trainfile, "file not exist", "CSV data file path for training set.");
DEFINE_string(testfile, "file not exist", "CSV data file path for test set.");
DEFINE_double(s0, 1., "sigma for gaussian prior.");
DEFINE_int32(max_iter, 500, "maximum number of iterations in gradient descent.");
//DEFINE_int32(Ntrain, 0, "number of training data instances.");
//DEFINE_int32(Ntest, 0, "number of test data instances.");
//DEFINE_int32(d, 0, "data dimension.");
DEFINE_int32(labelC1, -2, "class 1 label.");  // The rest will be class 2. Class 1 has Y = 0
DEFINE_bool(save_binary, false, "save Armadillo binary after loading the data.");


using namespace arma;


void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init);
void read_data(string &filename, fmat &X, icolvec &Y, int* d_aug);
void read_data_fast(string &filename, fmat &X, icolvec &Y, int* d_aug);
double classification_error(const fmat &Xtest, const icolvec &Ytest, 
    const fcolvec &w);
double sigmoid(const fcolvec& w, const fcolvec& x);
void delta_variational(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init);


int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  icolvec Ytrain;
  fmat Xtrain;
  int d_aug;
  read_data(FLAGS_trainfile, Xtrain, Ytrain, &d_aug);
  //read_data_fast(FLAGS_trainfile, Xtrain, Ytrain, FLAGS_Ntrain, d_aug);
  
  // compare the two Xtrain
  //cout << "diff Xtrain = " << as_scalar(accu(Xtrain - Xtrain2)) << endl;

  // TODO: remove below!!
  d_aug = 100;
  Xtrain = Xtrain.cols(0, 99);

  if (FLAGS_save_binary) {
    Xtrain.save(FLAGS_trainfile + ".X.arm");
    Ytrain.save(FLAGS_trainfile + ".Y.arm");
  }

  fcolvec w(d_aug);
  //wmap_grad_desc(Xtrain, Ytrain, w, FLAGS_s0, FLAGS_n_init);
  delta_variational(Xtrain, Ytrain, w, FLAGS_s0, FLAGS_n_init);
  cout << "||w||^2 = " << as_scalar(sum(w.t() * w)) << endl;

  double training_error = classification_error(Xtrain, Ytrain, w);
  cout << "training error = " << training_error << endl;

  // Testing
  fmat Xtest;
  icolvec Ytest;
  int d_aug_test;
  read_data(FLAGS_testfile, Xtest, Ytest, &d_aug_test);
  // TODO: remove followings
  //assert(d_aug_test == d_aug);
  Xtest = Xtest.cols(0,99);


  //read_data_fast(FLAGS_testfile, Xtest, Ytest, FLAGS_Ntest, d_aug);
  


  if (FLAGS_save_binary) {
    Xtest.save(FLAGS_testfile + ".X.arm");
    Ytest.save(FLAGS_testfile + ".Y.arm");
  }
  
  double test_error = classification_error(Xtest, Ytest, w);
  cout << "classification error = " << test_error << endl;

  return 0;
}


/**
 * Read filename into X and Y. filename must be a csv with the class label
 * being last column. Do some conversion so that class label Y \in {0,1}.
 * Note that if the data set has d' dimension, we have d = d' + 1 for the
 * intercept (augmented design matrix).
 *
 * X - [N x d+1] augmented design matrix. X[:,1] = 1.
 */
void read_data(string &filename, fmat &X, icolvec &Y, int *d_aug) {
  cout << "Reading using armadillo load: " << filename << endl;
  fmat m;    // temporary variable
  
  clock_t init, final;  // record the file reading time.
  init = clock();

  m.load(filename);
  int N = m.n_rows;
  int d = m.n_cols - 1; // last column is class label, but we augment it.
  *d_aug = d + 1;

  X.resize(N, d + 1);
  X.ones();
  X.cols(1, d) = m.cols(0, d-1);  // first row of X is 1's.

  Y.resize(N);
  Y = conv_to<icolvec>::from(m.col(d-1));
  for (int n = 0; n < N; n++) {
    if (Y(n) == FLAGS_labelC1) Y(n) = 0;
    else Y(n) = 1;
  }
  
  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;

  cout << "X(0, 0...9) = " << X(0, span(0, 9)) << endl << endl;
  cout << "Y(0) = " << Y(0) << endl; 
}

/**
 * filename needs to be space delimited, not csv, and need prior knowledge
 * of N and d (augmented).
 */
void read_data_fast(string &filename, fmat &X, icolvec &Y, int N, int d_aug) {
  cout << "Start fast reading " << filename << endl;
  clock_t init, final;  // record the file reading time.
  init = clock();

  // For multi-class, we'll have < N instances.
  X.resize(N, d_aug);
  X.ones(); // make sure the first column is 1.
  Y.resize(N);

  ifstream indata;
  indata.open(filename.c_str()); // opens the file
  if(!indata) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  for (int j = 1; j < d_aug; j++) { // read first line
    // read attributes
    float attr;
    indata >> attr;
    X(0, j) = attr;
  }
  int label;
  indata >> label;  // reading class label 
  Y(0) = (label == FLAGS_labelC1) ? 0 : 1;

  int i;
  for(i = 1; !indata.eof() && i < N; i++) {
    for (int j = 1; j < d_aug; j++) { // read first line
      // read attributes
      float attr;
      indata >> attr;
      X(i, j) = attr;
    }
    indata >> label;  // reading class label 
    Y(i) = (label == FLAGS_labelC1) ? 0 : 1;
  }

  if (i < N) {
    cerr << "Warning: expected " << N << 
    " data instances, but only got " << i << endl;
  }
 
  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d_aug << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;
}


void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init) {

  clock_t init, final;  // record the training time.
  init = clock();

  int d = X.n_cols;
  int N = X.n_rows;

  double eta = 0.0001;  // learning rate
  double thresholdSq = 0.001 * 0.001; // (convergence threshold)^2

  // initialize wmap uniform random on [-1,1]
  w.randu(); // uniform random on [0,1]
  double lambda = 1/s0/s0;

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  int i;
  for(i = 0; change_w > thresholdSq && i < FLAGS_max_iter; i++) {
    if (i % 10 == 0) {
      cout << "grad desc: iteration " << i << 
      " ||w||^2_2 = " << as_scalar(sum(w.t() * w)) << endl;
    }

    fcolvec update_w(d);
    update_w.zeros();

    // compute \sum_n (t_n - y_n) \phi_n
    for(int n = 0; n < N; n++) {
      update_w += as_scalar(Y(n) - sigmoid(w, X.row(n).t())) * X.row(n).t();
      //cout << "Y = " << as_scalar(Y(n)) << "; coeff " << n << " = " << as_scalar(Y(n) - pow(1 + exp(-w.t() * X[n]), -1)) << endl;
    }

    update_w -= lambda * w;
    update_w *= eta;
    change_w = as_scalar(sum(pow(update_w,2)));
    w += update_w;
  }

  if (i == FLAGS_max_iter) {
    cerr << "Gradient descent failed to converge." << endl;
    exit(1);
  }

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
  double time_per_iter = (double) time_elapsed / i;
  cout << "Gradient descent converged in " << i << " steps, totaling " 
    <<  time_elapsed << " seconds (" << time_per_iter << " seconds/iter." 
    << endl;
}


/**
 * Return the classification error \in [0, 1].
 */
double classification_error(const fmat &Xtest, const icolvec &Ytest, const fcolvec &w) {
  int Ntest = Xtest.n_rows;
  icolvec Ypred(Ntest);  // prediction

  // The followings is to simulate sign function. (sign(X_test * w)+1)./2
  fcolvec lin_pred = conv_to<fcolvec>::from(Xtest * w);
  for(int n = 0; n < Ntest; n++)  Ypred(n) = (lin_pred(n) > 0) ? 1 : 0;

  icolvec diff = Ypred - Ytest;

  int error_count = 0;
  for(int n = 0; n < Ntest; n++) error_count += (diff(n) == 0) ? 0 : 1;

  return (double) error_count / Ntest;
}




/**
 *  Delta Method
 */
void delta_variational(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init) {

  clock_t init, final;  // record the training time.
  init = clock();

  int d = X.n_cols;
  int N = X.n_rows;

  double eta = 0.0001;  // learning rate
  double thresholdSq = 0.001 * 0.001; // (convergence threshold)^2

  // initialize wmap uniform random on [-1,1]
  w.randu(); // uniform random on [0,1]
  double lambda = 1/s0/s0;
  fmat S0 = lambda * eye<fmat>(d, d);

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  int i;
  for(i = 0; change_w > thresholdSq && i < FLAGS_max_iter; i++) {
    if (i % 10 == 0) {
      cout << "grad desc: iteration " << i << 
      " ||w||^2_2 = " << as_scalar(sum(w.t() * w)) << endl;
    }

    fcolvec update_w(d);
    update_w.zeros();
  
    //clock_t init, final;  // record the training time.
    //init = clock();

    // compute Sigma at t-1
    vector<double> sigmoid_vals(N);
    for(int n = 0; n < N; n++) {
      sigmoid_vals[n] = sigmoid(w, X.row(n).t());
    }
  
    //final = clock() - init;
    //int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
    //cout << "sigmoid takes " << time_elapsed << endl;
    
    //init = clock();

    fmat prev_sigma = S0;
    for(int n = 0; n < N; n++) {
      prev_sigma += sigmoid_vals[n] * (1-sigmoid_vals[n]) * X.row(n).t() * X.row(n);
    }
    
    //final = clock() - init;
    //time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
    //cout << "prev_sigma takes " << time_elapsed << endl;

    //init = clock();

    for(int n = 0; n < N; n++) {
      // gradient of f(mu)
      double sigmoid_val = sigmoid_vals[n];
      update_w += as_scalar(Y(n) - sigmoid_val) * X.row(n).t();

      // gradient of trace
      fcolvec b = solve(prev_sigma, X.row(n).t());
      update_w += -0.5 * sigmoid_val * (1-sigmoid_val) 
        * (1 - 2*sigmoid_val) * X.row(n).t() * X.row(n) * b;
    }
    
    //final = clock() - init;
    //time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
    //cout << "update_w computation takes " << time_elapsed<< endl;

    //update_w -= lambda * w;
    update_w *= eta;
    change_w = as_scalar(sum(pow(update_w,2)));
    w += update_w;
  }

  if (i == FLAGS_max_iter) {
    cerr << "Gradient descent failed to converge." << endl;
    exit(1);
  }

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
  double time_per_iter = (double) time_elapsed / i;
  cout << "Gradient descent (Delta method) converged in " << i << " steps, totaling " 
    <<  time_elapsed << " seconds (" << time_per_iter << " seconds/iter." 
    << endl;

}


double sigmoid(const fcolvec& w, const fcolvec& x) {
  return as_scalar(pow(1 + exp(-w.t() * x), -1));
}
