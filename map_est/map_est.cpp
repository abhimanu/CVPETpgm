//#include <glog/logging.h>   // Use Google logging library
#include <gflags/gflags.h>  // Use Google commandline library

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <armadillo> // Matrix operation library

#include <fstream>
using std::ifstream;

#include <cstdlib> // for exit function

using std::string;

#include <time.h>

DEFINE_int32(n_init, 1, "number of random initialization in gradient descent.");
DEFINE_string(trainfile, "file not exist", "CSV data file path for training set.");
DEFINE_string(testfile, "file not exist", "CSV data file path for test set.");
DEFINE_double(sigma, 1., "sigma for gaussian prior.");
DEFINE_int32(max_iter, 500, "maximum number of iterations in gradient descent.");
DEFINE_int32(Ntrain, 0, "number of training data instances.");
DEFINE_int32(Ntest, 0, "number of test data instances.");
DEFINE_int32(d, 0, "data dimension.");

DEFINE_int32(labelC1, -2, "class 1 label.");
DEFINE_int32(labelC2, -2, "class 2 label.");


using namespace arma;


void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double sigma, int n_init);
void read_data(string &filename, fmat &X, icolvec &Y, int &N, int &d);
void read_data_fast(string &filename, fmat &X, icolvec &Y, int N, int d_aug);
double classification_error(const fmat &Xtest, const icolvec &Ytest, 
    const fcolvec &w);


int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  icolvec Ytrain;
  //int Ntrain, d;
  fmat Xtrain;
  //read_data(FLAGS_trainfile, Xtrain, Ytrain, Ntrain, d);
  int Ntrain = FLAGS_Ntrain, d_aug = FLAGS_d + 1;
  read_data_fast(FLAGS_trainfile, Xtrain, Ytrain, Ntrain, d_aug);

  fcolvec w(d_aug);
  wmap_grad_desc(Xtrain, Ytrain, w, FLAGS_sigma, FLAGS_n_init);
  cout << "||w||^2 = " << as_scalar(sum(w.t() * w)) << endl;

  double training_error = classification_error(Xtrain, Ytrain, w);
  cout << "training error = " << training_error << endl;

  // Testing
  int Ntest = FLAGS_Ntest;
  fmat Xtest;
  icolvec Ytest;
  //read_data(FLAGS_testfile, Xtest, Ytest, Ntest, d);
  read_data_fast(FLAGS_testfile, Xtest, Ytest, Ntest, d_aug);
  
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
void read_data(string &filename, fmat &X, icolvec &Y, int &N, int &d) {
  fmat m;    // temporary variable
  
  clock_t init, final;  // record the file reading time.
  init = clock();

  m.load(filename);
  N = m.n_rows;
  d = m.n_cols; // last column is class label, but we augment it.

  X.resize(N, d);
  X.ones();
  X.cols(1, d-1) = m.cols(0, d-2);  // first row of X is 1's.

  Y.resize(N);
  Y = conv_to<icolvec>::from(m.col(d-1));
  Y = (Y + 1)/2;  // change from {-1,1} to {0,1}.
  
  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;
}

/**
 * filename needs to be space delimited, not csv, and need prior knowledge
 * of N and d (augmented).
 */
void read_data_fast(string &filename, fmat &X, icolvec &Y, int N, int d_aug) {
  clock_t init, final;  // record the file reading time.
  init = clock();

  // For multi-class, we'll have < N instances.
  X.resize(N, d_aug);
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
  Y(0) = label;

  for(int i = 1; !indata.eof() && i < N; i++) {
    for (int j = 1; j < d_aug; j++) { // read first line
      // read attributes
      int attr;
      indata >> attr;
      X(i, j) = attr;
    }
    indata >> label;  // reading class label 
    Y(i) = label;
  }
  
  Y = (Y + 1)/2;  // change from {-1,1} to {0,1}.

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d_aug << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;
}


void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double sigma, int n_init) {

  clock_t init, final;  // record the training time.
  init = clock();

  int d = X.n_cols;
  int N = X.n_rows;

  double eta = 0.0001;  // learning rate
  double thresholdSq = 0.001 * 0.001; // (convergence threshold)^2

  // initialize wmap uniform random on [-1,1]
  w.randu(); // uniform random on [0,1]
  double lambda = 1/sigma/sigma;

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  int i;
  for(i = 0; change_w > thresholdSq && i < FLAGS_max_iter; i++) {
  //for(int i = 0; i < 1; i++) {
    cout << "grad desc: iteration " << i << endl;

    fcolvec update_w(d);
    update_w.zeros();

    // compute \sum_n (t_n - y_n) \phi_n
    for(int n = 0; n < N; n++) {
      update_w += as_scalar(Y(n) - pow(1 + exp(-w.t() * X.row(n).t()), -1)) * X.row(n).t();
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
