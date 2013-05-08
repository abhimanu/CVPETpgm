#include "util.hpp"
#include <fstream>
using std::ifstream;


/**
 * Read filename into X and Y. filename must be a csv with the class label
 * being last column. Do some conversion so that class label Y \in {0,1}.
 * Note that if the data set has d' dimension, we have d = d' + 1 for the
 * intercept (augmented design matrix).
 *
 * X - [N x d+1] augmented design matrix. X[:,1] = 1.
 */
void read_data(string &filename, fmat &X, icolvec &Y, int *d_aug, int label_c1) {
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
  Y = conv_to<icolvec>::from(m.col(m.n_cols - 1));
  for (int n = 0; n < N; n++) {
    if (Y(n) == label_c1) Y(n) = 1;
    else Y(n) = 0;
  }
  
  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;
  cout << "class split: Y(i) = 1 : " << double(as_scalar(sum(Y)))/double(Y.n_rows) << endl;
  cout << "class 1: " << double(as_scalar(sum(Y))) << endl;
  cout << "n_data = " << double(Y.n_rows) << endl;

  cout << "X(0, 0...9) = " << X(0, span(0, 9)) << endl << endl;
  cout << "Y(0) = " << Y(0) << endl; 

}

/**
 * filename needs to be space delimited, not csv, and need prior knowledge
 * of N and d (augmented).
 */
void read_data_fast(string &filename, fmat &X, icolvec &Y, int N, int d_aug, int label_c1) {
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
  Y(0) = (label == label_c1) ? 1 : 0;

  int i;
  for(i = 1; !indata.eof() && i < N; i++) {
    for (int j = 1; j < d_aug; j++) { // read first line
      // read attributes
      float attr;
      indata >> attr;
      X(i, j) = attr;
    }
    indata >> label;  // reading class label 
    Y(i) = (label == label_c1) ? 1 : 0;
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



/**
 * Return the classification error \in [0, 1].
 */
double classification_accuracy(const fmat &Xtest, const icolvec &Ytest, const fcolvec &w) {
  int Ntest = Xtest.n_rows;
  icolvec Ypred(Ntest);  // prediction

  // The followings is to simulate sign function. (sign(X_test * w)+1)./2
  fcolvec lin_pred = conv_to<fcolvec>::from(Xtest * w);
  for(int n = 0; n < Ntest; n++)  Ypred(n) = (lin_pred(n) > 0) ? 1 : 0;

  icolvec diff = Ypred - Ytest;

  int correct_count = 0;
  for(int n = 0; n < Ntest; n++) correct_count += (diff(n) == 0) ? 1 : 0;

  return (double) correct_count / Ntest;
}
