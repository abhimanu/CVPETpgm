#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <armadillo> // Matrix operation library

#include <fstream>
using std::ifstream;

#include <cstdlib> // for exit function

using std::string;

using namespace arma;

const int N = 435;
const int d = 16;
const double sigma = 1.;

void wmap_grad_desc(const colvec *X, const colvec& Y, colvec& w);

int main(int argc, char** argv) {

  string data_file = "../dataset/hw1-data.txt";

  // data matrix X:
  colvec *X = new colvec[N];
  for (int i = 0; i < N; i++) {
    X[i].set_size(d+1); // data X = [d+1 x N]
    X[i](0) = 1;        // X[0, i] = 1 is the intercept
  }
  //mat X(N, d+1);    // X(i, 0) = 1
  //X.ones();
  
  // class label Y
  colvec Y(N);

  // Reading data
  ifstream indata;
  indata.open(data_file.c_str()); // opens the file
  if(!indata) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  char label;
  indata >> label;  // reading class label 
  Y(0) = (label == 'A') ? 0 : 1;  // 'A' = 0, 'B' = 1
  for (int j = 1; j <= d; j++) { // read first line
    // read attributes
    int attr;
    indata >> attr;
    X[0](j) = attr;
  }
  for(int i = 1; !indata.eof() && i < N; i++) {
    indata >> label;  // reading class label 
    Y(i) = (label == 'A') ? 0 : 1;  // 'A' = 0, 'B' = 1
    for (int j = 1; j <= d; j++) { // read first line
      // read attributes
      int attr;
      indata >> attr;
      X[i](j) = attr;
    }
  }

  colvec w(d+1);
  wmap_grad_desc(X, Y, w);
  cout << "Final w: " << endl;
  cout << w << endl;

  return 0;
}

void wmap_grad_desc(const colvec *X, const colvec& Y, colvec& w) {

  double eta = 0.0001;  // learning rate
  double thresholdSq = 0.001 * 0.001; // (convergence threshold)^2

  // initialize wmap uniform random on [-1,1]
  //w.randu(); // uniform random on [0,1]
  //w = w * 2 - 1;
  //w = {0.8315, 0.5844, 0.9190, 0.3115, -0.9286, 0.6983, 0.8680, 0.3575
  //     , 0.5155, 0.4863, -0.2155, 0.3110, -0.6576, 0.4121, -0.9363
  //     , -0.4462, -0.9077};
  w.ones();

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  for(int i = 0; change_w > thresholdSq; i++) {
  //for(int i = 0; i < 1; i++) {
    cout << "grad desc: iteration " << i << endl;

    colvec update_w(d+1);
    update_w.zeros();

    // compute \sum_n (t_n - y_n) \phi_n
    for(int n = 0; n < N; n++) {
      update_w += as_scalar(Y(n) - pow(1 + exp(-w.t() * X[n]), -1)) * X[n];
      //cout << "Y = " << as_scalar(Y(n)) << "; coeff " << n << " = " << as_scalar(Y(n) - pow(1 + exp(-w.t() * X[n]), -1)) << endl;
    }

    update_w -= (1/sigma/sigma) * w;
    update_w *= eta;
    change_w = as_scalar(sum(pow(update_w,2)));
    w += update_w;
  }
}
