#include "approx_solver.hpp"
#include <vector>
#include <math.h>
using std::vector;
double sigmoid(const fcolvec& w, const fcolvec& x);

void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter, int *runtime, const fmat &Xtest, const icolvec &Ytest) {

  clock_t init, final;  // record the training time.
  init = clock();

  int d = X.n_cols;
  int N = X.n_rows;

  double eta = 0.003;  // learning rate
  double thresholdSq = 0.003; // 

  // initialize wmap uniform random on [-1,1]
  w.randu(); // uniform random on [0,1]
  double lambda = 1/s0/s0;

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  int i;
  for(i = 0; change_w > thresholdSq && i < max_iter; i++) {
    if (i % 1000 == 0) {
      cout << "grad desc: iteration " << i << 
      " ||w||^2_2 = " << as_scalar(sum(w.t() * w)) << endl;
      cout << "change_w = " << change_w << endl; 
      double accuracy = classification_accuracy(Xtest, Ytest, w);
      cout << "accuracy = " << accuracy << endl;
    }

    fcolvec update_w(d);
    update_w.zeros();

    // compute \sum_n (t_n - y_n) \phi_n
    for(int n = 0; n < N; n++) {
      update_w += as_scalar(Y(n) - sigmoid(w, X.row(n).t())) * X.row(n).t();
      //cout << "Y = " << as_scalar(Y(n)) << "; coeff " << n << " = " << as_scalar(Y(n) - pow(1 + exp(-w.t() * X[n]), -1)) << endl;
    }

    update_w -= lambda * w;
    //update_w *= eta;
    update_w *= eta * pow(200 + i, -0.5);
    //change_w = as_scalar(sum(pow(update_w,2)));
    fcolvec update_w_abs = abs(update_w);
    change_w = as_scalar(update_w_abs.max());
    w += update_w;
  }

  if (i == max_iter) {
    cerr << "Gradient descent failed to converge." << endl;
    //exit(1);
  }

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
  *runtime = time_elapsed;
  double time_per_iter = (double) time_elapsed / i;
  cout << "Gradient descent converged in " << i << " steps, totaling " 
    <<  time_elapsed << " seconds (" << time_per_iter << " seconds/iter." 
    << endl;
}





/**
 *  Delta Method
 */
void delta_variational(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter, int *runtime, const fmat &Xtest, const icolvec &Ytest) {

  clock_t init, final;  // record the training time.
  init = clock();

  int d = X.n_cols;
  int N = X.n_rows;

  double eta = 0.005;  // learning rate, was 0.0001
  //double eta = 1./2./d/2.;  // learning rate, was 0.0001
  cout << "initial eta = " << eta << endl;
  //double thresholdSq = 0.001 * 0.001; // (convergence threshold)^2
  //double threshold = 0.003; // used for yeast
  double threshold = 0.0001; // spambase 

  // initialize wmap uniform random on [-1,1]
  w.randu(); // uniform random on [0,1]
  double lambda = 1/s0/s0;
  fmat S0 = lambda * eye<fmat>(d, d);

  double change_w = 9999.;  // ||w_new - w_old||^2, initialized to large number

  int i;
  for(i = 0; change_w > threshold && i < max_iter; i++) {
    if (i % 10 == 0) {
      cout << "grad desc: iteration " << i << 
      " ||w||^2_2 = " << as_scalar(sum(w.t() * w)) << endl;
      cout << "change_w = " << change_w << endl; 
      double accuracy = classification_accuracy(Xtest, Ytest, w);
      cout << "accuracy = " << accuracy << endl;
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
    double t0 = 200;
    update_w *= eta * pow(200 + i, -0.5);
    fcolvec update_w_abs = abs(update_w);
    change_w = as_scalar(update_w_abs.max());
    if (i == 0) cout << "iter 0: change_w = " << change_w << endl;
    w += update_w;
  }

  if (i == max_iter) {
    cerr << "Gradient descent failed to converge." << endl;
    //exit(1);
  }

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);
  *runtime = time_elapsed;
  double time_per_iter = (double) time_elapsed / i;
  cout << "Gradient descent (Delta method) converged in " << i << " steps, totaling " 
    <<  time_elapsed << " seconds (" << time_per_iter << " seconds/iter." 
    << endl;

}


double sigmoid(const fcolvec& w, const fcolvec& x) {
  return as_scalar(pow(1 + exp(-w.t() * x), -1));
}
