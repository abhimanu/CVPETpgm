//#include <glog/logging.h>   // Use Google logging library
#include <gflags/gflags.h>  // Use Google commandline library

#include "util.hpp"
#include "approx_solver.hpp"

#include <iostream>
#include <cassert>
#include <fstream>
#include <iterator>
using std::cerr;
using std::cout;
using std::endl;

#include <armadillo> // Matrix operation library
using namespace arma;

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
//DEFINE_int32(Ntrain, 0, "number of training data instances.");
//DEFINE_int32(Ntest, 0, "number of test data instances.");
//DEFINE_int32(d, 0, "data dimension.");
DEFINE_bool(save_binary, false, "save Armadillo binary after loading the data.");
DEFINE_string(algo, "laplace", "laplace/delta");
DEFINE_int32(labelC1, -2, "class 1 label.");  // The rest will be class 2. Class 1 has Y = 1
DEFINE_int32(max_iter, 500, "maximum number of iterations in gradient descent.");


void read_data_yeast(string &filename, fmat &X, imat &Y, int *d_aug);



int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  // options
  string algo = FLAGS_algo;
  assert(algo == "laplace" || algo == "delta");
  cout << "running " << algo << endl;

  //icolvec Ytrain, Ytest;  // single class
  imat Ytrain, Ytest;       // multi class
  fmat Xtrain, Xtest;
  int d_aug, d_aug_test;
  read_data_yeast(FLAGS_trainfile, Xtrain, Ytrain, &d_aug); // read training data
  read_data_yeast(FLAGS_testfile, Xtest, Ytest, &d_aug_test); // read testing data
  assert(d_aug_test == d_aug);
  
  // compare the two Xtrain
  //cout << "diff Xtrain = " << as_scalar(accu(Xtrain - Xtrain2)) << endl;

  if (FLAGS_save_binary) {
    Xtrain.save(FLAGS_trainfile + ".X.arm");
    Ytrain.save(FLAGS_trainfile + ".Y.arm");
  }
  int n_class = 14;
  vector<double> train_accuracies(n_class);
  vector<double> test_accuracies(n_class);
  vector<double> runtimes(n_class);
  double train_accuracy_total = 0;
  double test_accuracy_total = 0;


  fmat w_all(d_aug, n_class);
  for (int iclass = 0; iclass < n_class; iclass++) {
    fcolvec w(d_aug);
    int runtime;
    if (algo == "laplace") {
      wmap_grad_desc(Xtrain, Ytrain.col(iclass), w, FLAGS_s0, FLAGS_n_init, FLAGS_max_iter, &runtime);
    }
    else if (algo == "delta") {
      delta_variational(Xtrain, Ytrain.col(iclass), w, FLAGS_s0, FLAGS_n_init, FLAGS_max_iter, &runtime);
    }
    cout << "||w||^2 = " << as_scalar(sum(w.t() * w)) << endl;

    double train_accuracy = classification_accuracy(Xtrain, Ytrain.col(iclass), w);
    double test_accuracy = classification_accuracy(Xtest, Ytest.col(iclass), w);
    
    // store the result
    train_accuracies[iclass] = train_accuracy;
    test_accuracies[iclass] = test_accuracy;
    runtimes[iclass] = runtime;

    train_accuracy_total += train_accuracy;
    test_accuracy_total += test_accuracy;

    w_all.col(iclass) = w;

    cout << "train accuracy (class " << iclass << ") = " 
      << train_accuracy << endl;
    cout << "test accuracy (class " << iclass << ") = " 
      << train_accuracy << endl;
  }
  cout << "average train accuracy = " << train_accuracy_total / n_class 
    << endl;
  cout << "average test accuracy = " << test_accuracy_total / n_class 
    << endl;

  if (FLAGS_save_binary) {
    Xtest.save(FLAGS_testfile + ".X.arm");
    Ytest.save(FLAGS_testfile + ".Y.arm");
  }

  // write to file
  w_all.save("w_yeast_" + algo + ".arm");

  // train_accuracies
  string train_acc_filename = "./yeast_train_acc_" + algo +".dat";
  std::ofstream output_file(train_acc_filename.c_str());
  std::ostream_iterator<double> output_iterator(output_file, "\n");
  std::copy(train_accuracies.begin(), train_accuracies.end(), output_iterator);

  string test_acc_filename = "./yeast_test_acc_" + algo +".dat";
  std::ofstream output_file2(test_acc_filename.c_str());
  std::ostream_iterator<double> output_iterator2(output_file2, "\n");
  std::copy(test_accuracies.begin(), test_accuracies.end(), output_iterator2);
  
  string runtime_filename = "./yeast_runtime_" + algo +".dat";
  std::ofstream output_file3(runtime_filename.c_str());
  std::ostream_iterator<double> output_iterator3(output_file3, "\n");
  std::copy(runtimes.begin(), runtimes.end(), output_iterator3);

  return 0;
}




// Specialized reading function for yeast data set, where the last 14 fields
// are binary class labels. Return imat Y (14 columns)
void read_data_yeast(string &filename, fmat &X, imat &Y, int *d_aug) {
  cout << "Reading yeast using armadillo load: " << filename << endl;
  fmat m;    // temporary variable

  clock_t init, final;  // record the file reading time.
  init = clock();
  int ncols_class = 14; // number of columns that are class labels.
  // number of classes. sometimes n_class != nclass_class because one 
  // column can represent numerical class
  int n_class = 14;     


  m.load(filename);
  int N = m.n_rows;
  int d = m.n_cols - ncols_class; // last column is class label, but we augment it.
  *d_aug = d + 1;   // augment first column to be all 1's

  X.resize(N, *d_aug);
  X.ones();
  X.cols(1, d) = m.cols(0, m.n_cols - ncols_class - 1);  // first row of X is 1's.
  Y = conv_to<imat>::from(m.cols(m.n_cols - ncols_class, m.n_cols-1));

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N 
    << ", d = " << d << ". Total time = " 
    << time_elapsed << " seconds. *****" << endl;

  cout << "X(0, 0...9) = " << X(0, span(0, 9)) << endl << endl;
  cout << "Y(0) = " << Y(0) << endl; 
}


