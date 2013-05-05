//#include <glog/logging.h>   // Use Google logging library
#include <gflags/gflags.h>  // Use Google commandline library

#include "util.hpp"
#include "approx_solver.hpp"

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
//DEFINE_int32(Ntrain, 0, "number of training data instances.");
//DEFINE_int32(Ntest, 0, "number of test data instances.");
//DEFINE_int32(d, 0, "data dimension.");
DEFINE_bool(save_binary, false, "save Armadillo binary after loading the data.");
DEFINE_string(algo, "laplace", "laplace/delta");
DEFINE_int32(labelC1, -2, "class 1 label.");  // The rest will be class 2. Class 1 has Y = 1
DEFINE_int32(max_iter, 500, "maximum number of iterations in gradient descent.");


using namespace arma;

int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  // options
  string algo = FLAGS_algo;
  assert(algo == "laplace" || algo == "delta");
  cout << "running " << algo << endl;

  icolvec Ytrain, Ytest;  // single class
  //imat Ytrain, Ytest;       // multi class
  fmat Xtrain, Xtest;
  int d_aug, d_aug_test;
  read_data(FLAGS_trainfile, Xtrain, Ytrain, &d_aug, FLAGS_labelC1); // read training data
  read_data(FLAGS_testfile, Xtest, Ytest, &d_aug_test, FLAGS_labelC1); // read testing data
  assert(d_aug_test == d_aug);
  
  // compare the two Xtrain
  //cout << "diff Xtrain = " << as_scalar(accu(Xtrain - Xtrain2)) << endl;

  if (FLAGS_save_binary) {
    Xtrain.save(FLAGS_trainfile + ".X.arm");
    Ytrain.save(FLAGS_trainfile + ".Y.arm");
  }

  fcolvec w(d_aug);
  if (algo == "laplace") {
    wmap_grad_desc(Xtrain, Ytrain, w, FLAGS_s0, FLAGS_n_init, FLAGS_max_iter);
  }
  else if (algo == "delta") {
    delta_variational(Xtrain, Ytrain, w, FLAGS_s0, FLAGS_n_init, FLAGS_max_iter);
  }
  cout << "||w||^2 = " << as_scalar(sum(w.t() * w)) << endl;

  double training_accuracy = classification_accuracy(Xtrain, Ytrain, w);
  cout << "training accuracy = " << training_accuracy << endl;

  if (FLAGS_save_binary) {
    Xtest.save(FLAGS_testfile + ".X.arm");
    Ytest.save(FLAGS_testfile + ".Y.arm");
  }

  double test_accuracy = classification_accuracy(Xtest, Ytest, w);
  cout << "classification accuracy = " << test_accuracy << endl;

  return 0;
}

