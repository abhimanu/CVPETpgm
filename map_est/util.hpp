#ifndef PGM_util
#define PGM_util
#include <string>
#include <armadillo>
using std::string;
using namespace arma;

void read_data(string &filename, fmat &X, icolvec &Y, int* d_aug, int label_c1);

void read_data_fast(string &filename, fmat &X, icolvec &Y, int* d_aug, int label_c1);

double classification_accuracy(const fmat &Xtest, const icolvec &Ytest, 
    const fcolvec &w);

#endif
