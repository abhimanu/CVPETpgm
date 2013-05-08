#ifndef PGM_APPROX_SOLVER
#define PGM_APPROX_SOLVER

#include <armadillo>
#include "util.hpp"
using namespace arma;

void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter, int *runtime, const fmat &Xtest, const icolvec &Ytest);
void delta_variational(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter, int *runtime, const fmat &Xtest, const icolvec &Ytest);

#include <armadillo> // Matrix operation library

#endif // PGM_APPROX_SOLVER
