#ifndef PGM_APPROX_SOLVER
#define PGM_APPROX_SOLVER

#include <armadillo>
using namespace arma;

void wmap_grad_desc(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter);
void delta_variational(const fmat &X, const icolvec& Y, fcolvec& w, 
    double s0, int n_init, int max_iter);

#include <armadillo> // Matrix operation library

#endif // PGM_APPROX_SOLVER
