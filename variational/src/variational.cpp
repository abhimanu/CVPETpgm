
/**
 * Variational approach for Bayesian Logistic Gaussian
 * Algorithm follows Jordan's paper 1996.
 */


#include <unistd.h>
#include <boost/program_options.hpp>
namespace boost_po = boost::program_options;
#include <iostream>
#include <armadillo>
#include <cmath>
#include <ctime>

#define START_TIMER(T) T = clock();
#define END_TIMER(T, P) T = clock() - P;
#define TO_SEC(T) ((float) T/CLOCKS_PER_SEC)

#define COV_FAC 0.5
#define CONVERGE_THRE 0.000001


void convert_binary(arma::icolvec &Y){
  int32_t num_row = Y.n_rows;
  int32_t i;
  for(i = 0; i < num_row; i++){
    if(Y[i] > 1) Y[i] = (Y[i]%2) ? 1 : 0;
    if(Y[i] == -1) Y[i] = 0;
  }
}

inline double sigmoid(double x){
  return 1/(1 + exp(-x));
}

inline double lambda(double x){
  return (0.5 - sigmoid(x))/2*x;
}

arma::mat get_post_sigma_inv(double epsilon, arma::colvec &x, arma::mat &sigma_inv){

  return sigma_inv + 2*abs(lambda(epsilon))*(x*x.t());
}

arma::colvec get_post_mu(arma::colvec &x, arma::mat &sigma_inv, arma::mat &post_sigma, arma::colvec &mu, double s){
  
  return post_sigma*(sigma_inv*mu + (s - 1.0/2)*x);
}

double get_epsilon_sqr(arma::colvec &x, arma::mat &post_sigma, arma::colvec &post_mu){
  arma::mat v = x.t()*post_sigma*x;
  arma::mat xmu = x.t()*post_mu;
  
  return as_scalar(v) + as_scalar(xmu)*as_scalar(xmu);
}

int classify(double epsilon, arma::mat &sigma, arma::mat &sigma_inv, arma::colvec &x, arma::colvec &mu){
  arma::mat sigma_t_inv = get_post_sigma_inv(epsilon, x, sigma_inv);
  arma::mat sigma_t = sigma_t_inv.i();
  arma::colvec mu_t; 

  int s = 0;
  mu_t = get_post_mu(x, sigma_inv, sigma_t, mu, s);
  
  double log_0 = log(sigmoid(epsilon)) - epsilon/2 - lambda(epsilon)*epsilon*epsilon - 0.5*as_scalar(mu.t()*sigma_inv*mu) + 0.5*as_scalar(mu_t.t()*sigma_t.t()*mu_t) + 0.5*log(det(sigma_t)/det(sigma));

  s = 1;
  mu_t = get_post_mu(x, sigma_inv, sigma_t, mu, s);
  
  double log_1 = log(sigmoid(epsilon)) - epsilon/2 - lambda(epsilon)*epsilon*epsilon - 0.5*as_scalar(mu.t()*sigma_inv*mu) + 0.5*as_scalar(mu_t.t()*sigma_t.t()*mu_t) + 0.5*log(det(sigma_t)/det(sigma));

  if(log_1 > log_0) return 1;
  else return 0;

}

double classification_error(double epsilon, arma::mat &sigma, arma::mat &sigma_inv, arma::colvec &mu, arma::mat &X, arma::icolvec &Y){
  int32_t n_samples = Y.n_rows;
  int32_t i;
  int32_t n_errors = 0;
  for(i = 0; i < n_samples; i++){
    arma::colvec x = X.row(i).t();
    int c = classify(epsilon, sigma, sigma_inv, x, mu);
    if(c != Y[i]) ++n_errors;
  }
  return ((double) n_errors)/n_samples;
}

int main(int argc, char *argv[]){
  boost_po::options_description options("Allowed options");

  std::string trainfname;
  std::string testfname;
  
  options.add_options()
    ("train", boost_po::value<std::string>(&trainfname)->required(), "train")
    ("test", boost_po::value<std::string>(&testfname)->required(), "test");
  boost_po::variables_map options_map;
  boost_po::store(boost_po::parse_command_line(argc, argv, options), options_map);
  boost_po::notify(options_map);

  arma::mat X;
  arma::icolvec Y;
  
  clock_t t;
  std::cout << "start loading " << trainfname << std::endl;
  START_TIMER(t);
  X.load(trainfname, arma::csv_ascii);
  END_TIMER(t, t);

  std::cout << "data matrix loaded, size: " << X.n_rows << ", " << X.n_cols << " time = " << TO_SEC(t) << " sec" << std::endl;

  Y = arma::conv_to<arma::icolvec>::from(X.col(X.n_cols - 1));
  X.shed_col(X.n_cols - 1);

  convert_binary(Y);

  const int32_t num_samples = X.n_rows;
  const int32_t num_features = X.n_cols;

  arma::colvec mu(num_features); //gaussian mean
  arma::mat sigma(num_features, num_features); //gaussian covariance  
  double epsilon = 2;
  double delta = 1;
  
  //initialize gaussian parameters
  mu.zeros();
  sigma = sigma.eye()*COV_FAC;

  std::cout << "start inverting sigma" << std::endl;
  
  START_TIMER(t);
  arma::mat sigma_inv = sigma.i();
  END_TIMER(t, t);
  std::cout << "finished inverting, time = " << TO_SEC(t) << " secs" << std::endl;

  clock_t ts;
  std::cout << "starts training" << std::endl;
  START_TIMER(ts);
  while(abs(delta) > CONVERGE_THRE){
    arma::mat post_sigma_inv;
    arma::mat post_sigma;
    arma::colvec post_mu;
    double epsilon_sqr;
    double post_epsilon;
    int32_t i;

    for(i = 0; i < num_samples; i++){
      arma::colvec x = X.row(i).t();
      post_sigma_inv = get_post_sigma_inv(epsilon, x, sigma_inv);
  
      post_sigma = post_sigma_inv.i();
      
      post_mu = get_post_mu(x, sigma_inv, post_sigma, mu, Y(i));
      
      epsilon_sqr = get_epsilon_sqr(x, post_sigma, post_mu);
      post_epsilon = sqrt(epsilon_sqr);
      
      sigma = post_sigma;
      sigma_inv = post_sigma_inv;
      mu = post_mu;

      delta = post_epsilon - epsilon;
      epsilon = post_epsilon;
      END_TIMER(t, ts);
      std::cout << "epsilon = " << epsilon << " delta = " << delta << " time = " << TO_SEC(t) << " sec" << std::endl;
    }
  }
  
  END_TIMER(t, ts);
  std::cout << "time = " << TO_SEC(t) << " sec" << std::endl;
  std::cout << " training done" << std::endl;
  std::cout << "epsilon = " << epsilon << std::endl;

  arma::mat X_test;
  arma::icolvec Y_test;

  std::cout << "start loading test matrix" << trainfname << std::endl;
  START_TIMER(t);
  X_test.load(testfname, arma::csv_ascii);
  END_TIMER(t, t);

  std::cout << "data matrix loaded, size: " << X_test.n_rows << ", " << X_test.n_cols << " time = " << TO_SEC(t) << " sec" << std::endl;

  Y_test = arma::conv_to<arma::icolvec>::from(X_test.col(X_test.n_cols - 1));
  X_test.shed_col(X_test.n_cols - 1);

  double test_err = classification_error(epsilon, sigma, sigma_inv, mu, X_test, Y_test);
  std::cout << "test error = " << test_err << std::endl;

  double train_err = classification_error(epsilon, sigma, sigma_inv, mu, X, Y);
  std::cout << "train error = " << train_err << std::endl;


  sigma.save(trainfname + ".sigma.out", arma::csv_ascii);
  mu.save(trainfname + ".mu.out", arma::csv_ascii);
  X.save(trainfname + ".out", arma::csv_ascii);
  
}
