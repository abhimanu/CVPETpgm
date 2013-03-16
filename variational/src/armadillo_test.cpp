
#include <unistd.h>
#include <boost/program_options.hpp>
namespace boost_po = boost::program_options;
#include <iostream>
#include <armadillo>
#include <cmath>

#define COV_VAR 0.5

int main(int argc, char *argv[]){
  boost_po::options_description options("Allowed options");

  std::string infname;
  
  options.add_options()
    ("in", boost_po::value<std::string>(&infname)->required(), "input");

  boost_po::variables_map options_map;
  boost_po::store(boost_po::parse_command_line(argc, argv, options), options_map);
  boost_po::notify(options_map);

  arma::mat X;

  arma::colvec Y;
  X.load(infname, arma::csv_ascii);
  Y = X.col(X.n_cols - 1);
  X.shed_col(X.n_cols - 1);

  const int32_t num_samples = X.n_rows;
  const int32_t num_features = X.n_cols;

  std::cout << "data matrix loaded, size: " << num_samples << ", " << num_features << std::endl;
  
  arma::colvec mu(num_features); //gaussian mean
  arma::mat sigma(num_features, num_features); //gaussian covariance
  
  //double epsilon = 1;

  mu.randn();
  sigma = sigma.eye()*0.5;

  arma::mat sigma_inv = sigma.i();

  X.save(infname + ".out", arma::csv_ascii);
  arma::mat XpX = Y.t()*X*Y;
  arma::mat YpY = Y.t()*Y;
  YpY.print();
  XpX.save(infname + ".xpx.out", arma::csv_ascii);
  sigma.save(std::string("sigma.out"), arma::csv_ascii);
  sigma_inv.save(std::string("sigma_inv.out"), arma::csv_ascii);

}
