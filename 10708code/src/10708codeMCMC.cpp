//============================================================================
// Name        : 10708code.cpp
// Authors      : abhimanu kumar, weid dai, jinliang wei
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "armadillo"
#include <boost/random/normal_distribution.hpp>

using namespace arma;
using namespace std;

imat getZSample(fmat Mean, fmat Var, fmat Y){
	unsigned int rows = Y.n_rows();
	imat Z(rows,1);
	for(int i=0;i<rows;i++){
		std::tr1::normal_distribution<double> normal(Mean(i,0), Var(i,0));
		sample = normal(eng);
		if(Y(i,0)==1)
			Z(i,0) = (sample>0)? 1:-1;
		else
			Z(i,0) = (sample<=0)? 1:-1;
	}
	return Z;
}

fmat getLogisticCoeffs(fmat X, fmat Y){
	unsigned int rows = X.n_rows;
	unsigned int cols = X.n_cols;
	fmat beta = zeros<fmat>(rows,1);

	// intialize big-lamda
	fmat lamda(rows,rows);
	lamda.eye();

	imat Z = getZSample(Mean,Var,Y);

	return beta;
}


int main() {
	fmat X;
	string filename = "farm_ads.csv";
	X.load(filename, csv_ascii);
	unsigned int num_rows = X.n_rows;
	unsigned int num_cols = X.n_cols-1;	//actual number of columns including target since there is extra comma at the end
	fmat Y=X.col(num_cols-1);	//get the target (Y)
	// if Y contains +1 and -1 we will have to convert it to +1 and 0 ?
	X=X.cols(0,num_cols-2);	//get the Xs
	num_cols -=1;	// total number of features
	cout<<num_rows<<endl<<num_cols<<endl<<X(0,num_cols-1)<<endl<<X(0,num_cols-2)<<endl;
	cout<<Y(0,0)<<Y(1,0)<<Y(2,0)<<Y(3,0)<<Y(4,0)<<Y(5,0)<<endl;
	fmat beta = getLogisticCoeffs(X, Y);
	return 0;
}
