//============================================================================
// Name        : 10708code.cpp
// Authors      : abhimanu kumar, weid dai, jinliang wei
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include<random>

#include "armadillo"
#include <boost/random/normal_distribution.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

#define NUM_SAMPLES 1000
#define REGULARIZATION 10

using namespace arma;
using namespace std;

float getzSample(float mean, float var, float y){
	float z;
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	boost::normal_distribution<> nd(mean, var);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	float sample = var_nor();
	if(y==1.0)
		z = sample*((sample>0)? 1:0);
	else
		z = sample*((sample<=0)? 1:0); // since it's a truncated Normal it should be 1 and 0;
	return z;
}

fmat getZSample(fmat Mean, fmat Var, fmat Y){
	unsigned int rows = Y.n_rows;
	fmat Z(rows,1);
	for(int i=0;i<rows;i++)
		Z(i,0)=getzSample(Mean(i,0), Var(i,0), Y(i,0));
	return Z;
}

fmat getNormalSample(fmat Mean, fmat Var){
	unsigned int rows = Mean.n_rows;
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	fmat Z(rows,1);
	for(int i=0;i<rows;i++){
		boost::normal_distribution<> nd(Mean(i,0), Var(i,0));
		boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
		//optimize; same mean and var all the time
		Z(i,0)=var_nor();
	}
	return Z;
}

fmat getLogisticCoeffs(fmat X, fmat Y){
	unsigned int rows = X.n_rows;
	unsigned int cols = X.n_cols;
	fmat beta = zeros<fmat>(NUM_SAMPLES,rows);

	// intialize big-lamda
	fmat Lamda(rows,rows);
	Lamda.eye();						//n><n
	//initialize Z samples
	fmat Z = getZSample(zeros<fmat>(rows,1),ones<fmat>(rows,1),Y);	//n><1
	//prior variance
	fmat v(cols,cols);
	v.eye()*REGULARIZATION;	//p><p

	//sampling variables
	fmat V, B, L, T, S;
//	fmat H,W = ;

	//sampling starts here
	for(int i=0; i<NUM_SAMPLES; i++){
		V = inv(trans(X)*inv(Lamda)*X + v/(REGULARIZATION^2));	//optimize inverses p><p
		L = trans(chol(V));	// we need to get the lower triangulation thus the transpose P><p
		S=V*trans(X);					//p><n
		B=S*inv(Lamda)*Z;				//p><1
		for(int j=0; j<rows; j++){
			float z_old = Z(j,0);
			fmat x_j = reshape(conv_to<fmat>::from(X.row(j)),1,cols);
			fmat s_j = reshape(conv_to<fmat>::from(S.col(j)),cols,1);
			float h_j = conv_to<float>::from(x_j*s_j);
			float w_j = h_j/(Lamda(j,j)-h_j) ;
			float mean = conv_to<float>::from(x_j*B);
			mean = mean - w_j*(Z(j,0)-mean);
			float var =Lamda(j,j)*(w_j+1);
			Z(j,0) = getzSample(mean,var,Y(i,0));
			B = B + ((Z(j,0)-z_old)/Lamda(j,j))*s_j;
		}
		// draw beta vals
		T = getNormalSample(zeros<fmat>(rows,1),ones<fmat>(rows,1));
		beta.row(i) = B + L*T;
		cout<<"sample # "<<i<<" "<<norm(beta.row(i),2);
	}

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
	X=X.cols(0,4000);
	fmat beta = getLogisticCoeffs(X, Y);
	return 0;
}
