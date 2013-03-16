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


using namespace arma;
using namespace std;

#define NUM_SAMPLES 1000
#define REGULARIZATION 5

/**
 * filename needs to be space delimited, not csv, and need prior knowledge
 * of N and d (augmented).
 */

float getNormalSample(float mean, float var){
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	boost::normal_distribution<> nd(0, 1);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	return var_nor();
}

float getUniformZeroOne(){
	boost::mt19937 rng;
	static boost::uniform_01<boost::mt19937> zeroone(rng);
	return zeroone();;
}

bool rightmostInterva(float u, float lamda){
	float z=1;
	float x = exp(0.5*lamda);
	int j =0;
	while (true){
		j = j+1;
		z=z-(pow(j+1,2)*pow(x,pow(j+1,2)-1));
		if(z>u)
			return true;
		j=j+1;
		z=z+(pow(j+1,2)*pow(x,pow(j+1,2)-1));
		if(z<u)
			return false;
	}
	return false;
}

bool leftmostInterval(float u, float lamda){
	float h = 0.5*log(2) + 2.5*(log(M_PI)-log(lamda)) - pow(M_PI,2)/(2*lamda) + 0.5*lamda;
	float lU = log(u);
	float z=1;
	float x = exp(-pow(M_PI,2)/(2*lamda));
	float k = lamda/pow(M_PI,2);
	int j=0;
	while(true){
		j=j+1;
		z = z - k*pow(x,pow(j,2)-1);
		if(h+log(z)>lU)
			return true;
		j=j+1;
		z = z + pow(j+1,2)*pow(x,pow(j+1,2)-1);
		if(h+log(z)<lU)
			return false;
	}
}

float sampleFromKSmodified(float r){
	float lamda;
	bool flag = false;
	while(!flag){
		float y=getNormalSample(0,1);
		y=pow(y,2);
		y=1+((y-sqrt(y*(4*r+y)))/(2*r));
		float u = getUniformZeroOne();
		if(u<=1/(1+y))
			lamda = r/y;
		else
			lamda=r*y;
		// lamda being drawn from GIG(0.5,1,^2);

		if(lamda>4.0/3)
			flag = rightmostInterva(u,lamda);
		else
			flag = leftmostInterval(u,lamda);
	}
}

void read_data_fast(string &filename, fmat &X, fmat &Y, int N, int d_aug) {
  clock_t init, final;  // record the file reading time.
  init = clock();

  X.resize(N, d_aug);
  Y.resize(N,1);

  ifstream indata;
  indata.open(filename.c_str()); // opens the file
  if(!indata) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  for (int j = 1; j < d_aug; j++) { // read first line
    // read attributes
    float attr;
    indata >> attr;
    X(0, j) = attr;
  }
  int label;
  indata >> label;  // reading class label
  Y(0,0) = label;

  for(int i = 1; !indata.eof() && i < N; i++) {
    for (int j = 1; j < d_aug; j++) { // read first line
      // read attributes
      int attr;
      indata >> attr;
      X(i, j) = attr;
    }
    indata >> label;  // reading class label
    Y(i,0) = label;
  }

//  Y = (Y + 1)/2;  // change from {-1,1} to {0,1}.

  final = clock() - init;
  int time_elapsed = (double)final / ((double)CLOCKS_PER_SEC);

  cout << "***** Done reading " << filename << ". N = " << N
    << ", d = " << d_aug << ". Total time = "
    << time_elapsed << " seconds. *****" << endl;
}

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

fmat getNormalMatSample(fmat Mean, fmat Var){
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
	cout<<"rows, cols: "<<rows<<", "<<cols<<endl;
	fmat beta = zeros<fmat>(NUM_SAMPLES,cols);

	// intialize big-lamda
	fmat Lamda(rows,rows);
	Lamda.eye();						//n><n
	//initialize Z samples
	fmat Z = getZSample(zeros<fmat>(rows,1),ones<fmat>(rows,1),Y);	//n><1
	//prior variance
	fmat v(cols,cols);
	v.eye();	//p><p
	v=v*REGULARIZATION;
	//sampling variables
	fmat V, B, L, T, S;
//	fmat H,W = ;

	//sampling starts here
	fmat temp = trans(X)*inv(Lamda)*X;
	cout<<"temp rows and cols "<<temp.n_rows<<", "<<temp.n_cols<<endl;
	cout<<"REGULARIZATION*4"<<REGULARIZATION*4<<endl;
	cout<<"det(trans(X)*inv(Lamda)*X) "<<det(trans(X)*inv(Lamda)*X)<<endl;
	cout<<"det(v/(pow(REGULARIZATION,2))) "<<det(v/(pow(REGULARIZATION,2)))<<endl;
	cout<<"det(v/(pow(REGULARIZATION,2)) + v/(pow(REGULARIZATION,2))) "<<det(v/(pow(REGULARIZATION,2))  + v/(pow(REGULARIZATION,2)))<<endl;
	cout<<"det(trans(X)*inv(Lamda)*X + v/(pow(REGULARIZATION,2))) "<<det(trans(X)*inv(Lamda)*X + v/(pow(REGULARIZATION,2)))<<endl;
	for(int i=0; i<NUM_SAMPLES; i++){
		cout<<"iter "<<i<<endl;
//		cout<<"v/(REGULARIZATION^2) "<<endl<<v/(REGULARIZATION^2)<<endl;
		fmat intermediateMat = trans(X)*inv(Lamda)*X + v/(pow(REGULARIZATION,2));
		cout<<"det(intermediateMat) "<<det(intermediateMat)<<endl;
		V = inv(intermediateMat);	//optimize inverses p><p
		cout<<"det(V) "<<det(V)<<endl;
		L = trans(chol(V));	// we need to get the lower triangulation thus the transpose P><p
		S=V*trans(X);					//p><n
//		cout<<"==HELLO1=="<<endl;
		B=S*inv(Lamda)*Z;				//p><1
//		cout<<"==HELLO2=="<<endl;
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
		T = getNormalMatSample(zeros<fmat>(cols,1),ones<fmat>(cols,1));
//		cout<<"==HELLO3=="<<endl;
		beta.row(i) = trans(B + L*T);
//		cout<<"==HELLO4=="<<endl;
		cout<<"SAMPLE # "<<i<<" "<<norm(beta.row(i),2)<<endl;
		// new values for mixing variances
		for(int j=0; j<rows; j++){
			float r_j = Z(j) - accu(X.row(j)%beta.row(j));
			Lamda(j,j) = sampleFromKSmodified(r_j);
		}
	}

	return beta;
}


fmat throwAwayZeroRows(fmat X, fmat Y){
	fmat rowSum = sum(X,1);
	int deletedRows = 0;
	for(int i=0; i<rowSum.n_elem; i++){
		int currentIndex = i-deletedRows;
		if(rowSum(i,0)==0){
//			cout<<"sum: "<<rowSum(currentIndex,0)<<", deleted row index "<<i<<endl;
			X.shed_row(currentIndex);
			Y.shed_row(currentIndex);
			deletedRows+=1;
		}
//		else{
//			cout<<"current Index nonzero: "<<currentIndex<<endl<<find(X.row(currentIndex));
//		}
	}
	cout<<"total deleted rows "<<deletedRows<<", total Rows remaining "<<X.n_rows<<endl;

//	cout<<sum(X,1)<<endl;
	fmat colSum = sum(X,0);
	cout<<"colSum nonZero "<<endl<<(conv_to<umat>::from(find(colSum>0))).n_elem<<endl;
	cout<<"colSum"<<endl<<colSum<<endl;
	cout<<"colSum(1) "<<colSum(1);//<<" colSum(1,0) "<<colSum(1,0);
	cout<<" colSum(0,1) "<<colSum(0,1)<<endl;
	int deletedCols=0;
	for(int i=0; i<colSum.n_elem; i++){
		int currentIndex = i-deletedCols;
		if(colSum(i)==0){
			cout<<"sum: "<<colSum(i)<<", deleted col index "<<i<<endl;
			X.shed_col(currentIndex);
			deletedCols+=1;
		}
	}
	cout<<"total deleted cols "<<deletedCols<<", total Cols remaining "<<X.n_cols<<endl;
	return join_rows(X,Y);
}

int main() {
	fmat X;
	string filename = "farm_ads.dat";//"farm_ads.csv";
	fmat Y;
	read_data_fast(filename, X, Y, 4143, 54878);
//	Y=conv_to<fmat>::from(Y);

//	X.load(filename, csv_ascii);
	unsigned int num_rows = X.n_rows;
	unsigned int num_cols = X.n_cols;//X.n_cols-1;	//actual number of columns including target since there is extra comma at the end
//	Y.resize(num_rows,1);
//	fmat Y=X.col(num_cols-1);	//get the target (Y)
	// if Y contains +1 and -1 we will have to convert it to +1 and 0 ?
//	X=X.cols(0,num_cols-2);	//get the Xs
//	num_cols -=1;	// total number of features
	cout<<num_rows<<endl<<num_cols<<endl<<X(0,num_cols-1)<<endl<<X(0,num_cols-2)<<endl;
	cout<<Y(0,0)<<Y(1,0)<<Y(2,0)<<Y(3,0)<<Y(4,0)<<Y(5,0)<<endl;
	cout<<"accu(X.col(0))"<<accu(X.col(0))<<endl;
	X=X.cols(0,10);
	fmat combinedMat=throwAwayZeroRows(X, Y);
	X = combinedMat.cols(0,combinedMat.n_cols-2);
	Y = combinedMat.col(combinedMat.n_cols-1);
//	cout<<Y<<endl;//<<X.n_cols<<endl<<sum(X,1)<<endl;
	fmat beta = getLogisticCoeffs(X, Y);
	return 0;
}
