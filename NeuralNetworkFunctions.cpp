//
//  NeuralNetworkFunctions.cpp
//  
//
//  Created by Connor Bowley on 3/4/16.
//
//

#include "NeuralNetworkFunctions.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <limits>
#include <vector>
#include <iomanip>
#include <math.h>
#include <random>
#include <time.h>

#define GETMAX(x,y) (x > y) ? x: y

using namespace cv;
using namespace std;


/******************************
 *
 * Functions
 *
 ******************************/

void printVector(const vector<vector<vector<vector<double> > > > &vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		printVector(vect[n]);
		cout << endl;
	}
}

void printVector(const vector<vector<vector<double> > > &vect)
{
	for(int i=0;i < vect.size();i++) // rows
	{
		cout << "|";
		for(int j=0;j < vect[0].size();j++)
		{
			for(int k=0; k < vect[0][0].size(); k++)
			{
				cout << setw(4) << vect[i][j][k];
				if(k != vect[0][0].size()-1) cout << ",";
			}
			cout << "|";
		}
		cout << endl;
	}
}

void convert1DArrayTo3DVector(const double *array, int width, int height, int depth, vector<vector<vector<double> > > &dest)
{
	//resize dest vector
	dest.resize(width);
	for(int i=0; i < width; i++)
	{
		dest[i].resize(height);
		for(int j=0; j < height; j++)
		{
			dest[i][j].resize(depth);
		}
	}
	
	for(int i=0; i < width;i++)
	{
		for(int j=0; j < height; j++)
		{
			for(int k=0; k < depth; k++)
			{
				dest[i][j][k] = *array++;
			}
		}
	}
}


void convertColorMatToVector(Mat m, vector<vector<vector<double> > > &dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}
	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	
	for(int i=0; i< m.rows; i++)
	{
		for(int j=0; j< m.cols; j++)
		{
			Vec3b curPixel = m.at<Vec3b>(i,j);
			dest[i][j][0] = curPixel[0];
			dest[i][j][1] = curPixel[1];
			dest[i][j][2] = curPixel[2];
		}
	}
}


/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
void padZeros(const vector<vector<vector<double> > > &source, int numZeros, vector<vector<vector<double> > > &dest)
{
	int width2 = source.size() + 2*numZeros;
	int height2 = source[0].size() + 2*numZeros;
	int depth2 = source[0][0].size();
	//resize dest vector
	resize3DVector(dest,width2,height2,depth2);
	for(int i=numZeros; i< dest.size()-numZeros; i++) // rows
	{
		for(int j=numZeros; j< dest[0].size()-numZeros; j++) // cols
		{
			for(int k=0; k< dest[0][0].size(); k++) // depth
			{
				/*
				if(i < numZeros || i >= dest.size() - numZeros || j < numZeros || j >= dest[0].size() - numZeros)
				{
					dest[i][j][k] = 0;
				}
				else
				{*/
					//printf("%d %d %d\n",i,j,k);
					dest[i][j][k] = source[i-numZeros][j-numZeros][k];
				//}
			}
		}
	}
}

void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth)
{
	vect.resize(width);
	for(int i=0; i < width; i++)
	{
		vect[i].resize(height);
		for(int j=0; j < height; j++)
		{
			vect[i][j].resize(depth);
		}
	}
}


void maxPool(const vector<vector<vector<double> > > &source, int squarePoolSize, vector<vector<vector<double> > > &dest)
{
	if(source.size() % squarePoolSize != 0 || source[0].size() % squarePoolSize != 0)
	{
		printf("The number of rows and cols need to be evenly divisible by the pool size\n");
		throw "Invalid pool size";
	}
	int width2 = source.size()/squarePoolSize;
	int height2 = source[0].size()/squarePoolSize;
	int depth2 = source[0][0].size();
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	//printf("%d%d\n",out->rows,out->cols);
	int oX=0, oY=0;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size();i+=squarePoolSize) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size();j+=squarePoolSize) // cols
			{
				double maxVal = numeric_limits<double>::min();
				for(int s=0;s<squarePoolSize;s++)
				{
					for(int r=0; r< squarePoolSize; r++)
					{
						maxVal = GETMAX(source[i+s][j+r][k],maxVal);
					}
				}
				dest[oX][oY++][k] = maxVal;
			}
			oX++;
		}
	}
}


void convolve(const vector<vector<vector<double> > > &source, const vector<vector<vector<vector<double> > > > &weights, vector<double> &bias, int stride, void(*activ)(vector<vector<vector<double> > >&),vector<vector<vector<double> > > &dest)
{
	double fspatialSize = (source.size() - weights[0].size())/((double)stride) + 1;
	cout << fspatialSize <<endl;
	if(std::fmod(fspatialSize, 1.0) > .001)
	{
		cout << "The stride appears to be the wrong size" << endl;
		throw "Wrong stride size";
	}
	int numFilters = weights.size();
	int subsetWidth = weights[0].size();
	int subsetHeight = weights[0][0].size();
	// formula from http://cs231n.github.io/convolutional-networks/
	int depth2 = numFilters; // num of filters => depth 2
	int width2 =  (source.size() - weights[0].size())/stride + 1;
	int height2 = (source[0].size() - weights[0][0].size())/stride + 1;
	printf("(%lu  - %lu) / %d + 1\n",source.size(),weights[0].size(),stride);
	int oX, oY;
	//resize dest vector
	resize3DVector(dest,width2,height2,depth2);
	double sum;
	for(int f=0; f<numFilters; f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i < source.size()-1; i+= stride) // row in source
		{
			oY = 0;
			for(int j=0; j < source[i].size()-1;j+=stride) // col in source
			{
				//now we go into the stride subset
				sum = 0;
				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < source[i][j].size(); k++) // depth in source
						{
							sum += source[i+s][j+r][k] * weights[f][s][r][k];
							//sum += source.at(i+s).at(j+r).at(k) * weights.at(f).at(s).at(r).at(k);
						}
					}
				}
				
				// add bias
				sum += bias[f];
				// add into out[i%stride, j%stride, f]
				dest[oX][oY][f] = sum;
				oY++;
			}
			oX++;
		}
	}
	
	(*activ)(dest);
}

void fullyConnectToNewLayer(const vector<vector<vector<double> > > &source, const vector<vector<vector<vector<double> > > > &weights, vector<double> &bias, void(*activ)(vector<vector<vector<double> > >&),  vector<vector<vector<double> > > &dest)
{
	//int numFilters = weights.size();
	int width2 = weights[0].size();
	int height2 = weights[0][0].size();
	int depth2 = weights.size(); // numFilters
	resize3DVector(dest, width2,height2,depth2);
	int sum;
	for(int ii = 0; ii < dest.size(); ii++)
	{
		for(int jj = 0; jj < dest[0].size(); jj++)
		{
			for(int f=0; f < dest[0][0].size(); f++)
			{
				sum = 0;
				for(int i=0; i< source.size(); i++)
				{
					for(int j=0; j< source[0].size(); j++)
					{
						for(int k=0; k< source[0][0].size(); k++)
						{
							for (int kk=0;kk<weights[0][0][0].size(); kk++)
							{
								sum+=source[i][j][k] * weights[f][ii][jj][kk];
							}
						}
					}
				}
				dest[ii][jj][f] = sum + bias[f];
			}
		}
	}
	//(*activ)(dest);
	
}

double getMean(const vector<vector<vector<double> > > &source)
{
	double sum = 0.0;
	for (int i=0; i < source.size(); i++)
	{
		for(int j=0; j < source[0].size(); j++)
		{
			for(int k=0; k < source[0][0].size(); k++)
			{
				sum+=source[i][j][k];
			}
		}
	}
	return sum / (source.size() * source[0].size() * source[0][0].size());
}

void meanSubtraction(const vector<vector<vector<double> > >&source, vector<vector<vector<double> > > &dest)
{
	resize3DVector(dest,source.size(),source[0].size(),source[0][0].size());
	double mean = getMean(source);
	for (int i=0; i < source.size(); i++)
	{
		for(int j=0; j < source[0].size(); j++)
		{
			for(int k=0; k < source[0][0].size(); k++)
			{
				dest[i][j][k] = source[i][j][k] - mean;
			}
		}
	}
}

void meanSubtraction(vector<vector<vector<double> > >&source)
{
	double mean = getMean(source);
	for (int i=0; i < source.size(); i++)
	{
		for(int j=0; j < source[0].size(); j++)
		{
			for(int k=0; k < source[0][0].size(); k++)
			{
				 source[i][j][k] -= mean;
			}
		}
	}
}

double vectorESum(const vector<vector<vector<double> > > &source)
{
	double sum = 0;
	for(int i=0; i < source.size(); i++)
	{
		for(int j=0; j< source[0].size(); j++)
		{
			for(int k=0; k< source[0][0].size(); k++)
			{
				sum += exp(source[i][j][k]);
			}
		}
	}
	return sum;
}

void softmax(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest)
{
	resize3DVector(dest,source.size(),source[0].size(),source[0][0].size());
	meanSubtraction(source,dest);
	double denom = vectorESum(dest);
	for(int i=0; i< dest.size(); i++)
	{
		for(int j=0; j< dest[0].size(); j++)
		{
			for(int k=0; k< dest[0][0].size(); k++)
			{
				dest[i][j][k] = exp(dest[i][j][k])/denom;
			}
		}
	}
}

void softmax(vector<vector<vector<double> > > &source)
{
	meanSubtraction(source);
	double denom = vectorESum(source);
	for(int i=0; i< source.size(); i++)
	{
		for(int j=0; j< source[0].size(); j++)
		{
			for(int k=0; k< source[0][0].size(); k++)
			{
				source[i][j][k] = exp(source[i][j][k])/denom;
			}
		}
	}
}

void randWeightFill(vector<vector<vector<vector<double> > > > &source)
{
	default_random_engine gen(time(0));
	uniform_real_distribution<double> distr(0.0001,1.0);
	for(int f = 0;f<source.size(); f++)
	{
		for(int i=0; i< source[0].size(); i++)
		{
			for(int j=0; j< source[0][0].size(); j++)
			{
				for(int k=0; k< source[0][0][0].size(); k++)
				{
					source[f][i][j][k] = distr(gen);
				}
			}
		}
	}
}

void vectorClone(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest)
{
	resize3DVector(dest,source.size(),source[0].size(),source[0][0].size());
	for(int i=0; i< dest.size(); i++)
	{
		for(int j=0; j< dest[0].size(); j++)
		{
			for(int k=0; k< dest[0][0].size(); k++)
			{
				dest[i][j][k] = source[i][j][k];
			}
		}
	}
	
}



/*
 * Note f1 and f2 are the same function, just f1 takes one param and modifies it and f2 takes 2 params, a source and a dest so the source isn't modified.
 */
void numericalGradient(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest, void (*f2)(vector<vector<vector<double> > >, vector<vector<vector<double> > >))
{
	vectorClone(source, dest);
	vector<vector<vector<double> > > fx, fxh;
	(*f2)(source,fx); // original f(x)
	double h = .00001;
	for(int i=0; i< source.size(); i++)
	{
		for(int j=0; j< source[0].size(); j++)
		{
			for(int k=0; k< source[0][0].size(); k++)
			{
				dest[i][j][k] += h;
			}
		}
	}
	(*f2)(dest,fxh); // f(x+h)
	for(int i=0; i< source.size(); i++)
	{
		for(int j=0; j< source[0].size(); j++)
		{
			for(int k=0; k< source[0][0].size(); k++)
			{
				dest[i][j][k] = (fxh[i][j][k] - fx[i][j][k]) / h;
			}
		}
	}
}

int origMain(int argc, char**argv)
{
	vector<vector<vector<double> > > testVector;//(1,vector <vector <double> >(1, vector<double>(1, 0.0)));
	Mat M(4,4,CV_8UC3, Scalar(100, 150, 200));
	M += 5*Mat::eye(M.rows,M.cols, M.type());
	cout << "M = " << endl << M << "\n\n";
	
	
	double testArray[] = {
		0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
		2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
		1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
		1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
		2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};
	
	convert1DArrayTo3DVector(testArray,5,5,3,testVector);
	
	/* // testing mean subtraction
	 double meanArray[] = {20, 30, 40};
	 vector<vector<vector<double> > > meanVector, mean2Vector;
	 convert1DArrayTo3DVector(meanArray,1,1,3,meanVector);
	 meanSubtraction(meanVector, mean2Vector);
	 meanSubtraction(meanVector);
	 printVector(meanVector);
	 printVector(mean2Vector);
	 */
	
	vector<vector<vector<double> > > padVector, pad2;
	//cout << "at pad zeros" << endl;
	padZeros(testVector,1,padVector);
	padZeros(testVector,2,pad2);
	
	printVector(padVector);
	cout << endl;
	printVector(pad2);
	
	vector<vector<vector<double> > > convVector;
	
	vector<vector<vector<vector<double> > > > filters(2);
	
	double filter0[] = {
		1, 0, 1,	-1, 0, 0,	-1, 1, 0,
		0, 1, 0,	 1, 0, 0,	 1,-1,-1,
		-1, 0, 0,	 1, 1,-1,	-1, 1, 0
	};
	double filter1[] = {
		0, 1,-1,	-1,-1,-1,	-1, 0,-1,
		1, 1,-1,	 0, 0,-1,	 0, 1,-1,
		1,-1, 1,	 1, 1,-1,	-1, 0, 1
	};
	double bias[] = {1,0};
	
	
	convert1DArrayTo3DVector(filter0,3,3,3,filters[0]);
	convert1DArrayTo3DVector(filter1,3,3,3,filters[1]);
	
	//convolve(padVector,filters,bias,2,convVector);
	//printVector(convVector);
	/*solution to conv
	 | 1, 1| 2, 3| 6, 4|
	 | 3,-4| 3,-4| 4,-1|
	 | 4,-5| 4,-3| 5,-1|
	 */
	
	return 0;
}

int exclusiveOrNN(int argc, char**argv)
{
	int numFilters = 2;
	//4 dim vector for input. [imageNum][pixX][pixY][pixZ (RGB)]
	vector<vector<vector<vector<double> > > > input(4);
	vector<vector<vector<vector<double> > > > layers(2);
	for(int n = 0; n < input.size(); n++)
	{
		resize3DVector(input[n],1,1,2);
	}
	input[0][0][0][0] = 0;
	input[0][0][0][1] = 0;
	
	input[1][0][0][0] = 0;
	input[1][0][0][1] = 1;
	
	input[2][0][0][0] = 1;
	input[2][0][0][1] = 0;
	
	input[3][0][0][0] = 1;
	input[3][0][0][1] = 1;
	
	vector<double> output(4);
	output[0] = 0;
	output[1] = 1;
	output[2] = 1;
	output[3] = 0;
	
	vector<vector<double> > bias(numFilters);
	bias[0].resize(2);
	bias[1].resize(2);
	
	bias[0][0] = 0;
	bias[0][1] = 1;
	bias[1][0] = 0;
	bias[1][1] = 1;
	
	//5 dim vector for weights [hiddenLayer#][class/filter#][weight][weight][weight]
	vector<vector<vector<vector<vector<double> > > > >weights(numFilters);
	weights[0].resize(1);
	weights[1].resize(1);
	resize3DVector(weights[0][0],3,2,1);
	resize3DVector(weights[1][0],1,1,3);
	randWeightFill(weights[0]);
	randWeightFill(weights[1]);
	cout << "weights[0]" << endl;
	printVector(weights[0]);
	cout << "weights[1]" << endl;
	printVector(weights[1]);
	
	fullyConnectToNewLayer(input[3],weights[0],bias[0],softmax,layers[0]);
	
	printVector(layers[0]);
	
	return 0;
}

int main(int argc, char** argv)
{
	origMain(argc,argv);
}













