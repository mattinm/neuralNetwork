//
//  NeuralNetworkFunctions.h
//  
//
//  Created by Connor Bowley on 3/4/16.
//
//

#ifndef ____NeuralNetworkFunctions__
#define ____NeuralNetworkFunctions__

#include <stdio.h>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;

void convert1DArrayTo3DVector(const double *array, int width, int height, int depth, vector<vector<vector<double> > > &dest);

void convertColorMatToVector(cv::Mat m, vector<vector<vector<double> > > &dest);

void convolve(const vector<vector<vector<double> > > &source, const vector<vector<vector<vector<double> > > > &weights, vector<double> &bias, int stride, void(*activ)(vector<vector<vector<double> > >&), vector<vector<vector<double> > > &dest);

void fullyConnectToNewLayer(const vector<vector<vector<double> > > &, const vector<vector<vector<vector<double> > > > &, vector<double> &, void(*activ)(vector<vector<vector<double> > >&), vector<vector<vector<double> > > &);

double getMean(const vector<vector<vector<double> > > &source);

void maxPool(const vector<vector<vector<double> > > &source, int squarePoolSize, vector<vector<vector<double> > > &dest);

void meanSubtraction(vector<vector<vector<double> > >&source);

void meanSubtraction(const vector<vector<vector<double> > >&source, vector<vector<vector<double> > > &dest);

void numericalGradient(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest, void (*f2)(vector<vector<vector<double> > >, vector<vector<vector<double> > >));

void padZeros(const vector<vector<vector<double> > > &source, int numZeros, vector<vector<vector<double> > > &dest);

void printVector(const vector<vector<vector<double> > > &dest);

void printVector(const vector<vector<vector<vector<double> > > > &vect);

void randWeightFill(vector<vector<vector<vector<double> > > > &source);

void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth);

void softmax(vector<vector<vector<double> > > &source);

void softmax(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest);

void vectorClone(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest);

double vectorESum(const vector<vector<vector<double> > > &source);


#endif /* defined(____NeuralNetworkFunctions__) */
