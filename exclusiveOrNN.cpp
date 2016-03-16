//
//  exclusiveOrNN.cpp
//  
//
//  Created by Connor Bowley on 3/11/16.
//
//

#include <stdio.h>
#include <iostream>
#include <vector>
#include "NeuralNetworkFunctions.h"

using namespace std;

int main(void)
{
	vector<vector<vector<vector<double> > > > input(4);
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
	
	vector<vector<vector<vector<vector<double> > > > >weights(2);
	weights[0].resize(1);
	weights[1].resize(1);
	resize3DVector(weights[0][0],1,1,6);
	resize3DVector(weights[1][0],1,1,3);
	randWeightFill(weights[0]);
	randWeightFill(weights[1]);
	cout << "weights[0]" << endl;
	//printVector(weights[0]);
	cout << "weights[1]" << endl;
}
