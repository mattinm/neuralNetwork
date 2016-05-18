//
//  ConvNetCL.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNetCL__
#define ____ConvNetCL__

//defines for layers
#define ABSTRACT_LAYER -1;
#define CONV_LAYER 0;
#define MAX_POOL_LAYER 1;
#define ACTIV_LAYER 2;

//defines for ActivTypes
#define RELU 0;
#define LEAKY_RELU 1;

#include <vector>

typedef std::vector<std::vector<std::vector<double> > > imVector;

#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

class Net{
public: 	// static const members
	//static const int ABSTRACT_LAYER = -1;
	//static const int CONV_LAYER = 0;
private: 	// structs
	struct Layer{
		int layerType = ABSTRACT_LAYER;
	};

	struct ConvLayer : Layer{
		int layerType = CONV_LAYER;
		double* weights;
		double* biases;
		int numWeights;
		int numBiases;
		int numNeurons;
		int padding;
		int stride;
		int prevNeuronWidth;
		int paddedNeuronWidth;
		int prevNeuronHeight;
		int paddedNeuronDepth;
		int prevNeuronDepth;
		int maxSizeNeeded;
		int paddedNeuronSize;
	};

	struct MaxPoolLayer : Layer{
		int layerType = MAX_POOL_LAYER;
		int stride;
		int poolSize;
		int numNeurons;
		int prevWidth;
		int prevDepth;
	};

	struct ActivLayer : Layer{
		int layerType = ACTIV_LAYER;
		int activationType;
	};

private: 	// members
	//members dealing with layers
	std::vector<Layer> __layers;
	bool autoActiveLayer = true;
	std::vector<int> neuronSizes;
	int defaultActivType = 0;

	//data and related members
	std::vector<std::vector<imVector> > __data; // class<list of<imVectors> >
		//training
		std::vector<double> __trueVals; // parallel vector of true values for __data
		//running
		std::vector<std::vector<double> > __confidences; // image<list of confidences for each class<confidence> > 

	//OpenCL related members
public: 	// functions
	//functions dealing with layers
	bool addActivLayer();
	bool addActivLayer(int activationType);
	
private:	// functions

};

#endif /* defined(____ConvNetCL__) */