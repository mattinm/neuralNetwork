/*******************************************************
*
* ConvNetCL is an open source Convolutional Neural Network (CNN) implementation using C++ and OpenCL. It trains the CNNs
* using backpropagation with Nesterov Momementum and L2 regularization
*
* It's main purpose is to make a CNN that can classify images. It can save and load CNNs to a custom file format.
* Said format can be customized by hand if needed (rarely needed though).
*
* The architecture for a CNN you want to train can be specified using a NetConfig file, which can check to make sure
* the dimensions you think it will have after certain layers match up with what the software says they will be.
* An explanation and example of the format, called CNN_config_example.txt, can be found in the root directory of the repo.
*
*******************************************************/

#include "ConvNetCL.h"
#include "ConvNetCommon.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>


// #define _DEBUG 0 //uncomment for some print statements to help with debugging

// 	includes brought in from ConvNetCL.h
//
// #include <vector>	
// #include <string> 
// #include <time.h>
// #include <thread>
// 
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
//
// #ifdef __APPLE__
//  	#include "OpenCL/opencl.h"
// #else
//  	#include "CL/cl.h"
// #endif
//typedef std::vector<std::vector<std::vector<double> > > imVector; // typedef pulled from the .h file

using namespace std;
using namespace cv;
using namespace convnet;

typedef unsigned int uint;

/*****************************************
 * Constructors and Destructors and inits
 *
 * NOTE ON CONSTRUCTORS AND ASSIGNMENTS:
 * Using the copy constructor or = will copy
 * the structure and weights of the CNN, but 
 * it will not copy the training or test data 
 *
 *****************************************/

Net::Net()
{

}

// See NOTE ON CONSTRUCTORS AND ASSIGNMENTS above
Net::Net(const Net &other) // Copy constructor
{
	// printf("copy ------------------\n");
	if(this != &other)
	{
		// printf("in copy-----------\n");
		//clean up any currently used OpenCL stuff besides the context
		// destroy();
	
		this->__inited = true;
		//hyperparameters
		this->__learningRate = other.__learningRate;
		this->__RELU_CAP = other.__RELU_CAP;
		this->__LEAKY_RELU_CONST = other.__LEAKY_RELU_CONST;
		this->__l2Lambda = other.__l2Lambda;
		this->__MOMENT_CONST = other.__MOMENT_CONST;
		this->__MAX_NORM_CAP = other.__MAX_NORM_CAP;

		//Copy all layers
		this->copyLayers(other);
		//neuronSizes set by copyLayers
		//neuronDims set by copyLayers
		this->__autoActivLayer = other.__autoActivLayer;
		this->__maxNeuronSize = other.__maxNeuronSize;
		this->__defaultActivType = other.__defaultActivType;
		this->__maxWeightSize = other.__maxWeightSize;

		this->__isFinalized = false;
		this->__errorLog = "";

		//data and related members
		//__numClasses set in addTrainingData
		// this->__classes = other.__classes;
		this->__useMomentum = other.__useMomentum;
		this->__stuffBuilt = false;

		this->__isTraining = false;

		this->__mean = other.__mean;
		this->__stddev = other.__stddev;
		this->__trainingSize = other.__trainingSize;

		this->initOpenCL();
	}
	// printf("END copy --- \n");
}

Net::Net(const char* filename)
{
	bool success = load(filename);
	if(!success)
		exit(1);
}

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	init(inputWidth, inputHeight, inputDepth);
}

void Net::destroy()
{
	// printf("start destroy\n");
	// printf("%d\n", __inited);
	if(__inited)
	{

		// printf("inited\n");
		Layer *point;
		for(int i=0; i< __layers.size(); i++)
		{
			point = __layers[i];
			if(point->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)point;
				delete conv->weights;
				delete conv->biases;
			}
			delete point;
		}

		for(int t=0; t < __trainingData.size(); t++) // class
			for(int i = 0; i < __trainingData[t].size(); i++) // image in class
				delete __trainingData[t][i];

		destroyBatchNormCLMems();

		if(__isFinalized)
		{
			clReleaseCommandQueue(queue);
			clReleaseMemObject(*neurons);
			clReleaseMemObject(*prevNeurons);
			for(int w = 0; w < clWeights.size(); w++)
			{
				clReleaseMemObject(clWeights[w]);
				clReleaseMemObject(clBiases[w]);
			}
			clReleaseMemObject(denom);
			//running
			clReleaseKernel(convKernelF);
			clReleaseKernel(convKernelFC);
			clReleaseKernel(zeroPadKernelF);
			clReleaseKernel(maxPoolKernelF);
			clReleaseKernel(reluKernelF);
			clReleaseKernel(leakyReluKernelF);
			clReleaseKernel(softmaxKernelF);
			clReleaseProgram(CNForward);

			//training
			clReleaseKernel(reluKernel);
			clReleaseKernel(reluBackKernel);
			clReleaseKernel(leakyReluKernel);
			clReleaseKernel(leakyReluBackKernel);
			clReleaseKernel(convKernel);
			clReleaseKernel(convBackNeuronsKernel);
			clReleaseKernel(convBackBiasesKernel);
			clReleaseKernel(convBackWeightsKernel);
			clReleaseKernel(convBackWeightsMomentKernel);
			clReleaseKernel(zeroPadKernel);
			clReleaseKernel(zeroPadBackKernel);
			clReleaseKernel(maxPoolKernel);
			clReleaseKernel(maxPoolBackKernel);
			clReleaseKernel(softmaxKernel);
			clReleaseKernel(softmaxBackKernel);
			clReleaseKernel(copyArrayKernel);
			clReleaseKernel(maxSubtractionKernel);
			clReleaseKernel(vectorESumKernel);
			clReleaseKernel(plusEqualsKernel);
			clReleaseKernel(divideEqualsKernel);
			clReleaseKernel(zeroMemKernel);
			clReleaseKernel(convBackWeightsNoUpdateAccumKernel);
			clReleaseKernel(convBackBiasesNoUpdateAccumKernel);
			clReleaseKernel(updateWeightsKernel);
			clReleaseKernel(updateWeightsMomentKernel);
			clReleaseKernel(updateBiasesKernel);
			clReleaseKernel(batchNormKernel);
			clReleaseKernel(batchNormBackKernel);
			clReleaseKernel(batchNormRunKernel);
			clReleaseKernel(updateGammaAndBetaKernel);

			clReleaseProgram(CNTraining);
		}

		clReleaseContext(__context);
	}
	// printf("end destroy\n");
}

Net::~Net()
{
	destroy();
}

// See NOTE ON CONSTRUCTORS AND ASSIGNMENTS above
Net& Net::operator=(const Net& other)
{
	if(this != &other)
	{
		// printf("in copy-----------\n");
		//clean up any currently used OpenCL stuff besides the context
		destroy();
	
		this->__inited = true;
		//hyperparameters
		this->__learningRate = other.__learningRate;
		this->__RELU_CAP = other.__RELU_CAP;
		this->__LEAKY_RELU_CONST = other.__LEAKY_RELU_CONST;
		this->__l2Lambda = other.__l2Lambda;
		this->__MOMENT_CONST = other.__MOMENT_CONST;
		this->__MAX_NORM_CAP = other.__MAX_NORM_CAP;

		//Copy all layers
		this->copyLayers(other);
		//neuronSizes set by copyLayers
		//neuronDims set by copyLayers
		this->__autoActivLayer = other.__autoActivLayer;
		this->__maxNeuronSize = other.__maxNeuronSize;
		this->__defaultActivType = other.__defaultActivType;
		this->__maxWeightSize = other.__maxWeightSize;

		this->__isFinalized = false;
		this->__errorLog = "";

		//data and related members
		//__numClasses set in addTrainingData
		// this->__classes = other.__classes;
		this->__useMomentum = other.__useMomentum;
		this->__stuffBuilt = false;

		this->__isTraining = false;

		this->__mean = other.__mean;
		this->__stddev = other.__stddev;
		this->__trainingSize = other.__trainingSize;

		this->initOpenCL();
	}
	return *this;
}

/*****************************
*
* Net::copyLayers is a private function used by operator= and the copy constructor
* to copy over the CNN structure and layer weights. Uses the __layers member of this and other
*
*****************************/
void Net::copyLayers(const Net& other)
{
	// printf("copyLayers\n");
	//nicely delete all current layers
	Layer *point;
	while(!__layers.empty())
	{
		point = __layers.back();
		if(point->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)point;
			delete conv->weights;
			delete conv->biases;
		}
		__layers.pop_back();
		delete point;
	}
	__layers.resize(0);

	__layers.push_back(new Layer); //input layer
	int wid = other.__neuronDims[0][0];
	int hei = other.__neuronDims[0][1];
	int dep = other.__neuronDims[0][2];
	pushBackLayerSize(wid,hei,dep);

	for(int l = 1; l < other.__layers.size(); l++)
	{
		if(other.__layers[l]->layerType == CONV_LAYER)
		{
			const ConvLayer* conv = (ConvLayer*) other.__layers[l];

			ConvLayer* myconv = new ConvLayer();
			*myconv = *conv;
			myconv->layerType = CONV_LAYER;

			int widthNumer  = conv->paddedNeuronWidth - conv->filterSize;//prevWidth - filterSize + 2 * pad;
			int heightNumer = conv->paddedNeuronHeight - conv->filterSize;//prevHeight- filterSize + 2 * pad;
			int newWidth = widthNumer/conv->stride + 1;
			int newHeight = heightNumer/conv->stride + 1;
			int newDepth = conv->numBiases;// = numFilters;
			pushBackLayerSize(newWidth,newHeight,newDepth);
			__layers.push_back(myconv);
		}
		else if(other.__layers[l]->layerType == MAX_POOL_LAYER)
		{
			const MaxPoolLayer* pool = (MaxPoolLayer*)other.__layers[l];
			addMaxPoolLayer(pool->poolSize, pool->stride);
			// MaxPoolLayer* mypool = (MaxPoolLayer*)__layers.back();

			// printf("pool x stride. Old: %d x %d. New: %d x %d.\n",pool->poolSize,pool->stride, mypool->poolSize, mypool->stride);
		}
		else if(other.__layers[l]->layerType == ACTIV_LAYER)
		{
			const ActivLayer* act = (ActivLayer*)other.__layers[l];
			addActivLayer(act->activationType);
		}
		else if(other.__layers[l]->layerType == BATCH_NORM_LAYER)
		{
			const BatchNormLayer *otherBatch = (BatchNormLayer*)other.__layers[l];
			addBatchNormLayer(otherBatch->byFeatureMap);
			BatchNormLayer *batch = (BatchNormLayer*)__layers.back();

			for(int g = 0; g < batch->gamma.size(); g++)
			{
				batch->gamma[g] = otherBatch->gamma[g];
				batch->beta[g] = otherBatch->beta[g];
				batch->e[g] = otherBatch->e[g];
				batch->var[g] = otherBatch->var[g];
			}
		}
	}
	// printf("end copyLayers\n");
}

/*****************************
*
* Net::init creates the input layer and allows other layers to be added
* It also calls initOpenCL
*
*****************************/
void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	if(__inited)
		return;
	pushBackLayerSize(inputWidth,inputHeight,inputDepth);
    Layer* abstract = new Layer;
	__layers.resize(1);
    __layers.back() = abstract;
    __inited = true;
	initOpenCL();

	//find paths for the training and forward kernels
	if(fileExists("ConvNetForward_kernel.cl"))
		CNForwardPath = "ConvNetForward_kernel.cl";
	else if(fileExists("../kernels/ConvNetForward_kernel.cl"))
		CNForwardPath = "../kernels/ConvNetForward_kernel.cl";
	else if(fileExists("../../kernels/ConvNetForward_kernel.cl"))
		CNForwardPath = "../../kernels/ConvNetForward_kernel.cl";
	else if(fileExists("/Users/connorbowley/Development/neuralNetwork/kernels/ConvNetForward_kernel.cl"))
		CNForwardPath = "/Users/connorbowley/Development/neuralNetwork/kernels/ConvNetForward_kernel.cl";
	else if(fileExists("..\\..\\..\\kernels\\ConvNetForward_kernel.cl"))
		CNForwardPath = "..\\..\\..\\kernels\\ConvNetForward_kernel.cl";
	else
	{
		printf("Cannot find ConvNetForward_kernel.cl\n");
		exit(1);
	}

	if(fileExists("ConvNetTraining_kernel.cl"))
		CNTrainingPath = "ConvNetTraining_kernel.cl";
	else if(fileExists("../kernels/ConvNetTraining_kernel.cl"))
		CNTrainingPath = "../kernels/ConvNetTraining_kernel.cl";
	else if(fileExists("../../kernels/ConvNetTraining_kernel.cl"))
		CNTrainingPath = "../../kernels/ConvNetTraining_kernel.cl";
	else if(fileExists("/Users/connorbowley/Development/neuralNetwork/kernels/ConvNetTraining_kernel.cl"))
		CNTrainingPath = "/Users/connorbowley/Development/neuralNetwork/kernels/ConvNetTraining_kernel.cl";
	else if(fileExists("..\\..\\..\\kernels\\ConvNetTraining_kernel.cl"))
		CNTrainingPath = "..\\..\\..\\kernels\\ConvNetTraining_kernel.cl";
	else
	{
		printf("Cannot find ConvNetForward_kernel.cl\n");
		exit(1);
	}
}

/*****************************
*
* Net::initOpenCL gets platformIDs, deviceIDs and creates the context for OpenCL
*
*****************************/
void Net::initOpenCL()
{

	// printf("OpenCL initing\n");
	cl_int error;
	//platforms
	clGetPlatformIDs(0, nullptr, &__platformIdCount);
	__platformIds.resize(__platformIdCount);
	clGetPlatformIDs(__platformIdCount, __platformIds.data(), nullptr);

	//devices
	clGetDeviceIDs(__platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &__deviceIdCount);
	__deviceIds.resize(__deviceIdCount);
	clGetDeviceIDs(__platformIds[0], CL_DEVICE_TYPE_ALL, __deviceIdCount, __deviceIds.data(), nullptr);

	//context
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(__platformIds[0]),
		0,0
	};
	__context = clCreateContext(contextProperties, __deviceIdCount, __deviceIds.data(),
		nullptr, nullptr, &error);
	CheckError(error);

}

/*****************************************
 * Functions dealing with layers
 *****************************************/

/*****************************
*
* Adds an activation layer of whatever the default type has been set to.
* Initially set to RELU
*
*****************************/
bool Net::addActivLayer()
{
	return addActivLayer(__defaultActivType);
}

/*****************************
*
* Adds an activation layer. Usually (and by default) placed right after a Convolutional Layer
*
*****************************/
bool Net::addActivLayer(int activType)
{
	if(activType >= MAX_ACTIV || activType < 0)
	{
		printf("ActivLayer failing to add with type number %d\n", activType);
		return false;
	}

	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	ActivLayer* activ = new ActivLayer();
	activ->layerType = ACTIV_LAYER;
	activ->activationType = activType;

	__layers.push_back(activ);
	pushBackLayerSize(prevWidth,prevHeight,prevDepth);
	__isFinalized = false;
	return true;
}

/*****************************
*
* Adds a Convolutional (Conv) Layer with random weights. Used for untrained Conv Layers.
* Public function
*
*****************************/
bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad)
{
	return addConvLayer(numFilters, stride, filterSize, pad, string("random"));
}

/*****************************
*
* Adds a Batch Normalization (BatchNorm) Layer with default gamma and beta. Used for untrained BatchNorm Layers.
* Public function
*
*****************************/
bool Net::addBatchNormLayer(bool byFeatureMap)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	BatchNormLayer *batch = new BatchNormLayer();
	batch->layerType = BATCH_NORM_LAYER;
	if(byFeatureMap)
	{
		batch->gamma.resize(prevDepth,1);
		batch->beta.resize(prevDepth,0);
		batch->e.resize(prevDepth);
		batch->var.resize(prevDepth);
		batch->byFeatureMap = true;
	}
	else
	{
		int size = __neuronSizes.back();
		batch->gamma.resize(size,1);
		batch->beta.resize(size,0);
		batch->e.resize(size);
		batch->var.resize(size);
		batch->byFeatureMap = false; // by activation
	}

	__layers.push_back(batch);
	pushBackLayerSize(prevWidth,prevHeight,prevDepth);
	__isFinalized = false;
	return true;
}

bool Net::addBatchNormLayer(bool byFeatureMap, int gamma_size, const string& gamma, const string& beta, const string& e, const string& var)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	BatchNormLayer *batch = new BatchNormLayer();
	batch->layerType = BATCH_NORM_LAYER;
	if(byFeatureMap)
	{
		batch->gamma.resize(prevDepth,1);
		batch->beta.resize(prevDepth,0);
		batch->e.resize(prevDepth);
		batch->var.resize(prevDepth);
		batch->byFeatureMap = true;
	}
	else
	{
		int size = __neuronSizes.back();

		batch->gamma.resize(size,1);
		batch->beta.resize(size,0);
		batch->e.resize(size);
		batch->var.resize(size);
		batch->byFeatureMap = false; // by activation
	}
	vector<string> gamma_array = convnet::split(gamma,',');
	vector<string> beta_array = convnet::split(beta,',');
	vector<string> e_array = convnet::split(e,',');
	vector<string> var_array = convnet::split(var,',');
	for(int i = 0; i < gamma_size; i++)
	{
		batch->gamma[i] = stod(gamma_array[i]);
		batch->beta[i] = stod(beta_array[i]);
		batch->e[i] = stod(e_array[i]);
		batch->var[i] = stod(var_array[i]);
	}

	__layers.push_back(batch);
	pushBackLayerSize(prevWidth,prevHeight,prevDepth);
	__isFinalized = false;
	return true;
}

/*****************************
*
* Adds a Convolutional Layer with weights specified by the string (could be the string random as it is above.)
* Private function
*
*****************************/
bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, const string& weightsAndBiases)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];
	//check to make sure stride fits
	int widthNumer  = prevWidth - filterSize + 2 * pad;
	int heightNumer = prevHeight- filterSize + 2 * pad;
	if(widthNumer % stride != 0 || heightNumer % stride != 0) //incorrect hyperparameters
		return false;

	int newWidth = widthNumer/stride + 1;
	int newHeight = heightNumer/stride + 1;
	int newDepth = numFilters;


	ConvLayer* conv = new ConvLayer();
	conv->layerType = CONV_LAYER;
	conv->stride = stride;
	conv->filterSize = filterSize;
	//padding stuff
	conv->padding = pad;
	conv->paddedNeuronWidth = prevWidth + 2 * pad;
	conv->paddedNeuronHeight = prevHeight + 2 * pad;
	conv->paddedNeuronSize = conv->paddedNeuronWidth * conv->paddedNeuronHeight * prevDepth;

	int newNeuronSize = newWidth * newHeight * newDepth;
	conv->maxSizeNeeded = GETMAX(newNeuronSize, conv->paddedNeuronSize);

	//weights and biases
	conv->numBiases = numFilters;
	conv->numWeights = filterSize * filterSize * prevDepth * numFilters;
	conv->weights = new double[conv->numWeights];
	conv->biases = new double[conv->numBiases];
	if(weightsAndBiases.find("random") != string::npos)
		initRandomWeights(conv, prevDepth);
	else
		initWeights(conv, weightsAndBiases);

	__layers.push_back(conv);
	pushBackLayerSize(newWidth, newHeight, newDepth);

	//check to see if it has max weights
	if(conv->numWeights > __maxWeightSize)
		__maxWeightSize = conv->numWeights;

	if(__autoActivLayer)
		addActivLayer();
	__isFinalized = false;
	return true;
}

/*****************************
*
* Essential "fakes" a Fully Connected layer used in ANNs with a Convolutional Layer
* Public
*
*****************************/
bool Net::addFullyConnectedLayer(int outputSize)
{
	return addConvLayer(outputSize, 1, __neuronDims.back()[0], 0);
}

/*****************************
*
* Adds a Max Pooling Layer.
* Public
*
*****************************/
bool Net::addMaxPoolLayer(int poolSize, int stride)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	int widthNumer  = prevWidth - poolSize;
	int heightNumer = prevHeight- poolSize;
	if(widthNumer % stride != 0 || heightNumer % stride != 0) //incorrect hyperparameters
		return false;

	int newWidth = widthNumer/stride + 1;
	int newHeight = heightNumer/stride + 1;
	int newDepth = prevDepth;

	MaxPoolLayer* pool = new MaxPoolLayer();
	pool->layerType = MAX_POOL_LAYER;
	pool->stride = stride;
	pool->poolSize = poolSize;

	__layers.push_back(pool);
	pushBackLayerSize(newWidth, newHeight, newDepth);
	__isFinalized = false;
	return true;
}

/*****************************
*
* Adds an Avg Pooling Layer.
* Public
*
*****************************/
bool Net::addAvgPoolLayer(int poolSize, int stride)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	int widthNumer  = prevWidth - poolSize;
	int heightNumer = prevHeight- poolSize;
	if(widthNumer % stride != 0 || heightNumer % stride != 0) //incorrect hyperparameters
		return false;

	int newWidth = widthNumer/stride + 1;
	int newHeight = heightNumer/stride + 1;
	int newDepth = prevDepth;

	AvgPoolLayer* pool = new AvgPoolLayer();
	pool->layerType = AVG_POOL_LAYER;
	pool->stride = stride;
	pool->poolSize = poolSize;

	__layers.push_back(pool);
	pushBackLayerSize(newWidth, newHeight, newDepth);
	__isFinalized = false;
	return true;
}

/*****************************
*
* Adds the layer size info into __neuronSizes and __neuronDims
* Private
*
*****************************/
void Net::pushBackLayerSize(int width, int height, int depth)
{
	__neuronSizes.push_back(width * height * depth);
	__neuronDims.resize(__neuronDims.size() + 1);
	__neuronDims.back().resize(3);
	__neuronDims.back()[0] = width;
	__neuronDims.back()[1] = height;
	__neuronDims.back()[2] = depth;
}

/*****************************
*
* Initalizes random weights for a Convolutional layer. 
* @parameter conv A pointer to the ConvLayer to get random weights
* @parameter prevDepth The depth of the layer above
* Private
*
*****************************/
void Net::initRandomWeights(ConvLayer* conv, int prevDepth)
{
    //cout << "making random weights" << endl;
	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	//use the number of inputs to get the random start weights
	double numInputs = conv->filterSize * conv->filterSize * prevDepth + 1;
	double sqrtIn = pow(2/numInputs,.5);
	normal_distribution<double> distr(0,sqrtIn);

	//printf("num weights: %d sqrtIn %lf\n", conv->numWeights, sqrtIn);

	conv->weights = new double[conv->numWeights];
	conv->biases  = new double[conv->numBiases];

	for(int f = 0;f < conv->numWeights; f++)
		conv->weights[f] = distr(gen);

	for(int b=0; b < conv->numBiases; b++)
		conv->biases[b] = 0;
}

/*****************************
*
* Inits the weights of a Conv Layer according to string in format weight,weight,...,weight_bias,bias,bias
* Private
*
*****************************/
void Net::initWeights(ConvLayer* conv, const string& weights)
{
	// cout << "start init weights" << endl;
	int startIndex = 0, endIndex;
	for(int f = 0; f < conv->numWeights; f++)
	{
		endIndex = weights.find(',',startIndex);
		conv->weights[f] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;	
	}
	// now do the biases
	startIndex = weights.find('_') + 1;
	for(int b=0; b < conv->numBiases; b++)
	{
		endIndex = weights.find(',',startIndex);
		conv->biases[b] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;
	}
	// cout << "done init weights" << endl;
}

/*****************************
*
* Sets default activation type
* Public
*
*****************************/
bool Net::setActivType(int activationType)
{
	if(activationType >= MAX_ACTIV || activationType < 0)
		return false;
	__defaultActivType = activationType;
	return true;
}

/*****************************
*
* Prints the layer set and dimesions to the terminal
* Public
*
*****************************/
void Net::printLayerDims() const 
{
	printf("Input         %d x %d x %d\n", __neuronDims[0][0], __neuronDims[0][1], __neuronDims[0][2]);
	for(int i=1; i < __neuronDims.size(); i++)
		if(__layers[i]->layerType == CONV_LAYER)
			printf("Convolution   %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
			printf("Max Pool      %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			ActivLayer* act = (ActivLayer*)__layers[i];
			if(act->activationType == LEAKY_RELU)
				printf("Leaky RELU    %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
			else if(act->activationType == RELU)
				printf("RELU          %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		}
		else if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			if(((BatchNormLayer*)__layers[i])->byFeatureMap)
				printf("BatchNorm (f) %d x %d x %d\n",__neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
			else
				printf("BatchNorm (a) %d x %d x %d\n",__neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);

		}
		else
			printf("Unknown Layer: %d\n",__layers[i]->layerType);
}

/*****************************
*
* Returns input height as int
* Public
*
*****************************/
int Net::getInputHeight() const
{
	return __neuronDims[0][0];
}

/*****************************
*
* Returns input width as int
* Public
*
*****************************/
int Net::getInputWidth() const
{
	return __neuronDims[0][1];
}

/*****************************
*
* Sets whether or not an Activation Layer should automatically be added after every Conv Layer
* Defaults to true.
* Public
*
*****************************/
void Net::setAutoActivLayer(bool isAuto)
{
	__autoActivLayer = isAuto;
}

/*****************************
*
* Sets the name of the file the trained CNN should be saved to.
* Public
*
*****************************/
void Net::setSaveName(const char* saveName)
{
	__saveName = string(saveName);
	__saveNet = true;
}

/*****************************
*
* Sets the name of the file the trained CNN should be saved to.
* Public
*
*****************************/
void Net::setSaveName(string saveName)
{
	__saveName = saveName;
	__saveNet = true;
}

/*****************************
*
* Returns a string of errors (if any) that were found by Net::finalize()
* Public
*
*****************************/
string Net::getErrorLog() const
{
	return __errorLog;
}

/*****************************
*
* Checks to make sure it is a valid CNN. If it is valid, it sets up most of the OpenCL stuff
* Returns whether or not the CNN is valid.
* Public
*
*****************************/
bool Net::finalize()
{
	#ifdef _DEBUG
	printf("Starting finalize\n");
	#endif
	if(__isFinalized)
		return true;

	__errorLog = "";
	bool returnVal = true;

	//check to see if there is enough output neurons on last layer for each class to be represented
	if(__neuronSizes.back() < __numClasses)
	{
		__errorLog += "OUTPUT_SIZE_ERROR: There are not enough output neurons to represent all classes\n";
		returnVal = false;
	}

	//need at least one hidden layer
	if(__layers.size() < 2)
	{
		__errorLog += "NET_TOO_SMALL_ERROR: There must be at least one hidden layer.\n";
		returnVal = false;
	}

	//make sure if using TRAIN_RATIO we have good ratio vals for all classes
	if(__trainingType == TRAIN_RATIO)
	{
		for(int i = 0; i < __trainRatioAmounts.size(); i++)
			if(__trainRatioAmounts[i] < 0) // 0 is valid. It just means don't include any of this class
			{
				__errorLog += "INVALID_RATIOS_ERROR: There is a ratio of TRAIN_RATIO that is negative.\n";
				returnVal = false;
				break;
			}
	}

	//if we are going to return false, do it now. else set up OpenCL
	if(!returnVal)
		return returnVal;

	//Do a bunch of OpenCL stuff
	int q = 0;
	if(__device != -1)
		q = __device;
	else if(__useGPU)
	{
		for(int i = 1; i < __deviceIdCount; i++)
		{
			cl_device_type type;
			CheckError(clGetDeviceInfo(__deviceIds[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
			if(type == CL_DEVICE_TYPE_GPU)
				q = i;
		}
		__device = q;
	}
	else
		__device = 0;

	printf("Finalizing CNN using device %d\n",q);


	

	cl_int error;

	if(!__stuffBuilt) //if the OpenCL kernels have not been loaded and built
	{
		//build the program
			//running
		// const char* foptions = "-cl-mad-enable";
		const cl_device_id* deviceToBuild = &(__deviceIds[q]);

		__program_creation_mutex.lock();
		if(!__programs_already_created)
		{
			CNForward = CreateProgram(LoadKernel(CNForwardPath.c_str()), __context, RUNNING_PROGRAM);
			CheckError(clBuildProgram(CNForward, 1, deviceToBuild, nullptr, nullptr, nullptr));
			//training
			CNTraining = CreateProgram(LoadKernel(CNTrainingPath.c_str()), __context, TRAINING_PROGRAM);
			CheckError(clBuildProgram(CNTraining, 1, deviceToBuild, nullptr, nullptr, nullptr));
			__programs_already_created = true;
		}
		__program_creation_mutex.unlock();

		
		
		//Create the kernels; check for errors
			//running
		reluKernelF = clCreateKernel(CNForward, "relu", &error); CheckError(error);
		leakyReluKernelF = clCreateKernel(CNForward, "leakyRelu", &error); CheckError(error);
		convKernelF = clCreateKernel(CNForward, "convolve", &error); CheckError(error);
		convKernelFC = clCreateKernel(CNForward, "convolveConstant", &error); CheckError(error);
		zeroPadKernelF = clCreateKernel(CNForward, "zeroPad", &error); CheckError(error);
		maxPoolKernelF = clCreateKernel(CNForward, "maxPool", &error); CheckError(error);
		softmaxKernelF = clCreateKernel(CNForward, "softmax_allCL", &error); CheckError(error);

			//training
		reluKernel = clCreateKernel(CNTraining, "relu", &error); CheckError(error);
		reluBackKernel = clCreateKernel(CNTraining, "relu_back", &error); CheckError(error);
		leakyReluKernel = clCreateKernel(CNTraining, "leakyRelu", &error); CheckError(error);
		leakyReluBackKernel = clCreateKernel(CNTraining, "leakyRelu_back", &error); CheckError(error);
		convKernel = clCreateKernel(CNTraining, "convolve", &error); CheckError(error);
		convBackNeuronsKernel = clCreateKernel(CNTraining, "convolve_back_neurons", &error); CheckError(error);
		convBackBiasesKernel = clCreateKernel(CNTraining, "convolve_back_biases", &error); CheckError(error);
		convBackWeightsKernel = clCreateKernel(CNTraining, "convolve_back_weights", &error); CheckError(error);
		convBackWeightsMomentKernel = clCreateKernel(CNTraining, "convolve_back_weights_moment", &error); CheckError(error);
		zeroPadKernel = clCreateKernel(CNTraining, "zeroPad", &error); CheckError(error);
		zeroPadBackKernel = clCreateKernel(CNTraining, "zeroPad_back", &error); CheckError(error);
		maxPoolKernel = clCreateKernel(CNTraining, "maxPool", &error); CheckError(error);
		maxPoolBackKernel = clCreateKernel(CNTraining, "maxPool_back", &error); CheckError(error);
		softmaxKernel = clCreateKernel(CNTraining, "softmax", &error); CheckError(error);
		softmaxBackKernel = clCreateKernel(CNTraining, "softmax_back", &error); CheckError(error);
		copyArrayKernel = clCreateKernel(CNTraining, "copyArray", &error); CheckError(error);
		maxSubtractionKernel = clCreateKernel(CNTraining, "maxSubtraction", &error); CheckError(error);
		vectorESumKernel = clCreateKernel(CNTraining, "vectorESum", &error); CheckError(error);
		plusEqualsKernel = clCreateKernel(CNTraining, "plusEquals", &error); CheckError(error);
		divideEqualsKernel = clCreateKernel(CNTraining, "divideEquals", &error); CheckError(error);

		zeroMemKernel = clCreateKernel(CNTraining, "zero_out", &error); CheckError(error);
		convBackWeightsNoUpdateAccumKernel = clCreateKernel(CNTraining, "convolve_back_weights_no_update_accum", &error); CheckError(error);
		convBackBiasesNoUpdateAccumKernel = clCreateKernel(CNTraining, "convolve_back_biases_no_update_accum", &error); CheckError(error);
		updateWeightsKernel = clCreateKernel(CNTraining, "update_weights", &error); CheckError(error);
		updateWeightsMomentKernel = clCreateKernel(CNTraining, "update_weights_moment", &error); CheckError(error);
		updateBiasesKernel = clCreateKernel(CNTraining, "update_biases", &error); CheckError(error);

		// printf("making bn kernels in finalize... ");
		batchNormRunKernel = clCreateKernel(CNTraining, "batch_norm_run", &error); CheckError(error);
		batchNormKernel = clCreateKernel(CNTraining, "batch_norm", &error); CheckError(error);
		batchNormBackKernel = clCreateKernel(CNTraining, "batch_norm_back", &error); CheckError(error);
		updateGammaAndBetaKernel = clCreateKernel(CNTraining, "update_gamma_and_beta", &error); CheckError(error);
		// printf("done\n");

		//make the queue
		queue = clCreateCommandQueue(__context, __deviceIds[q], 0, &error); 
		CheckError(error);
		__stuffBuilt = true;
	}

	int numConvLayers = 0;
	__maxNeuronSize = __neuronSizes[0];

	//if it has been previously finalized there might be stuff in the weights and biases, so remove it
	for(int i = 0; i < clWeights.size(); i++)
	{
		clReleaseMemObject(clWeights[i]);
		clReleaseMemObject(clBiases[i]);
	}
	clWeights.resize(0);
	clBiases.resize(0);

	// for(int i = 0; i < __layers.size(); i++)
	// 	printf("type %d\n", __layers[i]->layerType);
	for(int i=1; i < __layers.size(); i++)
	{
		int type = __layers[i]->layerType;
		if(type == CONV_LAYER)
		{
			numConvLayers++;
			ConvLayer *conv = (ConvLayer*) __layers[i];
			if(__constantMem)
			{
				clWeights.push_back(clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numWeights, conv->weights, &error));
				CheckError(error);
				clBiases.push_back(clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numBiases, conv->biases, &error));
				CheckError(error);
			}
			else
			{
				clWeights.push_back(clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numWeights, conv->weights, &error));
				CheckError(error);
				clBiases.push_back(clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numBiases, conv->biases, &error));
				CheckError(error);
			}

			//none of the other layers have the ability to increase the size from the layer before
			if(conv->maxSizeNeeded > __maxNeuronSize)
			{
				__maxNeuronSize = conv->maxSizeNeeded;
			}
		}
		else if (type == MAX_POOL_LAYER); //these are here so they don't catch on the else statement
		else if (type == ACTIV_LAYER);
		else if (type == BATCH_NORM_LAYER);
		else
		{
			cout << "Unknown layer type. Aborting finalize." << endl;
			cout << "Type: " << type << endl;
			return false;
		}
	}
	// cout << "Max Neuron Size: " << __maxNeuronSize << endl;
	n = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	neurons = &n;
	p = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	prevNeurons = &p;

	denom = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);
	CheckError(error);


	__isFinalized = true;

	#ifdef _DEBUG
	printf("finalize passed\n");
	#endif
	return true;
}

/*****************************
*
* Releases memory held by OpenCL MemObjects
* Private
*
*****************************/
void Net::releaseCLMem()
{
	for(int i = 0; i < clWeights.size(); i++)
	{
		clReleaseMemObject(clWeights[i]);
		clReleaseMemObject(clBiases[i]);
	}
	clWeights.resize(0);
	clBiases.resize(0);

	clReleaseMemObject(*neurons);
	clReleaseMemObject(*prevNeurons);
}

/*****************************
*
* Sets the learning rate (aka step size) for backpropagation
* Public
*
*****************************/
bool Net::set_learningRate(double rate)
{
	if(rate < 0)
		return false;
	__learningRate = rate;
	return true;
}

/*****************************
*
* Sets the maximum absolute value (basically a cap) that can come out of a RELU or Leaky_RELU activation layer neuron
* Default 5000.0
* Public
*
*****************************/
bool Net::set_RELU_CAP(double cap)
{
	if(cap <= 0)
		return false;
	__RELU_CAP = cap;
	__isFinalized = false;
	return true;
}

/*****************************
*
* Sets the constant by which to multiply the neuron value by if it is negative. Constant must be between [0,1]
* Default 0.01
* Public
*
*****************************/
bool Net::set_LEAKY_RELU_CONST(double lconst)
{
	if(lconst < 0 || 1 < lconst)
		return false;
	__LEAKY_RELU_CONST = lconst;
	__isFinalized = false;
	return true;
}

/*****************************
*
* Sets the L2 regularization lambda constant (how strongly it regulates)
* Default 0.05
* Public
*
*****************************/
bool Net::set_l2Lambda(double lambda)
{
	if(lambda < 0)
		return false;
	__l2Lambda  = lambda;
	__isFinalized = false;
	return true;
}

/*****************************
*
* Sets the (decay) constant used for momentum
* Default 0.9
* Public
*
*****************************/
bool Net::set_MOMENT_CONST(double mconst)
{
	if(mconst < 0 || 1 < mconst)
		return false;
	__MOMENT_CONST = mconst;
	__isFinalized = false;
	return true;
}

/*****************************
*
* Sets the maximum value as weight can take.
* Default 3.0
* Public
*
*****************************/
bool Net::set_MAX_NORM_CAP(double cap)
{
	if(cap < 0)
		return false;
	__MAX_NORM_CAP = cap;
	__isFinalized = false;
	return true;
}

/*****************************************
 * Running and Training
 *****************************************/


/*****************************
*
* Net::run() is the main function used to run already trained CNNs
* Requires: init or load to have been called. addData or setData to have been called.
* Will finalize net if not done. Will preprocess data if not done.
*
* Puts output into __confidences. Can be accessed publically through getCalculatedClasses()
* and getConfidences()
*
* Uses OpenCL kernel calls
*
* Public function
*
*****************************/
void Net::run()
{

	// CheckError(-11);
 	// if(useGPU != __useGPU)
 	// {
 	// 	__useGPU = useGPU;
 	// 	__isFinalized = false;
 	// }

 	// for(int j =0 ; j < __neuronSizes.size(); j++)
 	// 	printf("Layer %d: Size %d\n", j, __neuronSizes[j]);

	//can call run no matter what for backward compatibility
 	// if(getNumBatchNormLayers() > 0)
 	// {
 		batchNormRun(); // this runs MUCH faster
 		return;
 	// }

 	if(!__isFinalized)
 		finalize();

 	if(!__dataPreprocessed)
 	{
 		// if(__preprocessIndividual)
 		if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 			preprocessDataIndividual();
 		else if(__preprocessType == __PREPROCESS_COLLECTIVE)
 			preprocessDataCollective();
 	}

 	// printf("__data[0][0] = %lf\n",__data[0][0]);

 	if(!__isTraining)
 	{
 		// printf("Not training\n");
 		__dataPointer = &__data;
 	}
 	else
 	{
 		__confidences.resize(__dataPointer->size());
 	}

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	for(int i=0; i < __confidences.size(); i++)
 		__confidences[i].resize(__neuronSizes.back());

 	vector<double> test(__maxNeuronSize);

	cl_mem *temp;

	// printf("Running\n");
 	for(int r = 0; r < __dataPointer->size(); r++)
 	{
 		// printf("Row image num %d\n",r);
 		//put in the next image
 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
				sizeof(double) * __neuronSizes[0],
				(*__dataPointer)[r].data(), 0, nullptr, nullptr));
		clFinish(queue);

		// printf("Image\n");
		// for(int j = 0; j < __dataPointer->at(r).size(); j++)
		// 	cout << __dataPointer->at(r)[j] << ",";
		// cout << endl << endl;

		int curConvLayer = 0;
		size_t globalWorkSize[] = {0,0,0};
		//go through the layers
		for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
		{
			// printf("Layer %d, type %d\n", i, __layers[i]->layerType);
			if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)__layers[i];

				// pull the weights from the cl_mem and see if they are the same
				// cout << "Weights ConvLayer " << curConvLayer << " Layer #" << i << endl;
				// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
				// 	test.data(), 0, nullptr, nullptr));
				// for(int j=0; j< conv->numWeights; j++)
				// {
				// 	cout << test[j] << ",";
				// }
				// cout << endl << endl;
				// getchar();


				
				if(conv->padding != 0) //if we need to do padding on the input
				{
					clSetKernelArg(zeroPadKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(zeroPadKernelF, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(zeroPadKernelF, 2, sizeof(int), &(conv->padding)); // padding
					clSetKernelArg(zeroPadKernelF, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
					clSetKernelArg(zeroPadKernelF, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
					clSetKernelArg(zeroPadKernelF, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

					// run it for the size of the new array
					globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
					CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					clFinish(queue);

					//swap the buffers so prevNeurons holds the zero padded data
					temp = neurons;
					neurons = prevNeurons;
					prevNeurons = temp;
				}

				if(__constantMem)
				{
					// int prevDepth = __neuronDims[i-1][2];
					// int strxdep = conv->stride * prevDepth;
					// int amountToNextLayer = (conv->paddedNeuronWidth - conv->filterSize) * prevDepth;
					// int filterLayerSize = conv->filterSize * prevDepth;
					// int numBlocksPerRow = (conv->paddedNeuronWidth - conv->filterSize)/conv->stride + 1;
					//cout << "using constant" << endl;
					clSetKernelArg(convKernelFC, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernelFC, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernelFC, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
					clSetKernelArg(convKernelFC, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
					clSetKernelArg(convKernelFC, 4, sizeof(int), &(conv->numBiases)); //numFilters
					clSetKernelArg(convKernelFC, 5, sizeof(int), &(conv->filterSize));
					clSetKernelArg(convKernelFC, 6, sizeof(int), &(conv->stride));
					// clSetKernelArg(convKernelFC, 6, sizeof(int), &strxdep);
					clSetKernelArg(convKernelFC, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
					clSetKernelArg(convKernelFC, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth
					// clSetKernelArg(convKernelFC, 8, sizeof(int), &amountToNextLayer);
					// clSetKernelArg(convKernelFC, 9, sizeof(int), &filterLayerSize);
					// clSetKernelArg(convKernelFC, 10, sizeof(int), &numBlocksPerRow);

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, convKernelFC, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else
				{
					clSetKernelArg(convKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernelF, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernelF, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
					clSetKernelArg(convKernelF, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
					clSetKernelArg(convKernelF, 4, sizeof(int), &(conv->numBiases)); //numFilters
					clSetKernelArg(convKernelF, 5, sizeof(int), &(conv->filterSize));
					clSetKernelArg(convKernelF, 6, sizeof(int), &(conv->stride));
					clSetKernelArg(convKernelF, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
					clSetKernelArg(convKernelF, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, convKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				curConvLayer++;
			}
			else if(__layers[i]->layerType == MAX_POOL_LAYER)
			{
				MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
				clSetKernelArg(maxPoolKernelF, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(maxPoolKernelF, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(maxPoolKernelF, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
				clSetKernelArg(maxPoolKernelF, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
				clSetKernelArg(maxPoolKernelF, 4, sizeof(int), &(pool->poolSize)); //poolsize
				clSetKernelArg(maxPoolKernelF, 5, sizeof(int), &(pool->stride)); //stride
				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(__layers[i]->layerType == ACTIV_LAYER)
			{
				int type = ((ActivLayer*)__layers[i])->activationType;
				if(type == RELU)
				{
					clSetKernelArg(reluKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluKernelF, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, reluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else if(type == LEAKY_RELU)
				{
					clSetKernelArg(leakyReluKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluKernelF, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
			}

			clFinish(queue);

			// cout << "Layer " << i << endl;
			// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
			// 	test.data(), 0, nullptr, nullptr));
			// for(int j=0; j< __neuronSizes[i]; j++)
			// {
			// 	cout << test[j] << ", ";
			// }
			// cout << endl << endl;
			// getchar();

			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;
		}

		if(usesSoftmax)
		{
			// cout << "Softmax" << endl;
			//softmax. 
			globalWorkSize[0] = 1;
			//maxSubtraction. arg 1 set above
			clSetKernelArg(maxSubtractionKernel, 0, sizeof(cl_mem), prevNeurons);
			CheckError(clEnqueueNDRangeKernel(queue, maxSubtractionKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);

			//vectorESum. arg 1 set above
			clSetKernelArg(vectorESumKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(vectorESumKernel, 2, sizeof(cl_mem), &denom);
			CheckError(clEnqueueNDRangeKernel(queue, vectorESumKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);
			
			//softmax
			globalWorkSize[0] = (size_t)__neuronSizes.back();
			clSetKernelArg(softmaxKernelF, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxKernelF, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxKernelF, 2, sizeof(cl_mem), &denom);
			CheckError(clEnqueueNDRangeKernel(queue, softmaxKernelF, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);
		}

		CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	__confidences[r].data(), 0, nullptr, nullptr));

	// for(int c=0; c < __confidences[r].size(); c++)
	// 	cout << __confidences[r][c] << " ";
	// cout << endl;
	// getchar();


 	}
 	// printf("End run\n");
}

/*****************************
*
* DO NOT USE
* Net::run_parallel() is a function used to run already trained CNNs. It is similar to run() but more threaded
* Requires: init or load to have been called. addData or setData to have been called.
* Will finalize net if not done. Will preprocess data if not done.
*
* Puts output into __confidences. Can be accessed publicly through getCalculatedClasses()
* and getConfidences()
*
* Uses OpenCL kernel calls
*
* Public function
*
*****************************/
void Net::run_parallel()
{
 	// if(useGPU != __useGPU)
 	// {
 	// 	__useGPU = useGPU;
 	// 	__isFinalized = false;
 	// }

 	// for(int j =0 ; j < __neuronSizes.size(); j++)
 	// 	printf("Layer %d: Size %d\n", j, __neuronSizes[j]);

 	if(!__isFinalized)
 		finalize();

 	if(!__dataPreprocessed)
 	{
 		if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 			preprocessDataIndividual();
 		else if(__preprocessType == __PREPROCESS_COLLECTIVE)
 			preprocessDataCollective();
 	}

 	// printf("__data[0][0] = %lf\n",__data[0][0]);

 	if(!__isTraining)
 	{
 		// printf("Not training\n");
 		__dataPointer = &__data;
 	}
 	else
 	{
 		__confidences.resize(__dataPointer->size());
 	}

 	//set some softmax related args that won't change
 	// clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	// clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	for(int i=0; i < __confidences.size(); i++)
 		__confidences[i].resize(__neuronSizes.back());

 	// vector<double> test(__maxNeuronSize);

 	/******************************
	*
	*  Set up everything for runin OpenCL in parallel
	*
	******************************/
 	cl_int error;
	uint clBatchSize = 10;
	vector<cl_mem> p(clBatchSize), n(clBatchSize);
	vector<cl_mem*> prevNeurons(clBatchSize), neurons(clBatchSize);
	vector<cl_command_queue> queues(clBatchSize);
	vector<cl_mem> denoms(clBatchSize);
	vector<Kernels> kerns(clBatchSize);
	int q = __device == -1 ? 0 : __device;
	for(int a = 0; a< clBatchSize; a++)
	{
		p[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		prevNeurons[a] = &(p[a]);
		n[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		neurons[a] = &(n[a]);

		queues[a] = clCreateCommandQueue(__context, __deviceIds[q], 0, &error);
		CheckError(error);

		denoms[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);
		CheckError(error);

		buildKernels(kerns[a], q);
	}
	for(int i = 0; i < kerns.size(); i++)
	{
	 	clSetKernelArg(kerns[i].maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
	 	clSetKernelArg(kerns[i].vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	}

 	vector<double> soft(__neuronSizes.back());
 	vector<thread> thr(clBatchSize);

	cl_mem *temp;
	// printf("Running\n");
	uint dataSize = __dataPointer->size();
 	for(int r = 0; r < dataSize;) //no r++ b/c of myclBatchSize
 	{
 		int myclBatchSize = r+clBatchSize >= dataSize ? dataSize-r : clBatchSize;
 		for(int t = 0; t < myclBatchSize; t++)
 		{
 			//write data
 			CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0,
 				sizeof(double) * __neuronSizes[0], (*__dataPointer)[r].data(), 0, nullptr, nullptr));

 			//spawn thread
 			cl_mem **prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
 			thr[t] = thread([=] {feedForward_running(prevptr, nptr, ref(__neuronDims), ref(__layers), ref(clWeights), ref(clBiases), 
 				ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
 		}
 		for(int t = 0; t < myclBatchSize; t++)
 		{
 			//join threads when finished
 			thr[t].join();
 			//put CNN output into __confidences
 			CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	__confidences[r+t].data(), 0, nullptr, nullptr));
 		}
 		r += myclBatchSize;
 	}

 	for(int i = 0; i < clBatchSize; i++)
 	{
 		clReleaseMemObject(p[i]);
		clReleaseMemObject(n[i]);
		clReleaseCommandQueue(queues[i]);
		clReleaseMemObject(denoms[i]);
 		// releaseKernels(kerns[i]); // automatic when Kernels destructor called
 	}
}


/*****************************
*
* Returns the calculated classes for each data element as found by run(), in the same order they were inputted.
* Returns as the true value int, can be mapped back to names using getClassForTrueVal(int) or 
* getClassNames() which uses the ClassInfo struct
*
* Whichever class has the highest value in __confidences for a data element is considered the calculated class
*
* @parameter dest std::vector<int>& A reference to the vector of ints you want the results stored in
*
* Public function
*
*****************************/
void Net::getCalculatedClasses(vector<int>& dest) const
{
 	dest.resize(__confidences.size());
 	for(int i=0; i < dest.size(); i++)
 		dest[i] = getMaxElementIndex(__confidences[i]);
}

/*****************************
*
* Returns the confidences (basically raw output) from the CNN for each data in the same order they were inputted.
* 
* @parameter confidences std::vector<std::vector<double> >& A reference to the container where you want the output stored into.
*
*****************************/
void Net::getConfidences(vector<vector<double> >& confidences) const
{
 	confidences = __confidences;
}

/*****************************
*
* Just gets the maximum element of a vector of doubles
* Private
*
*****************************/
int Net::getMaxElementIndex(const vector<double>& vect) const
{
	if(vect.size() < 1)
		return -1;
	double max = vect[0];
	double maxIndex = 0;
	for(int i=1; i< vect.size(); i++)
		if(vect[i] > max)
		{
			max = vect[i];
			maxIndex = i;
		}
	return maxIndex;
}

/*****************************
*
* Just gets the max element of an  array of ints
* Private
*
*****************************/
int Net::getMaxElementIndex(const vector<int>& vect) const
{
	if(vect.size() < 1)
		return -1;
	double max = vect[0];
	double maxIndex = 0;
	for(int i=1; i< vect.size(); i++)
		if(vect[i] > max)
		{
			max = vect[i];
			maxIndex = i;
		}
	return maxIndex;
}

void Net::feedForward_running(cl_mem** prevNeurons, cl_mem** neurons, vector<vector<int> >& __neuronDims, vector<Layer*>& __layers,
	const vector<cl_mem>& clWeights, const vector<cl_mem>& clBiases, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k)
{
	int curConvLayer = 0;
	size_t globalWorkSize[] = {0,0,0};
	//go through the layers
	cl_mem* temp;
	for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
	{
		// printf("Layer %d, type %d\n", i, __layers[i]->layerType);
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];

			// pull the weights from the cl_mem and see if they are the same
			// cout << "Weights ConvLayer " << curConvLayer << " Layer #" << i << endl;
			// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
			// 	test.data(), 0, nullptr, nullptr));
			// for(int j=0; j< conv->numWeights; j++)
			// {
			// 	cout << test[j] << ",";
			// }
			// cout << endl << endl;
			// getchar();


			
			if(conv->padding != 0) //if we need to do padding on the input
			{
				clSetKernelArg(k.zeroPadKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.zeroPadKernelF, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.zeroPadKernelF, 2, sizeof(int), &(conv->padding)); // padding
				clSetKernelArg(k.zeroPadKernelF, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
				clSetKernelArg(k.zeroPadKernelF, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
				clSetKernelArg(k.zeroPadKernelF, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

				// run it for the size of the new array
				globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue, k.zeroPadKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				clFinish(queue);

				//swap the buffers so prevNeurons holds the zero padded data
				temp = *neurons;
				*neurons = *prevNeurons;
				*prevNeurons = temp;
			}

			if(__constantMem)
			{
				// int prevDepth = __neuronDims[i-1][2];
				// int strxdep = conv->stride * prevDepth;
				// int amountToNextLayer = (conv->paddedNeuronWidth - conv->filterSize) * prevDepth;
				// int filterLayerSize = conv->filterSize * prevDepth;
				// int numBlocksPerRow = (conv->paddedNeuronWidth - conv->filterSize)/conv->stride + 1;
				//cout << "using constant" << endl;
				clSetKernelArg(k.convKernelFC, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.convKernelFC, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.convKernelFC, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(k.convKernelFC, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
				clSetKernelArg(k.convKernelFC, 4, sizeof(int), &(conv->numBiases)); //numFilters
				clSetKernelArg(k.convKernelFC, 5, sizeof(int), &(conv->filterSize));
				clSetKernelArg(k.convKernelFC, 6, sizeof(int), &(conv->stride));
				// clSetKernelArg(k.convKernelFC, 6, sizeof(int), &strxdep);
				clSetKernelArg(k.convKernelFC, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
				clSetKernelArg(k.convKernelFC, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth
				// clSetKernelArg(k.convKernelFC, 8, sizeof(int), &amountToNextLayer);
				// clSetKernelArg(k.convKernelFC, 9, sizeof(int), &filterLayerSize);
				// clSetKernelArg(k.convKernelFC, 10, sizeof(int), &numBlocksPerRow);

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.convKernelFC, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else
			{
				clSetKernelArg(k.convKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.convKernelF, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.convKernelF, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(k.convKernelF, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
				clSetKernelArg(k.convKernelF, 4, sizeof(int), &(conv->numBiases)); //numFilters
				clSetKernelArg(k.convKernelF, 5, sizeof(int), &(conv->filterSize));
				clSetKernelArg(k.convKernelF, 6, sizeof(int), &(conv->stride));
				clSetKernelArg(k.convKernelF, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
				clSetKernelArg(k.convKernelF, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.convKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			curConvLayer++;
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
			clSetKernelArg(k.maxPoolKernelF, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.maxPoolKernelF, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.maxPoolKernelF, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
			clSetKernelArg(k.maxPoolKernelF, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
			clSetKernelArg(k.maxPoolKernelF, 4, sizeof(int), &(pool->poolSize)); //poolsize
			clSetKernelArg(k.maxPoolKernelF, 5, sizeof(int), &(pool->stride)); //stride
			globalWorkSize[0] = (size_t)__neuronSizes[i];
			CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolKernelF, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		}
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)__layers[i])->activationType;
			if(type == RELU)
			{
				clSetKernelArg(k.reluKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.reluKernelF, 1, sizeof(cl_mem), *neurons);

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.reluKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				clSetKernelArg(k.leakyReluKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.leakyReluKernelF, 1, sizeof(cl_mem), *neurons);

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}

		clFinish(queue);

		// cout << "Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = *neurons;
		*neurons = *prevNeurons;
		*prevNeurons = temp;
	}

	if(usesSoftmax)
	{
		// cout << "Softmax" << endl;
		//softmax. 
		globalWorkSize[0] = 1;
		//maxSubtraction. arg 1 set above
		clSetKernelArg(k.maxSubtractionKernel, 0, sizeof(cl_mem), *prevNeurons);
		CheckError(clEnqueueNDRangeKernel(queue, k.maxSubtractionKernel, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		//vectorESum. arg 1 set above
		clSetKernelArg(k.vectorESumKernel, 0, sizeof(cl_mem), *prevNeurons);
		clSetKernelArg(k.vectorESumKernel, 2, sizeof(cl_mem), &denom);
		CheckError(clEnqueueNDRangeKernel(queue, k.vectorESumKernel, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);
		
		//softmax
		globalWorkSize[0] = (size_t)__neuronSizes.back();
		clSetKernelArg(k.softmaxKernelF, 0, sizeof(cl_mem), *prevNeurons);
		clSetKernelArg(k.softmaxKernelF, 1, sizeof(cl_mem), *neurons);
		clSetKernelArg(k.softmaxKernelF, 2, sizeof(cl_mem), &denom);
		CheckError(clEnqueueNDRangeKernel(queue, k.softmaxKernelF, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);
	}
}


/*****************************
*
* Net::trainSetup is called from Net::train(). Makes sure training doesn't use constant mem, finalizes if needed, 
* preprocesses data, sets up Net::run() to work on test data, sets some clKernelArgs for softmax, and calls
* setupLayerNeeds() (always) and initVelocities() if using momentum
* Private
*
*****************************/
void Net::trainSetup(vector<cl_mem>& layerNeeds, vector<cl_mem>& velocities)
{
	//make sure we're finalized without __constantMem
	if(!__isFinalized || __constantMem)
 	{
 		__constantMem = false;
 		if(!finalize())
 		{
 			printf("Net was unable to finalize. Aborting training.\n");
 			return;
 		}
 	}

 	//Tell which device we're using to train
 	size_t valueSize;
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, 0, NULL, &valueSize));
	char* name = new char[valueSize];
	printf("Training with device %d: ",__device);
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, valueSize, name, nullptr));
	printf("%s\n",name);

   	if(__trainingType == TRAIN_AS_IS)
   		printf("Training using distribution AS IS\n");
   	else if(__trainingType == TRAIN_EQUAL_PROP)
   		printf("Training using EQUAL PROPORTIONS\n");



 	//preprocess the data, training and test
 	// if(__preprocessIndividual)
 	if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 	{
	 	if(!__trainingDataPreprocessed)
	 		//preprocessTrainingDataCollective();
	        preprocessTrainingDataIndividual();
	    if(__testData.size() != 0 && !__testDataPreprocessed)
	    	preprocessTestDataIndividual();
	}
	else if(__preprocessType == __PREPROCESS_COLLECTIVE) // __preprocessCollective
	{
		if(!__trainingDataPreprocessed)
			preprocessTrainingDataCollective();
		if(!__testDataPreprocessed)
			preprocessTestDataCollective();
	}
	else
	{
		printf("NO PREPROCESSING OF TRAINING DATA\n");
	}

	//this is for when we are doing run() on the test data
 	__isTraining = true;
	if(__testData.size() != 0)
 	{
 		__dataPointer = &__testData;
 	}

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));

 	//init layerNeeds and velocities vectors
	setupLayerNeeds(layerNeeds);
	if(__useMomentum)
		initVelocities(velocities);

}

/*****************************
*
* Net::trainSetup is called from Net::train(). Makes sure training doesn't use constant mem, finalizes if needed, 
* preprocesses data, sets up Net::run() to work on test data, sets some clKernelArgs for softmax
* Private
*
*****************************/
void Net::trainSetup()
{
	//make sure we're finalized without __constantMem
	if(!__isFinalized || __constantMem)
 	{
 		__constantMem = false;
 		if(!finalize())
 		{
 			printf("Net was unable to finalize. Aborting training.\n");
 			return;
 		}
 	}

 	//Tell which device we're using to train
 	size_t valueSize;
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, 0, NULL, &valueSize));
	char* name = new char[valueSize];
	printf("Training with device %d: ",__device);
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, valueSize, name, nullptr));
	printf("%s\n",name);

   	if(__trainingType == TRAIN_AS_IS)
   		printf("Training using distribution AS IS\n");
   	else if(__trainingType == TRAIN_EQUAL_PROP)
   		printf("Training using EQUAL PROPORTIONS\n");



 	//preprocess the data, training and test
 	// if(__preprocessIndividual)
 	if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 	{
	 	if(!__trainingDataPreprocessed)
	 		//preprocessTrainingDataCollective();
	        preprocessTrainingDataIndividual();
	    if(__testData.size() != 0 && !__testDataPreprocessed)
	    	preprocessTestDataIndividual();
	}
	else if(__preprocessType == __PREPROCESS_COLLECTIVE) // __preprocessCollective
	{
		if(!__trainingDataPreprocessed)
			preprocessTrainingDataCollective();
		if(!__testDataPreprocessed)
			preprocessTestDataCollective();
	}

	//this is for when we are doing run() on the test data
 	__isTraining = true;
	if(__testData.size() != 0)
 	{
 		__dataPointer = &__testData;
 	}

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
}

/*****************************
*
* Runs the feed forward kernels for all Layers excluding softmax.
* Expects the image to be in the prevNeurons pointer. 
* Puts the output of the function into the prevNeurons pointer.
* Layer needs should have been set up by setupLayerNeeds() which is called from trainSetup()
* Private
*
*****************************/
void Net::feedForward(vector<cl_mem>& layerNeeds)
{
	int curConvLayer = 0;
	size_t globalWorkSize[] = {(size_t)__neuronSizes[0],0,0}; // initialized for copyArrayKernel
	cl_mem* temp;
	for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
	{
		#ifdef _DEBUG
		printf("Layer %d, type %d\n", i, __layers[i]->layerType);
		#endif
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];
			if(conv->padding != 0) //if we need to do padding on the input
			{
				clSetKernelArg(zeroPadKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(zeroPadKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(zeroPadKernel, 2, sizeof(int), &(conv->padding)); // padding
				clSetKernelArg(zeroPadKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
				clSetKernelArg(zeroPadKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
				clSetKernelArg(zeroPadKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

				// run it for the size of the new array
				globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				clFinish(queue);

				//swap the buffers so prevNeurons holds the zero padded data
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}

			//save the source array into the layerNeeds for this conv layer
			clSetKernelArg(copyArrayKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(copyArrayKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
			// global work size for the copy will either have been set by the zeroPadKernel, the previous layer, or its initialization
			CheckError(clEnqueueNDRangeKernel(queue, copyArrayKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			clSetKernelArg(convKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convKernel, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
			clSetKernelArg(convKernel, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
			clSetKernelArg(convKernel, 4, sizeof(int), &(conv->numBiases)); //numFilters
			clSetKernelArg(convKernel, 5, sizeof(int), &(conv->filterSize));
			clSetKernelArg(convKernel, 6, sizeof(int), &(conv->stride));
			clSetKernelArg(convKernel, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
			clSetKernelArg(convKernel, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

			globalWorkSize[0] = (size_t)__neuronSizes[i];
			CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			curConvLayer++;
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
			clSetKernelArg(maxPoolKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(maxPoolKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(maxPoolKernel, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
			clSetKernelArg(maxPoolKernel, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
			clSetKernelArg(maxPoolKernel, 4, sizeof(int), &(pool->poolSize)); //poolsize
			clSetKernelArg(maxPoolKernel, 5, sizeof(int), &(pool->stride)); //stride
			clSetKernelArg(maxPoolKernel, 6, sizeof(cl_mem), &(layerNeeds[i])); // for maxIndexes
			globalWorkSize[0] = (size_t)__neuronSizes[i];
			CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		}
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)__layers[i])->activationType;
			globalWorkSize[0] = (size_t)__neuronSizes[i];
			if(type == RELU)
			{
				clSetKernelArg(reluKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(reluKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(reluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); //dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, reluKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				clSetKernelArg(leakyReluKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(leakyReluKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(leakyReluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}

		clFinish(queue);

		// cout << "Forward Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = neurons;
		neurons = prevNeurons;
		prevNeurons = temp;
	}
}

/*****************************
*
* Runs the forward kernels need to do the softmax function
* Output is in *neurons.
* Private
*
*****************************/
void Net::softmaxForward()
{
	size_t globalWorkSize[] = {1, 0, 0};
	clSetKernelArg(maxSubtractionKernel, 0, sizeof(cl_mem), prevNeurons);
	CheckError(clEnqueueNDRangeKernel(queue, maxSubtractionKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

	//vectorESum. arg 1 set above
	clSetKernelArg(vectorESumKernel, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(vectorESumKernel, 2, sizeof(cl_mem), &denom);
	CheckError(clEnqueueNDRangeKernel(queue, vectorESumKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);
	
	//softmax
	globalWorkSize[0] = (size_t)__neuronSizes.back();
	clSetKernelArg(softmaxKernelF, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(softmaxKernelF, 1, sizeof(cl_mem), neurons);
	clSetKernelArg(softmaxKernelF, 2, sizeof(cl_mem), &denom);
	CheckError(clEnqueueNDRangeKernel(queue, softmaxKernelF, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);
}

/*****************************
*
* Identical to feed forward above except it doesn't use the class level variables. This method was used in DETrain and AntTrain to allow 
* speedup via threads, and should be used in run() in future work.
* Private
*
*****************************/
void Net::feedForward(cl_mem** prevNeurons, cl_mem** neurons, vector<vector<int> >& __neuronDims, vector<Layer*>& __layers,
	const vector<cl_mem>& layerNeeds, const vector<cl_mem>& clWeights, const vector<cl_mem>& clBiases, const cl_command_queue& queue, const cl_mem& denom,
	const Kernels& k)
{
	// printf("start feed\n");
	int curConvLayer = 0;
	size_t globalWorkSize[] = {(size_t)__neuronSizes[0],0,0}; // initialized for copyArrayKernel
	cl_mem* temp;
	for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
	{
		//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];
			if(conv->padding != 0) //if we need to do padding on the input
			{
				clSetKernelArg(k.zeroPadKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.zeroPadKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.zeroPadKernel, 2, sizeof(int), &(conv->padding)); // padding
				clSetKernelArg(k.zeroPadKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
				clSetKernelArg(k.zeroPadKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
				clSetKernelArg(k.zeroPadKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

				// run it for the size of the new array
				globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue, k.zeroPadKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				clFinish(queue);

				//swap the buffers so prevNeurons holds the zero padded data
				temp = *neurons;
				*neurons = *prevNeurons;
				*prevNeurons = temp;
			}

			//save the source array into the layerNeeds for this conv layer
			clSetKernelArg(k.copyArrayKernel, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.copyArrayKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
			// global work size for the copy will either have been set by the zeroPadKernel, the previous layer, or its initialization
			CheckError(clEnqueueNDRangeKernel(queue, k.copyArrayKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			clSetKernelArg(k.convKernel, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.convKernel, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.convKernel, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
			clSetKernelArg(k.convKernel, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
			clSetKernelArg(k.convKernel, 4, sizeof(int), &(conv->numBiases)); //numFilters
			clSetKernelArg(k.convKernel, 5, sizeof(int), &(conv->filterSize));
			clSetKernelArg(k.convKernel, 6, sizeof(int), &(conv->stride));
			clSetKernelArg(k.convKernel, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
			clSetKernelArg(k.convKernel, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

			globalWorkSize[0] = (size_t)__neuronSizes[i];
			CheckError(clEnqueueNDRangeKernel(queue, k.convKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			curConvLayer++;
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
			clSetKernelArg(k.maxPoolKernel, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.maxPoolKernel, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.maxPoolKernel, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
			clSetKernelArg(k.maxPoolKernel, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
			clSetKernelArg(k.maxPoolKernel, 4, sizeof(int), &(pool->poolSize)); //poolsize
			clSetKernelArg(k.maxPoolKernel, 5, sizeof(int), &(pool->stride)); //stride
			clSetKernelArg(k.maxPoolKernel, 6, sizeof(cl_mem), &(layerNeeds[i])); // for maxIndexes
			globalWorkSize[0] = (size_t)__neuronSizes[i];
			CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		}
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)__layers[i])->activationType;
			globalWorkSize[0] = (size_t)__neuronSizes[i];
			if(type == RELU)
			{
				clSetKernelArg(k.reluKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.reluKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.reluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); //dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, k.reluKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				clSetKernelArg(k.leakyReluKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.leakyReluKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.leakyReluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}

		clFinish(queue);

		// cout << "Forward Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = *neurons;
		*neurons = *prevNeurons;
		*prevNeurons = temp;
	}
	if(usesSoftmax)
		softmaxForward(*prevNeurons, *neurons, queue, denom, k);
	// printf("end feed\n");
}

/*****************************
*
* Batch Normalization feed forward method for training. Assumes multiple versions running concurrently.
* Does multiple images concurrently but sequentially by layer instead of sequentially by image
* Doesn't use the class level variables (except weights and biases). 
*
* start is inclusive start index in trainingData
* end is exclusive ending index in training Data. (if start = 0 and end = 3 will do indexes 0,1,2)
*
* Private
*
*****************************/
void Net::feedForward_BN(const int num_threads, const int minibatch_size, const int thread_num, const vector<vector<double>* >& trainingData, const vector<double>& trueVals, int start, int end, vector<cl_mem*>* prevNeurons, vector<cl_mem*>* neurons,//cl_mem** prevNeurons, cl_mem** neurons, 
	const vector<vector<cl_mem> >& layerNeeds, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k, spinlock_barrier* barrier)
{
	// printf("Thread %d: doing items [%d,%d)\n", thread_num,start,end);
	//do housekeeping to make sure everything is sized right
	size_t ss = prevNeurons->size();
	int amount = end - start;
	if(ss != neurons->size() || ss != layerNeeds.size() || amount > ss) // if we have extra mem allocated that we don't use I guess that's ok
	{
		printf("Feed forward with batch normalization: Inconsistent sizes\n");
		return;
	}
	if(amount < 1)
	{
		printf("Feed forward with batch normalization: was asked to run over no data.\n");
		return;
	}

	//load in original images
	for(int i = start, j = 0; i < end; i++, j++)
	{
 		CheckError(clEnqueueWriteBuffer(queue, *(prevNeurons->at(j)), CL_FALSE, 0,
				sizeof(double) * __neuronSizes[0],
				trainingData[i]->data(), 0, nullptr, nullptr));
	}
	clFinish(queue);

	// printf("start feed\n");
	int curConvLayer = 0, curBNLayer = 0;
	size_t globalWorkSize[] = {(size_t)__neuronSizes[0],0,0}; // initialized for copyArrayKernel
	cl_mem* temp;
	#ifdef _DEBUG
	printf("Thread %d: start feed\n",thread_num);
	#endif
	for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
	{
		#ifdef _DEBUG
		printf("Thread %d: layer %d, type %d\n",thread_num,i,__layers[i]->layerType);
		#endif
		//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			BatchNormLayer *batch = (BatchNormLayer*)__layers[i];

			//need to set mu and some stuff to 0. Every thread must do its part
			int curSize = mu[curBNLayer].size(); //all these are used to split up the work for resetting and dividing mu and sigma_squared
			int mu_start = curSize / num_threads * thread_num;
			int mu_end = mu_start + curSize / num_threads;
			int mu_remainder = curSize % num_threads;
			int mu_rstart = curSize - mu_remainder;

			// printf("%d %d %d %d %d\n",curSize, mu_start,mu_end,mu_remainder,mu_rstart);
			for(int m = mu_start; m < mu_end; m++)
			{
				mu[curBNLayer][m] = 0;
				sigma_squared[curBNLayer][m] = 0;
				delta_mu[curBNLayer][m] = 0;
				delta_sigma2[curBNLayer][m] = 0;
				delta_gamma[curBNLayer][m] = 0;
				delta_beta[curBNLayer][m] = 0;
			}
			if(thread_num < mu_remainder)
			{
				int m = mu_rstart + thread_num;
				mu[curBNLayer][m] = 0;
				sigma_squared[curBNLayer][m] = 0;
				delta_mu[curBNLayer][m] = 0;
				delta_sigma2[curBNLayer][m] = 0;
				delta_gamma[curBNLayer][m] = 0;
				delta_beta[curBNLayer][m] = 0;
			}

			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif
			////////////////
			// Calculate mu on CPU
			////////////////

			//pull from GPU
			// printf("\n");
			for(int a = 0; a < amount; a++)
			{
				// assert(bn_x[thread_num][a][curBNLayer].size() == __neuronSizes[i]);
				CheckError(clEnqueueReadBuffer(queue, *(*prevNeurons)[a], CL_TRUE, 0, sizeof(double) * bn_x[thread_num][a][curBNLayer].size(), 
					bn_x[thread_num][a][curBNLayer].data(), 0, nullptr, nullptr));
			}

			//add calculate the sums for mu
			int depth = __neuronDims[i][2];
			// assert(__neuronSizes[i] % mu[curBNLayer].size() == 0);
			int zero_count = 0;
			for(int m = 0; m < __neuronSizes[i]; m++)
			{
				int k = batch->byFeatureMap ? m % depth : m;
				double sum = 0;
				for(int a = 0; a < amount; a++)
				{
					sum += bn_x[thread_num][a][curBNLayer][m];
				}
				lock_guard<mutex> guard(mtx_a[m]);
				mu[curBNLayer][k] += sum;
			}

			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif

			//do the divisions for mu
			int adjusted_minibatch_size = batch->byFeatureMap ? minibatch_size * __neuronDims[i][0] * __neuronDims[i][1] : minibatch_size;
			// printf("Adjusted size: %d\n", adjusted_minibatch_size);
			// printf("Mu: \n");
			for(int m = mu_start; m < mu_end; m++)
			{
				mu[curBNLayer][m] /= adjusted_minibatch_size;
				batch->e[m] = moveAlpha * mu[curBNLayer][m] + (1 - moveAlpha) * batch->e[m];
			}
			if(thread_num < mu_remainder)
			{
				int m = mu_rstart + thread_num;
				mu[curBNLayer][m] /= adjusted_minibatch_size;
				batch->e[m] = moveAlpha * mu[curBNLayer][m] + (1 - moveAlpha) * batch->e[m];
			}
			

			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif

			//put mu on gpu AFTER all divisions
			if(thread_num == 0)
				CheckError(clEnqueueWriteBuffer(queue, mu_cl[curBNLayer], CL_TRUE, 0, sizeof(double) * mu[curBNLayer].size(),
					mu[curBNLayer].data(), 0, nullptr, nullptr));

			////////////////
			// Calculate sigma_squared on CPU
			////////////////

			#ifdef _DEBUG
			printf("Thread %d: bn calc sigma2\n",thread_num);
			#endif

			//do sums for sigma_squared
			for(int m = 0; m < __neuronSizes[i]; m++)
			{
				int k = batch->byFeatureMap ? m % depth : m;
				double sum = 0;
				for(int a = 0; a < amount; a++)
				{
					double sigma = bn_x[thread_num][a][curBNLayer][m] - mu[curBNLayer][k];
					sum += sigma * sigma;
				}
				lock_guard<mutex> guard(mtx_a[m]);
				sigma_squared[curBNLayer][k] += sum;
			}

			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif

			//do divisions for sigma_squared
			// div = batch->byFeatureMap ? minibatch_size * __neuronDims[i][0] * __neuronDims[i][1] : minibatch_size;
			// printf("Sigma squared: \n");
			double unbiased_variance_ratio = minibatch_size / (minibatch_size - 1);
			for(int m = mu_start; m < mu_end; m++)
			{
				sigma_squared[curBNLayer][m] /= adjusted_minibatch_size;
				batch->var[m] = moveAlpha * unbiased_variance_ratio * sigma_squared[curBNLayer][m] + (1 - moveAlpha) * batch->var[m];
			}
			// printf("\n"); getchar();
			if(thread_num < mu_remainder)
			{
				int m = mu_rstart + thread_num;
				sigma_squared[curBNLayer][m] /= adjusted_minibatch_size;
				batch->var[m] = moveAlpha * unbiased_variance_ratio * sigma_squared[curBNLayer][m] + (1 - moveAlpha) * batch->var[m];
			}

			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif

			//put sigma_squared on gpu AFTER all the divisions
			if(thread_num == 0)
				CheckError(clEnqueueWriteBuffer(queue, sigma_squared_cl[curBNLayer], CL_TRUE, 0, sizeof(double) * sigma_squared[curBNLayer].size(),
					sigma_squared[curBNLayer].data(), 0, nullptr, nullptr));

			//wait to finish write to gpu before any threads can proceed
			#ifdef _DEBUG
			printf("Thread %d: pre barrier\n",thread_num);
			#endif
			barrier->count_down_and_wait();
			#ifdef _DEBUG
			printf("Thread %d: post barrier\n",thread_num);
			#endif

			////////////////
			// Calculate output of BN layer on GPU
			////////////////

			#ifdef _DEBUG
			printf("Thread %d: bn calc dneurons\n",thread_num);
			#endif

			int adjusted_depth = batch->byFeatureMap ? __neuronDims[i][2] : 0; // number < 1 means by activation instead of by feature map. use number < 1 when previous layer is non-convLayers
			for(int a = 0; a < amount; a++)
			{
				// printf("forward a: %d\n", a);
				CheckError(clSetKernelArg(k.batchNormKernel, 0, sizeof(cl_mem), prevNeurons->at(a)));
				CheckError(clSetKernelArg(k.batchNormKernel, 1, sizeof(cl_mem), neurons->at(a)));
				CheckError(clSetKernelArg(k.batchNormKernel, 2, sizeof(cl_mem), &(gamma.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormKernel, 3, sizeof(cl_mem), &(beta.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormKernel, 4, sizeof(cl_mem), &(mu_cl.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormKernel, 5, sizeof(cl_mem), &(sigma_squared_cl.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormKernel, 6, sizeof(int), &adjusted_depth));
				//other args set above
				globalWorkSize[0] = (size_t) __neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.batchNormKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				clFinish(queue);
			}

			// getchar();

			//pull y from gpu to calc xhat for storage. Need current gamma and beta pulled in batchNormTrain(do differently later?)
			for(int a = 0; a < amount; a++)
			{
				CheckError(clEnqueueReadBuffer(queue, *(*neurons)[a], CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
					bn_xhat[thread_num][a][curBNLayer].data(), 0, nullptr, nullptr));
				for(int f = 0; f < __neuronSizes[i]; f++)
				{
					int k = batch->byFeatureMap ? f % depth : f;
					bn_xhat[thread_num][a][curBNLayer][f] = (bn_xhat[thread_num][a][curBNLayer][f] - batch->beta[k]) / batch->gamma[k];
				}

				temp = (*neurons)[a];
				(*neurons)[a] = (*prevNeurons)[a];
				(*prevNeurons)[a] = temp;
			}

			curBNLayer++;

			#ifdef _DEBUG
			printf("Thread %d: END bn\n",thread_num);
			#endif
		}
		else
		{
			//if not a batch norm layer, threads run asynchronously
			for(int a = 0; a < amount; a++)
			{
				#ifdef _DEBUG
				printf("Thread %d: a %d of %d. pn: %lu, n: %lu, ln: %lu \n",thread_num,a,amount,prevNeurons->size(), neurons->size(), layerNeeds.size());
				#endif
				if(__layers[i]->layerType == CONV_LAYER)
				{
					ConvLayer* conv = (ConvLayer*)__layers[i];
					if(conv->padding != 0) //if we need to do padding on the input
					{
						#ifdef _DEBUG
						printf("Thread %d: starting padding...\n",thread_num);
						#endif
						clSetKernelArg(k.zeroPadKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
						clSetKernelArg(k.zeroPadKernel, 1, sizeof(cl_mem), (*neurons)[a]);
						clSetKernelArg(k.zeroPadKernel, 2, sizeof(int), &(conv->padding)); // padding
						clSetKernelArg(k.zeroPadKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
						clSetKernelArg(k.zeroPadKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
						clSetKernelArg(k.zeroPadKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

						// run it for the size of the new array
						globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
						CheckError(clEnqueueNDRangeKernel(queue, k.zeroPadKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
						clFinish(queue);

						//swap the buffers so prevNeurons holds the zero padded data
						temp = (*neurons)[a];
						(*neurons)[a] = (*prevNeurons)[a];
						(*prevNeurons)[a] = temp;
						#ifdef _DEBUG
						printf("Thread %d: padding done\n",thread_num);
						#endif
					}

					#ifdef _DEBUG
					printf("Thread %d: Copy layerNeeds\n",thread_num);
					#endif
					//save the source array into the layerNeeds for this conv layer
					clSetKernelArg(k.copyArrayKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
					clSetKernelArg(k.copyArrayKernel, 1, sizeof(cl_mem), &(layerNeeds[a][i]));
					// global work size for the copy will either have been set by the zeroPadKernel, the previous layer, or its initialization
					CheckError(clEnqueueNDRangeKernel(queue, k.copyArrayKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

					clSetKernelArg(k.convKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
					clSetKernelArg(k.convKernel, 1, sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.convKernel, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
					clSetKernelArg(k.convKernel, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
					clSetKernelArg(k.convKernel, 4, sizeof(int), &(conv->numBiases)); //numFilters
					clSetKernelArg(k.convKernel, 5, sizeof(int), &(conv->filterSize));
					clSetKernelArg(k.convKernel, 6, sizeof(int), &(conv->stride));
					clSetKernelArg(k.convKernel, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
					clSetKernelArg(k.convKernel, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, k.convKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

					#ifdef _DEBUG
					printf("Thread %d: End conv layer\n",thread_num);
					#endif
					if(a == amount - 1)
						curConvLayer++;
				}
				else if(__layers[i]->layerType == MAX_POOL_LAYER)
				{
					MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
					clSetKernelArg(k.maxPoolKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
					clSetKernelArg(k.maxPoolKernel, 1, sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.maxPoolKernel, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
					clSetKernelArg(k.maxPoolKernel, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
					clSetKernelArg(k.maxPoolKernel, 4, sizeof(int), &(pool->poolSize)); //poolsize
					clSetKernelArg(k.maxPoolKernel, 5, sizeof(int), &(pool->stride)); //stride
					clSetKernelArg(k.maxPoolKernel, 6, sizeof(cl_mem), &(layerNeeds[a][i])); // for maxIndexes
					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else if(__layers[i]->layerType == ACTIV_LAYER)
				{
					#ifdef _DEBUG
					printf("Thread %d.%d: Start ActivLayer \n",thread_num,a);
					#endif
					int type = ((ActivLayer*)__layers[i])->activationType;
					globalWorkSize[0] = (size_t)__neuronSizes[i];
					if(type == RELU)
					{
						#ifdef _DEBUG
						printf("Thread %d.%d: RELU... \n",thread_num,a);
						#endif
						clSetKernelArg(k.reluKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
						clSetKernelArg(k.reluKernel, 1, sizeof(cl_mem), (*neurons)[a]);
						clSetKernelArg(k.reluKernel, 2, sizeof(cl_mem), &(layerNeeds[a][i])); //dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, k.reluKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
					else if(type == LEAKY_RELU)
					{
						#ifdef _DEBUG
						printf("Thread %d.%d: LEAKY_RELU... \n",thread_num,a);
						#endif
						CheckError(clSetKernelArg(k.leakyReluKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]));
						CheckError(clSetKernelArg(k.leakyReluKernel, 1, sizeof(cl_mem), (*neurons)[a]));
						CheckError(clSetKernelArg(k.leakyReluKernel, 2, sizeof(cl_mem), &(layerNeeds[a][i]))); // dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
				}
				clFinish(queue);
				temp = (*neurons)[a];
				(*neurons)[a] = (*prevNeurons)[a];
				(*prevNeurons)[a] = temp;
			}
			
		}
		

		

		// cout << "Forward Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

	}
	int numCorrect = 0;
	int numzeros = 0;
	vector<int> myclasscorrect(__neuronSizes.back(),0);
	vector<int> myclasstotal(__neuronSizes.back(),0);
	vector<double> soft(__neuronSizes.back());
	double error = 0;

	#ifdef NET_SHOW_ERRORS
	static int calls = 0;
	{
		lock_guard<mutex> guard(show_error_mtx);
		calls++;
	}
	#endif
	for(int a = 0; a < amount; a++)
	{
		// CheckError(clEnqueueReadBuffer(queue, *(*neurons)[a], CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		//  	soft.data(), 0, nullptr, nullptr));
		// stringstream ss("");
		// for(int s = 0; s < __neuronSizes.back(); s++)
		// 	ss << soft[s] << ", ";
		// ss << endl;
		// printf("pre-soft: %s\n", ss.str().c_str());

		#ifdef _DEBUG
		printf("Thread %d.%d: start softmax... \n",thread_num,a);
		#endif
		if(usesSoftmax)
			softmaxForward((*prevNeurons)[a], (*neurons)[a], queue, denom, k);
		CheckError(clEnqueueReadBuffer(queue, *(*neurons)[a], CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	soft.data(), 0, nullptr, nullptr));
		int predict = getMaxElementIndex(soft);
		int actual = trueVals[start + a];
		if(predict == actual)
		{
			numCorrect++;
			myclasscorrect[predict]++;
		}
		#ifdef NET_SHOW_ERRORS
		else
		{
			lock_guard<mutex> guard(show_error_mtx);
			if(calls > 3 * num_threads * (trainingData.size() / minibatch_size))
			{
				Mat show(__neuronDims[0][0],__neuronDims[0][1],CV_8UC3);
				int k = 0;
				for(int i = 0; i < __neuronDims[0][0]; i++)
					for(int j = 0; j < __neuronDims[0][1]; j++)
					{
						Vec3b& outPix = show.at<Vec3b>(i,j);
						outPix[0] = (unsigned char)(trainingData[start + a]->at(k++) * __stddev + __mean);
						outPix[1] = (unsigned char)(trainingData[start + a]->at(k++) * __stddev + __mean);
						outPix[2] = (unsigned char)(trainingData[start + a]->at(k++) * __stddev + __mean);
						// printf("pix - %d %d %d\n",outPix[0],outPix[1],outPix[2]);
					}

				char name[10];
				sprintf(name,"%d",(int)trueVals[start + a]);
				printf("name %s\n", name);
				cv::Size siz(50,50);
				// resize(show,show,siz);
				// imshow("name",show);
				imwrite("wrongImage.png",show);
				// waitKey(0);
				getchar();
			}
		}
		#endif
		myclasstotal[actual]++;

		// printf("im %d: soft %lf %lf %lf - actual %d, predict %d\n", start + a, soft[0], soft[1], soft[2], actual, predict);

		error -= log(soft[actual]);

		if(isnan(log(soft[actual])))
		{
			printf("BAD NAN!\n");
			getchar();
		}

		// if(predict == 0)
		// 	numzeros++;

		// ss.str(string());
		// for(int s = 0; s < __neuronSizes.back(); s++)
		// 	ss << soft[s] << ", ";
		// ss << endl;
		// printf("soft: %s\n", ss.str().c_str());

	}
	// printf("Thread %d: %d of %d correct\n", thread_num,numCorrect,amount);
	bnNumCorrect_mtx.lock();
	bnTotalError += error;
	bnNumCorrect += numCorrect;
	bnNumZeros   += numzeros;
	for(int i = 0; i < __neuronSizes.back(); i++)
	{
		bnClassCorrect[i] += myclasscorrect[i];
		bnClassTotal[i] += myclasstotal[i];
	}
	bnNumCorrect_mtx.unlock();
	// printf("end feed\n");
}

/*****************************
*
* Batch Normalization feed forward method for running. Assumes multiple versions running concurrently.
* Does multiple images sequentially by image
* Doesn't use the class level variables (except weights and biases). 
*
* start is inclusive start index in trainingData
* end is exclusive ending index in training Data. (if start = 0 and end = 3 will do indexes 0,1,2)
*
* Private
*
*****************************/
void Net::feedForward_BN_running(const int num_threads, const int minibatch_size, const int thread_num, int start, int end, vector<vector<double> >* __dataPointer, cl_mem** prevNeurons, cl_mem** neurons,
	 const cl_command_queue& queue, const cl_mem& denom, const Kernels& k)
{
	// printf("Thread %d: doing items [%d,%d)\n", thread_num,start,end);
	//do housekeeping to make sure everything is sized right
	int amount = end - start;
	if(amount < 1)
	{
		printf("Feed forward with batch normalization: was asked to run over no data.\n");
		return;
	}

	size_t globalWorkSize[] = {(size_t)__neuronSizes[0],0,0}; // initialized for copyArrayKernel
	cl_mem* temp;

	for(int a = 0; a < amount; a++)
	{
		//load in original images
		#ifdef _DEBUG
		printf("Thread %d.%d: start write image\n",thread_num,a);
		#endif
 		CheckError(clEnqueueWriteBuffer(queue, *(*prevNeurons), CL_TRUE, 0,
				sizeof(double) * __neuronSizes[0],
				__dataPointer->at(start + a).data(), 0, nullptr, nullptr));

 		// printf("image %d: ", start + a);
 		// for(int v = 0; v < __dataPointer->at(start + a).size(); v++)
 		// 	printf("%lf, ", __dataPointer->at(start + a)[v]);
 		// printf("\n");

		int curConvLayer = 0, curBNLayer = 0;
		
		#ifdef _DEBUG
		printf("Thread %d.%d: start feed\n",thread_num,a);
		#endif
		for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
		{
			#ifdef _DEBUG
			printf("Thread %d.%d: layer %d, type %d\n",thread_num,a,i,__layers[i]->layerType);
			#endif
			//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
			if(__layers[i]->layerType == BATCH_NORM_LAYER)
			{
				BatchNormLayer *batch = (BatchNormLayer*)__layers[i];

				////////////////
				// Calculate output of BN layer on GPU
				////////////////

				#ifdef _DEBUG
				printf("Thread %d: bn calc dneurons\n",thread_num);
				#endif


				int depth = batch->byFeatureMap ? __neuronDims[i][2] : 0; // number < 1 means by activation instead of by feature map. use number < 1 when previous layer is non-convLayers
				CheckError(clSetKernelArg(k.batchNormRunKernel, 0, sizeof(cl_mem), *prevNeurons));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 1, sizeof(cl_mem), *neurons));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 2, sizeof(cl_mem), &gamma.at(curBNLayer)));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 3, sizeof(cl_mem), &beta.at(curBNLayer)));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 4, sizeof(cl_mem), &(mu_cl.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 5, sizeof(cl_mem), &(sigma_squared_cl.at(curBNLayer))));
				CheckError(clSetKernelArg(k.batchNormRunKernel, 6, sizeof(int), &depth));
				//other args set above
				globalWorkSize[0] = (size_t) __neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.batchNormRunKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

				curBNLayer++;

				#ifdef _DEBUG
				printf("Thread %d: END bn\n",thread_num);
				#endif
			}
			else if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)__layers[i];
				if(conv->padding != 0) //if we need to do padding on the input
				{
					#ifdef _DEBUG
					printf("Thread %d: starting padding...\n",thread_num);
					#endif
					clSetKernelArg(k.zeroPadKernelF, 0, sizeof(cl_mem), *prevNeurons);
					clSetKernelArg(k.zeroPadKernelF, 1, sizeof(cl_mem), *neurons);
					clSetKernelArg(k.zeroPadKernelF, 2, sizeof(int), &(conv->padding)); // padding
					clSetKernelArg(k.zeroPadKernelF, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
					clSetKernelArg(k.zeroPadKernelF, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
					clSetKernelArg(k.zeroPadKernelF, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

					// run it for the size of the new array
					globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
					CheckError(clEnqueueNDRangeKernel(queue, k.zeroPadKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					clFinish(queue);

					//swap the buffers so prevNeurons holds the zero padded data
					// printf("pad swap\n");
					temp = *neurons;
					*neurons = *prevNeurons;
					*prevNeurons = temp;
					#ifdef _DEBUG
					printf("Thread %d: padding done\n",thread_num);
					#endif
				}

				#ifdef _DEBUG
				printf("Thread %d: Copy layerNeeds\n",thread_num);
				#endif

				clSetKernelArg(k.convKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.convKernelF, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.convKernelF, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(k.convKernelF, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
				clSetKernelArg(k.convKernelF, 4, sizeof(int), &(conv->numBiases)); //numFilters
				clSetKernelArg(k.convKernelF, 5, sizeof(int), &(conv->filterSize));
				clSetKernelArg(k.convKernelF, 6, sizeof(int), &(conv->stride));
				clSetKernelArg(k.convKernelF, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
				clSetKernelArg(k.convKernelF, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.convKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

				#ifdef _DEBUG
				printf("Thread %d: End conv layer\n",thread_num);
				#endif
				curConvLayer++;
			}
			else if(__layers[i]->layerType == MAX_POOL_LAYER)
			{
				MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
				clSetKernelArg(k.maxPoolKernelF, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.maxPoolKernelF, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.maxPoolKernelF, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
				clSetKernelArg(k.maxPoolKernelF, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
				clSetKernelArg(k.maxPoolKernelF, 4, sizeof(int), &(pool->poolSize)); //poolsize
				clSetKernelArg(k.maxPoolKernelF, 5, sizeof(int), &(pool->stride)); //stride
				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(__layers[i]->layerType == ACTIV_LAYER)
			{
				#ifdef _DEBUG
				printf("Thread %d.%d: Start ActivLayer \n",thread_num,a);
				#endif
				int type = ((ActivLayer*)__layers[i])->activationType;
				globalWorkSize[0] = (size_t)__neuronSizes[i];
				if(type == RELU)
				{
					#ifdef _DEBUG
					printf("Thread %d.%d: RELU... \n",thread_num,a);
					#endif
					clSetKernelArg(k.reluKernelF, 0, sizeof(cl_mem), *prevNeurons);
					clSetKernelArg(k.reluKernelF, 1, sizeof(cl_mem), *neurons);
					CheckError(clEnqueueNDRangeKernel(queue, k.reluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else if(type == LEAKY_RELU)
				{
					#ifdef _DEBUG
					printf("Thread %d.%d: LEAKY_RELU... \n",thread_num,a);
					#endif
					CheckError(clSetKernelArg(k.leakyReluKernelF, 0, sizeof(cl_mem), *prevNeurons));
					CheckError(clSetKernelArg(k.leakyReluKernelF, 1, sizeof(cl_mem), *neurons));
					CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
			}
			
			clFinish(queue);

			// printf("swap\n");
			temp = *neurons;
			*neurons = *prevNeurons;
			*prevNeurons = temp;
			
		}

		#ifdef _DEBUG
		printf("Thread %d.%d: start softmax... \n",thread_num,a);
		#endif
		if(usesSoftmax)
			softmaxForward(*prevNeurons, *neurons, queue, denom, k);
		__confidences[start + a].resize(__neuronSizes.back());
		CheckError(clEnqueueReadBuffer(queue, **neurons, CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	__confidences[start + a].data(), 0, nullptr, nullptr));

		stringstream ss;
		for(int c = 0; c < __confidences[start + a].size(); c++)
			ss << __confidences[start + a][c] << ", ";
		// printf("%d.%d: conf %s\n", thread_num, a, ss.str().c_str());


	}
}


/*****************************
*
* Identical to softmaxForward above but does not use class level variables.
* Private
*
*****************************/
void Net::softmaxForward(cl_mem* prevNeurons, cl_mem* neurons, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k)
{
		// vector<double> soft(__neuronSizes.back());
		// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		// 		soft.data(), 0, nullptr, nullptr));
		// clFinish(queue);
		// //print soft
		// cout << "Softmax before" << endl;
		// for(int s = 0; s < soft.size(); s++)
		//     cout << "| " << soft[s] << " ";
		// cout << "|\n";

	size_t globalWorkSize[] = {1, 0, 0};
	clSetKernelArg(k.maxSubtractionKernel, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(k.maxSubtractionKernel, 1, sizeof(int), &__neuronSizes.back());
	CheckError(clEnqueueNDRangeKernel(queue, k.maxSubtractionKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

		// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
	 // 		soft.data(), 0, nullptr, nullptr));
		// clFinish(queue);
		// //print soft
		// cout << "Softmax max sub" << endl;
		// for(int s = 0; s < soft.size(); s++)
		//     cout << "| " << soft[s] << " ";
		// cout << "|\n";

	//vectorESum. arg 1 set above
	clSetKernelArg(k.vectorESumKernel, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(k.vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
	clSetKernelArg(k.vectorESumKernel, 2, sizeof(cl_mem), &denom);
	CheckError(clEnqueueNDRangeKernel(queue, k.vectorESumKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

		// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
	 // 		soft.data(), 0, nullptr, nullptr));
		// clFinish(queue);
		// //print soft
		// cout << "Softmax vec e sum" << endl;
		// for(int s = 0; s < soft.size(); s++)
		//     cout << "| " << soft[s] << " ";
		// cout << "|\n";
	
	//softmax
	globalWorkSize[0] = (size_t)__neuronSizes.back();
	clSetKernelArg(k.softmaxKernelF, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(k.softmaxKernelF, 1, sizeof(cl_mem), neurons);
	clSetKernelArg(k.softmaxKernelF, 2, sizeof(cl_mem), &denom);
	CheckError(clEnqueueNDRangeKernel(queue, k.softmaxKernelF, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
	 // 		soft.data(), 0, nullptr, nullptr));
		// clFinish(queue);
		// //print soft
		// cout << "Softmax done" << endl;
		// for(int s = 0; s < soft.size(); s++)
		//     cout << "| " << soft[s] << " ";
		// cout << "|\n";
}

/*****************************
*
* Runs the backprop kernels for softmax
* Expects the output of softmax to be in *neurons
* Puts the output of the backprop in *neurons
* Private
*
*****************************/
void Net::softmaxBackprop(int curTrueVal)
{
	// int curTrueVal = trueVals[r];
	clSetKernelArg(softmaxBackKernel, 0, sizeof(cl_mem), prevNeurons);
	clSetKernelArg(softmaxBackKernel, 1, sizeof(cl_mem), neurons);
	clSetKernelArg(softmaxBackKernel, 2, sizeof(int), &curTrueVal);
	size_t globalWorkSize[] = {(size_t)__neuronSizes.back(), 0, 0};
	CheckError(clEnqueueNDRangeKernel(queue,softmaxBackKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
//  	soft.data(), 0, nullptr, nullptr));
//       clFinish(queue);
//       //print soft
//       // cout << "Softmax back" << endl;
//       for(int s = 0; s < soft.size(); s++)
//           cout << "| " << soft[s] << " ";
//       cout << "|\n";

	cl_mem* temp = neurons;
	neurons = prevNeurons;
	prevNeurons = temp;

}


/*****************************
*
* Runs Backprop kernels for all layers excluding softmax
* Expects output of softmaxBackprop to be in *neurons.
* Does weight updates
* Private
*
*****************************/
void Net::backprop(vector<cl_mem>& layerNeeds, vector<cl_mem>& velocities)
{
	int curConvLayer = clWeights.size() - 1;
	size_t globalWorkSize[] = {0, 0, 0};
	cl_mem* temp;
	for(int i = __layers.size() -1; i > 0; i--)
	{
		// printf("layer %d\n", i);
		if(__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)__layers[i])->activationType;
			globalWorkSize[0] = (size_t)__neuronSizes[i-1];
			if(type == RELU)
			{
				// cout << "running reluBackKernel " << endl;
				clSetKernelArg(reluBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(reluBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, reluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				// cout << "running leakyReluBackKernel " << endl;
				clSetKernelArg(leakyReluBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(leakyReluBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, leakyReluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			// cout << "running maxPoolBackKernel " << endl;
			int numIndexes = __neuronSizes[i];
			clSetKernelArg(maxPoolBackKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(maxPoolBackKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i]));
			clSetKernelArg(maxPoolBackKernel, 3, sizeof(int), &numIndexes);
			clSetKernelArg(maxPoolBackKernel, 4, sizeof(int), &(__neuronDims[i][2]));
			// cout << "args set" << endl;
			globalWorkSize[0] = (size_t)__neuronSizes[i-1];
			// printf("globalWorkSize %lu\n", globalWorkSize[0]);
			CheckError(clEnqueueNDRangeKernel(queue, maxPoolBackKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			// cout << "kernel started" << endl;
		}
		else if(__layers[i]->layerType == CONV_LAYER)
		{
			//backprop neurons					
			// cout << "running convBackNeuronsKernel" << endl;
			ConvLayer* conv = (ConvLayer*)__layers[i];
			clSetKernelArg(convBackNeuronsKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(convBackNeuronsKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convBackNeuronsKernel, 2, sizeof(cl_mem), &(clWeights[curConvLayer]));
			clSetKernelArg(convBackNeuronsKernel, 3, sizeof(int), &(conv->numBiases)); //numFilters
			clSetKernelArg(convBackNeuronsKernel, 4, sizeof(int), &(conv->filterSize)); //filterSize
			clSetKernelArg(convBackNeuronsKernel, 5, sizeof(int), &(conv->stride)); //stride
			clSetKernelArg(convBackNeuronsKernel, 6, sizeof(int), &(conv->paddedNeuronWidth));
			clSetKernelArg(convBackNeuronsKernel, 7, sizeof(int), &(__neuronDims[i-1][2])); //depth

			globalWorkSize[0] = (size_t) conv->paddedNeuronSize;					
			CheckError(clEnqueueNDRangeKernel(queue, convBackNeuronsKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			clFinish(queue); //MUST finish before weights start getting updated

			//backprop and update biases
			// cout << "running convBackBiasesKernel " << endl;
			clSetKernelArg(convBackBiasesKernel, 0, sizeof(cl_mem), &(clBiases[curConvLayer]));
			clSetKernelArg(convBackBiasesKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convBackBiasesKernel, 2, sizeof(int), &(__neuronSizes[i]));
			clSetKernelArg(convBackBiasesKernel, 3, sizeof(int), &(conv->numBiases)); // numFilters = dneuronsDepth
			clSetKernelArg(convBackBiasesKernel, 4, sizeof(double), &(__learningRate));
			globalWorkSize[0] = (size_t) conv->numBiases;
			CheckError(clEnqueueNDRangeKernel(queue, convBackBiasesKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		// clFinish(queue);
		// cout << "ConvLayer back biases " << curConvLayer << endl;
		// CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
		// 	test.data(), 0, nullptr, nullptr));
		// printArray(test.data(), globalWorkSize[0]);
		// getchar();

			//backprop and update weights					
			// cout << "running convBackWeightsKernel" << endl;
			if(!__useMomentum) // no momentum
			{
				clSetKernelArg(convBackWeightsKernel, 0, sizeof(cl_mem), &(clWeights[curConvLayer]));
				clSetKernelArg(convBackWeightsKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
				clSetKernelArg(convBackWeightsKernel, 2, sizeof(cl_mem), neurons);
				clSetKernelArg(convBackWeightsKernel, 3, sizeof(int), &(__neuronDims[i-1][2])); // depth
				clSetKernelArg(convBackWeightsKernel, 4, sizeof(int), &(conv->stride)); // stride
				clSetKernelArg(convBackWeightsKernel, 5, sizeof(int), &(conv->paddedNeuronWidth));
				clSetKernelArg(convBackWeightsKernel, 6, sizeof(int), &(conv->filterSize)); // filterSize
				clSetKernelArg(convBackWeightsKernel, 7, sizeof(int), &(conv->numBiases)); // numFilters
				clSetKernelArg(convBackWeightsKernel, 8, sizeof(double), &__learningRate);
				globalWorkSize[0] = (size_t)conv->numWeights;
				CheckError(clEnqueueNDRangeKernel(queue, convBackWeightsKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else // with momentum
			{
				clSetKernelArg(convBackWeightsMomentKernel, 0, sizeof(cl_mem), &(clWeights[curConvLayer]));
				clSetKernelArg(convBackWeightsMomentKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
				clSetKernelArg(convBackWeightsMomentKernel, 2, sizeof(cl_mem), neurons);
				clSetKernelArg(convBackWeightsMomentKernel, 3, sizeof(int), &(__neuronDims[i-1][2])); // depth
				clSetKernelArg(convBackWeightsMomentKernel, 4, sizeof(int), &(conv->stride)); // stride
				clSetKernelArg(convBackWeightsMomentKernel, 5, sizeof(int), &(conv->paddedNeuronWidth));
				clSetKernelArg(convBackWeightsMomentKernel, 6, sizeof(int), &(conv->filterSize)); // filterSize
				clSetKernelArg(convBackWeightsMomentKernel, 7, sizeof(int), &(conv->numBiases)); // numFilters
				clSetKernelArg(convBackWeightsMomentKernel, 8, sizeof(double), &__learningRate);
				clSetKernelArg(convBackWeightsMomentKernel, 9, sizeof(cl_mem), &(velocities[curConvLayer]));
				globalWorkSize[0] = (size_t)conv->numWeights;
				CheckError(clEnqueueNDRangeKernel(queue, convBackWeightsMomentKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}

	// clFinish(queue);
	// cout << "ConvLayer back weights " << curConvLayer << endl;
	// cout << "numWeights " << conv->numWeights << endl;
	// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
	// 	test.data(), 0, nullptr, nullptr));
	// printArray(test.data(), conv->numWeights);
	// getchar();

			//backprop zeroPad if necessary
			if(conv->padding != 0) 
			{
				clFinish(queue); //so the weights finish updating before zeroPadBackKernel starts changing prevNeurons and neurons

				//swap prev and cur neuron pointers
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;

				clSetKernelArg(zeroPadBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(zeroPadBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(zeroPadBackKernel, 2, sizeof(int), &(conv->padding)); //padding
				clSetKernelArg(zeroPadBackKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); //prevWidth (non-padded)
				clSetKernelArg(zeroPadBackKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); //prevHeight(non-padded)
				clSetKernelArg(zeroPadBackKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); //depth
				// cout << "Running zeroPadBackKernel" << endl;
				globalWorkSize[0] = (size_t)conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue,zeroPadBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			curConvLayer--;
		} // end if-elseif for backprop of single layer

		clFinish(queue);
		// cout << "post finish" << endl;

		// cout << "Backprop Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = neurons;
		neurons = prevNeurons;
		prevNeurons = temp;
	}// end for loop for backprop
}


/*****************************
*
* Same as softmaxBackprop above but doesn't use class level variables
* Expects softmax output in *neurons
* Puts output into *neurons
* Private
*
*****************************/
void Net::softmaxBackprop(int curTrueVal, cl_mem** prevNeurons, cl_mem** neurons, const cl_command_queue& queue, const Kernels& k, Net* net)
{
	// int curTrueVal = trueVals[r];
	clSetKernelArg(k.softmaxBackKernel, 0, sizeof(cl_mem), *prevNeurons);
	clSetKernelArg(k.softmaxBackKernel, 1, sizeof(cl_mem), *neurons);
	clSetKernelArg(k.softmaxBackKernel, 2, sizeof(int), &curTrueVal);
	size_t globalWorkSize[] = {(size_t)net->__neuronSizes.back(), 0, 0};
	CheckError(clEnqueueNDRangeKernel(queue,k.softmaxBackKernel, 1,
		nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
	clFinish(queue);

// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
//  	soft.data(), 0, nullptr, nullptr));
//       clFinish(queue);
//       //print soft
//       // cout << "Softmax back" << endl;
//       for(int s = 0; s < soft.size(); s++)
//           cout << "| " << soft[s] << " ";
//       cout << "|\n";

	cl_mem* temp = *neurons;
	*neurons = *prevNeurons;
	*prevNeurons = temp;

}


/*****************************
*
* Same as backprop above but doesn't use class level variables
* Expects output from softmax forwared output in *neurons
* Updates Weights and runs all the kernels for backprop (including softmax)
* Private
*
*****************************/
void Net::backprop(int curTrueVal, cl_mem** prevNeurons, cl_mem** neurons, Net* net, const vector<cl_mem>& layerNeeds, const vector<cl_mem>& velocities, 
	const vector<cl_mem>& clWeights, const vector<cl_mem>& clBiases, const cl_command_queue queue, const Kernels& k)
{
	if(usesSoftmax)
		softmaxBackprop(curTrueVal, prevNeurons, neurons, queue, k, net);

	int curConvLayer = clWeights.size() - 1;
	size_t globalWorkSize[] = {0, 0, 0};
	cl_mem* temp;
	for(int i = net->__layers.size() -1; i > 0; i--)
	{
		if(net->__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)net->__layers[i])->activationType;
			globalWorkSize[0] = (size_t)net->__neuronSizes[i-1];
			if(type == RELU)
			{
				//cout << "running reluBackKernel " << endl;
				clSetKernelArg(k.reluBackKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.reluBackKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, k.reluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				//cout << "running leakyReluBackKernel " << endl;
				clSetKernelArg(k.leakyReluBackKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.leakyReluBackKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}
		else if(net->__layers[i]->layerType == MAX_POOL_LAYER)
		{
			//cout << "running maxPoolBackKernel " << endl;
			int numIndexes = net->__neuronSizes[i];
			clSetKernelArg(k.maxPoolBackKernel, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.maxPoolBackKernel, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i]));
			clSetKernelArg(k.maxPoolBackKernel, 3, sizeof(int), &numIndexes);
			clSetKernelArg(k.maxPoolBackKernel, 4, sizeof(int), &(net->__neuronDims[i][2]));
			globalWorkSize[0] = (size_t)net->__neuronSizes[i-1];
			CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolBackKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		}
		else if(net->__layers[i]->layerType == CONV_LAYER)
		{
			//backprop neurons					
			//cout << "running convBackNeuronsKernel" << endl;
			ConvLayer* conv = (ConvLayer*)net->__layers[i];
			clSetKernelArg(k.convBackNeuronsKernel, 0, sizeof(cl_mem), *prevNeurons);
			clSetKernelArg(k.convBackNeuronsKernel, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.convBackNeuronsKernel, 2, sizeof(cl_mem), &(clWeights[curConvLayer]));
			clSetKernelArg(k.convBackNeuronsKernel, 3, sizeof(int), &(conv->numBiases)); //numFilters
			clSetKernelArg(k.convBackNeuronsKernel, 4, sizeof(int), &(conv->filterSize)); //filterSize
			clSetKernelArg(k.convBackNeuronsKernel, 5, sizeof(int), &(conv->stride)); //stride
			clSetKernelArg(k.convBackNeuronsKernel, 6, sizeof(int), &(conv->paddedNeuronWidth));
			clSetKernelArg(k.convBackNeuronsKernel, 7, sizeof(int), &(net->__neuronDims[i-1][2])); //depth

			globalWorkSize[0] = (size_t) conv->paddedNeuronSize;					
			CheckError(clEnqueueNDRangeKernel(queue, k.convBackNeuronsKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			clFinish(queue); //MUST finish before weights start getting updated

			//backprop and update biases
			//cout << "running convBackBiasesKernel " << curConvLayer << ": " << sizeOfNeurons << "/" << hyper[4] << endl;
			clSetKernelArg(k.convBackBiasesKernel, 0, sizeof(cl_mem), &(clBiases[curConvLayer]));
			clSetKernelArg(k.convBackBiasesKernel, 1, sizeof(cl_mem), *neurons);
			clSetKernelArg(k.convBackBiasesKernel, 2, sizeof(int), &(net->__neuronSizes[i]));
			clSetKernelArg(k.convBackBiasesKernel, 3, sizeof(int), &(conv->numBiases)); // numFilters = dneuronsDepth
			clSetKernelArg(k.convBackBiasesKernel, 4, sizeof(double), &(__learningRate));
			globalWorkSize[0] = (size_t) conv->numBiases;
			CheckError(clEnqueueNDRangeKernel(queue, k.convBackBiasesKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		// clFinish(queue);
		// cout << "ConvLayer back biases " << curConvLayer << endl;
		// CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
		// 	test.data(), 0, nullptr, nullptr));
		// printArray(test.data(), globalWorkSize[0]);
		// getchar();

			//backprop and update weights					
			//cout << "running convBackWeightsKernel" << endl;
			if(!__useMomentum) // no momentum
			{
				clSetKernelArg(k.convBackWeightsKernel, 0, sizeof(cl_mem), &(clWeights[curConvLayer]));
				clSetKernelArg(k.convBackWeightsKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
				clSetKernelArg(k.convBackWeightsKernel, 2, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.convBackWeightsKernel, 3, sizeof(int), &(net->__neuronDims[i-1][2])); // depth
				clSetKernelArg(k.convBackWeightsKernel, 4, sizeof(int), &(conv->stride)); // stride
				clSetKernelArg(k.convBackWeightsKernel, 5, sizeof(int), &(conv->paddedNeuronWidth));
				clSetKernelArg(k.convBackWeightsKernel, 6, sizeof(int), &(conv->filterSize)); // filterSize
				clSetKernelArg(k.convBackWeightsKernel, 7, sizeof(int), &(conv->numBiases)); // numFilters
				clSetKernelArg(k.convBackWeightsKernel, 8, sizeof(double), &__learningRate);
				globalWorkSize[0] = (size_t)conv->numWeights;
				CheckError(clEnqueueNDRangeKernel(queue, k.convBackWeightsKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else // with momentum
			{
				clSetKernelArg(k.convBackWeightsMomentKernel, 0, sizeof(cl_mem), &(clWeights[curConvLayer]));
				clSetKernelArg(k.convBackWeightsMomentKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
				clSetKernelArg(k.convBackWeightsMomentKernel, 2, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.convBackWeightsMomentKernel, 3, sizeof(int), &(net->__neuronDims[i-1][2])); // depth
				clSetKernelArg(k.convBackWeightsMomentKernel, 4, sizeof(int), &(conv->stride)); // stride
				clSetKernelArg(k.convBackWeightsMomentKernel, 5, sizeof(int), &(conv->paddedNeuronWidth));
				clSetKernelArg(k.convBackWeightsMomentKernel, 6, sizeof(int), &(conv->filterSize)); // filterSize
				clSetKernelArg(k.convBackWeightsMomentKernel, 7, sizeof(int), &(conv->numBiases)); // numFilters
				clSetKernelArg(k.convBackWeightsMomentKernel, 8, sizeof(double), &__learningRate);
				clSetKernelArg(k.convBackWeightsMomentKernel, 9, sizeof(cl_mem), &(velocities[curConvLayer]));
				globalWorkSize[0] = (size_t)conv->numWeights;
				CheckError(clEnqueueNDRangeKernel(queue, k.convBackWeightsMomentKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}

	// clFinish(queue);
	// cout << "ConvLayer back weights " << curConvLayer << endl;
	// cout << "numWeights " << conv->numWeights << endl;
	// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
	// 	test.data(), 0, nullptr, nullptr));
	// printArray(test.data(), conv->numWeights);
	// getchar();

			//backprop zeroPad if necessary
			if(conv->padding != 0) 
			{
				clFinish(queue); //so the weights finish updating before zeroPadBackKernel starts changing prevNeurons and neurons

				//swap prev and cur neuron pointers
				temp = *neurons;
				*neurons = *prevNeurons;
				*prevNeurons = temp;

				clSetKernelArg(k.zeroPadBackKernel, 0, sizeof(cl_mem), *prevNeurons);
				clSetKernelArg(k.zeroPadBackKernel, 1, sizeof(cl_mem), *neurons);
				clSetKernelArg(k.zeroPadBackKernel, 2, sizeof(int), &(conv->padding)); //padding
				clSetKernelArg(k.zeroPadBackKernel, 3, sizeof(int), &(net->__neuronDims[i-1][0])); //prevWidth (non-padded)
				clSetKernelArg(k.zeroPadBackKernel, 4, sizeof(int), &(net->__neuronDims[i-1][1])); //prevHeight(non-padded)
				clSetKernelArg(k.zeroPadBackKernel, 5, sizeof(int), &(net->__neuronDims[i-1][2])); //depth
				//cout << "Running zeroPadBackKernel" << endl;
				globalWorkSize[0] = (size_t)conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue,k.zeroPadBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			curConvLayer--;
		} // end if-elseif for backprop of single layer

		clFinish(queue);

		// cout << "Backprop Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = *neurons;
		*neurons = *prevNeurons;
		*prevNeurons = temp;
	}// end for loop for backprop
}


void Net::backprop_noUpdate(vector<cl_mem>& layerNeeds, vector<cl_mem>& gradients_weights, vector<cl_mem>& gradients_biases)
{
	int curConvLayer = clWeights.size() - 1;
	size_t globalWorkSize[] = {0, 0, 0};
	cl_mem* temp;
	for(int i = __layers.size() -1; i > 0; i--)
	{
		// printf("layer %d\n", i);
		if(__layers[i]->layerType == ACTIV_LAYER)
		{
			int type = ((ActivLayer*)__layers[i])->activationType;
			globalWorkSize[0] = (size_t)__neuronSizes[i-1];
			if(type == RELU)
			{
				// cout << "running reluBackKernel " << endl;
				clSetKernelArg(reluBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(reluBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, reluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(type == LEAKY_RELU)
			{
				// cout << "running leakyReluBackKernel " << endl;
				clSetKernelArg(leakyReluBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(leakyReluBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
				CheckError(clEnqueueNDRangeKernel(queue, leakyReluBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			// cout << "running maxPoolBackKernel " << endl;
			int numIndexes = __neuronSizes[i];
			clSetKernelArg(maxPoolBackKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(maxPoolBackKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i]));
			clSetKernelArg(maxPoolBackKernel, 3, sizeof(int), &numIndexes);
			clSetKernelArg(maxPoolBackKernel, 4, sizeof(int), &(__neuronDims[i][2]));
			// cout << "args set" << endl;
			globalWorkSize[0] = (size_t)__neuronSizes[i-1];
			// printf("globalWorkSize %lu\n", globalWorkSize[0]);
			CheckError(clEnqueueNDRangeKernel(queue, maxPoolBackKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			// cout << "kernel started" << endl;
		}
		else if(__layers[i]->layerType == CONV_LAYER)
		{
			//backprop neurons					
			// cout << "running convBackNeuronsKernel" << endl;
			ConvLayer* conv = (ConvLayer*)__layers[i];
			clSetKernelArg(convBackNeuronsKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(convBackNeuronsKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convBackNeuronsKernel, 2, sizeof(cl_mem), &(clWeights[curConvLayer]));
			clSetKernelArg(convBackNeuronsKernel, 3, sizeof(int), &(conv->numBiases)); //numFilters
			clSetKernelArg(convBackNeuronsKernel, 4, sizeof(int), &(conv->filterSize)); //filterSize
			clSetKernelArg(convBackNeuronsKernel, 5, sizeof(int), &(conv->stride)); //stride
			clSetKernelArg(convBackNeuronsKernel, 6, sizeof(int), &(conv->paddedNeuronWidth));
			clSetKernelArg(convBackNeuronsKernel, 7, sizeof(int), &(__neuronDims[i-1][2])); //depth

			globalWorkSize[0] = (size_t) conv->paddedNeuronSize;					
			CheckError(clEnqueueNDRangeKernel(queue, convBackNeuronsKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

			clFinish(queue); //MUST finish before weights start getting updated

			//backprop and update biases
			// cout << "running convBackBiasesKernel " << endl;
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 0, sizeof(cl_mem), &(clBiases[curConvLayer]));
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 2, sizeof(int), &(__neuronSizes[i]));
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 3, sizeof(int), &(conv->numBiases)); // numFilters = dneuronsDepth
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 4, sizeof(double), &(__learningRate));
			clSetKernelArg(convBackBiasesNoUpdateAccumKernel, 5, sizeof(cl_mem), &gradients_biases[curConvLayer]);
			globalWorkSize[0] = (size_t) conv->numBiases;
			CheckError(clEnqueueNDRangeKernel(queue, convBackBiasesNoUpdateAccumKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		// clFinish(queue);
		// cout << "ConvLayer back biases " << curConvLayer << endl;
		// CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
		// 	test.data(), 0, nullptr, nullptr));
		// printArray(test.data(), globalWorkSize[0]);
		// getchar();

			//backprop and update weights					
			// cout << "running convBackWeightsKernel" << endl;

			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 0,  sizeof(cl_mem), &(clWeights[curConvLayer]));
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 1,  sizeof(cl_mem), &(layerNeeds[i]));
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 2,  sizeof(cl_mem), neurons);
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 3,  sizeof(int), &(__neuronDims[i-1][2])); // depth
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 4,  sizeof(int), &(conv->stride)); // stride
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 5,  sizeof(int), &(conv->paddedNeuronWidth));
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 6,  sizeof(int), &(conv->filterSize)); // filterSize
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 7,  sizeof(int), &(conv->numBiases)); // numFilters
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 8,  sizeof(double), &__learningRate);
			clSetKernelArg(convBackWeightsNoUpdateAccumKernel, 9, sizeof(cl_mem), &gradients_weights[curConvLayer]);
			globalWorkSize[0] = (size_t)conv->numWeights;
			CheckError(clEnqueueNDRangeKernel(queue, convBackWeightsNoUpdateAccumKernel, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

	// clFinish(queue);
	// cout << "ConvLayer back weights " << curConvLayer << endl;
	// cout << "numWeights " << conv->numWeights << endl;
	// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
	// 	test.data(), 0, nullptr, nullptr));
	// printArray(test.data(), conv->numWeights);
	// getchar();

			//backprop zeroPad if necessary
			if(conv->padding != 0) 
			{
				clFinish(queue); //so the weights finish updating before zeroPadBackKernel starts changing prevNeurons and neurons

				//swap prev and cur neuron pointers
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;

				clSetKernelArg(zeroPadBackKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(zeroPadBackKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(zeroPadBackKernel, 2, sizeof(int), &(conv->padding)); //padding
				clSetKernelArg(zeroPadBackKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); //prevWidth (non-padded)
				clSetKernelArg(zeroPadBackKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); //prevHeight(non-padded)
				clSetKernelArg(zeroPadBackKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); //depth
				// cout << "Running zeroPadBackKernel" << endl;
				globalWorkSize[0] = (size_t)conv->paddedNeuronSize;
				CheckError(clEnqueueNDRangeKernel(queue,zeroPadBackKernel, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			curConvLayer--;
		} // end if-elseif for backprop of single layer

		clFinish(queue);
		// cout << "post finish" << endl;

		// cout << "Backprop Layer " << i << endl;
		// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 	test.data(), 0, nullptr, nullptr));
		// for(int j=0; j< __neuronSizes[i]; j++)
		// {
		// 	cout << test[j] << ", ";
		// }
		// cout << endl << endl;
		// getchar();

		temp = neurons;
		neurons = prevNeurons;
		prevNeurons = temp;
	}// end for loop for backprop
}

void Net::backprop_noUpdate_BN(const int num_threads, const int minibatch_size, const int thread_num, const int start, const int amount, 
	const vector<double>& trueVals, vector<cl_mem*> *prevNeurons, vector<cl_mem*> *neurons,
	const vector<vector<cl_mem> >& layerNeeds, const cl_command_queue& queue, const Kernels& k, spinlock_barrier* barrier,
	const vector<cl_mem>& gradients_weights, const vector<cl_mem>& gradients_biases, const vector<cl_mem>& bn_x_cl)
{
	int curConvLayer = clWeights.size() - 1, curBNLayer = gamma.size() - 1;
	size_t globalWorkSize[] = {0, 0, 0};
	cl_mem* temp;

	if(usesSoftmax)
		for(int a = 0; a < amount; a++)
		{
			cl_mem **pptr = &(*prevNeurons)[a], **nptr = &(*neurons)[a];
			softmaxBackprop(trueVals[start + a], pptr, nptr, queue, k, this);
		}

	for(int i = __layers.size() -1; i > 0; i--)
	{
		// auto cstarttime = chrono::system_clock::now();
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			#ifdef _DEBUG
			printf("Thread %d: Backprop - Start BNLayer %d\n", thread_num,i);
			#endif
			BatchNormLayer *batch = (BatchNormLayer*)__layers[i];
			int depth = __neuronDims[i][2];
			
			//delta_y is in neurons
			#ifdef _DEBUG
			printf("Thread %d: Backprop - read delta y %d\n", thread_num,i);
			#endif
			vector<vector<double> >delta_y(amount);
			for(int a = 0; a < amount; a++)
			{
				// printf("Thread %d.%d: delta_y ", thread_num,a);
				delta_y[a].resize(__neuronSizes[i]);
				CheckError(clEnqueueReadBuffer(queue, *(*neurons)[a], CL_TRUE, 0, sizeof(double) * delta_y[a].size(), 
					delta_y[a].data(), 0, nullptr, nullptr));
				// for(int y = 0; y < delta_y[a].size(); y++)
				// 	printf("%lf, ", delta_y[a][y]);
				// printf("\n");
			}
			// getchar();

			//calc delta_gamma and delta_beta and put on gpu
			#ifdef _DEBUG
			printf("Thread %d: Backprop - Calc delta gamma and delta beta %d\n", thread_num,i);
			#endif
			if(batch->byFeatureMap)
			{
				for(int d = 0; d < depth; d++) // for each feature map "d"
				{
					double sum_b = 0, sum_g = 0;
					for(int feat = d; feat < __neuronSizes[i]; feat += depth) // for each feature in feature map "d"
					{
						for(int a = 0; a < amount; a++)
						{
							sum_b += delta_y[a][feat];
							sum_g += delta_y[a][feat] * bn_xhat[thread_num][a][curBNLayer][feat];
						}
					}
					lock_guard<mutex> guard(mtx_a[d]); // there are extra limits on concurrency between this and the mtxs used in delta_sigma2 that slow down performance
					delta_beta[curBNLayer][d] += sum_b;
					delta_gamma[curBNLayer][d] += sum_g;
				}
			}
			else
			{
				for(int dy = 0; dy < delta_beta[curBNLayer].size(); dy++)
				{
					lock_guard<mutex> guard(mtx_a[dy]);
					for(int a = 0; a < amount; a++)
					{
						delta_beta[curBNLayer][dy] += delta_y[a][dy];
						delta_gamma[curBNLayer][dy] += delta_y[a][dy] * bn_xhat[thread_num][a][curBNLayer][dy];
					}
				}
			}

			#ifdef _DEBUG
			printf("Thread %d: Backprop - Calc delta sigma2 %d\n", thread_num,i);
			#endif

			//delta_sigma2
			if(batch->byFeatureMap)
			{
				for(int d = 0; d < depth; d++)
				{
					double sum = 0;
					for(int a = 0; a < amount; a++)
					{
						for(int feat = d; feat < __neuronSizes[i]; feat += depth)
						{
							double delta_xhat = delta_y[a][feat] * batch->gamma[d];
							int x_i = bn_x[thread_num][a][curBNLayer][feat];
							sum += delta_xhat * (bn_x[thread_num][a][curBNLayer][feat] - mu[curBNLayer][d])
								* -0.5 * pow(sigma_squared[curBNLayer][d] + 1e-8, -1.5);
						}
					}
					lock_guard<mutex> guard(mtx_a[d]);
					delta_sigma2[curBNLayer][d] += sum;
				}
			}
			else
			{
				for(int feat = 0; feat < __neuronSizes[i]; feat++) // for each feature
				{
					lock_guard<mutex> guard(mtx_a[feat]);
					for(int a = 0; a < amount; a++)
					{
						double delta_xhat = delta_y[a][feat] * batch->gamma[feat];
						int x_i = bn_x[thread_num][a][curBNLayer][feat];
						delta_sigma2[curBNLayer][feat] += delta_xhat * (bn_x[thread_num][a][curBNLayer][feat] - mu[curBNLayer][feat])
							* -0.5 * pow(sigma_squared[curBNLayer][feat] + 1e-8, -1.5);
					}
				}
			}

			//delta mu
			#ifdef _DEBUG
			printf("Thread %d: Backprop - Calc delta mu %d\n", thread_num,i);
			#endif
			barrier->count_down_and_wait();
			if(batch->byFeatureMap)
			{
				double m_num_features = minibatch_size * __neuronDims[i][0] * __neuronDims[i][1];
				for(int d = 0; d < depth; d++)
				{
					double sum = 0;
					for(int a = 0; a < amount; a++)
					{
						for(int feat = d; feat < __neuronSizes[i]; feat += depth)
						{
							double delta_xhat = delta_y[a][feat] * batch->gamma[d];
							int x_i = bn_x[thread_num][a][curBNLayer][feat];
							sum += -delta_xhat / pow(sigma_squared[curBNLayer][d] + 1e-8,0.5) 
								+ delta_sigma2[curBNLayer][d] * -2 * (x_i - mu[curBNLayer][d]) / m_num_features;
						}
					}
					lock_guard<mutex> guard(mtx_a[d]);
					delta_mu[curBNLayer][d] += sum;
				}
			}
			else
			{
				for(int feat = 0; feat < __neuronSizes[i]; feat++) // for each feature
				{
					double sum = 0;
					for(int a = 0; a < amount; a++)
					{
						double delta_xhat = delta_y[a][feat] * batch->gamma[feat];
						int x_i = bn_x[thread_num][a][curBNLayer][feat];
						sum += -delta_xhat / pow(sigma_squared[curBNLayer][feat] + 1e-8,0.5) 
					 		+ delta_sigma2[curBNLayer][feat] * -2 * (x_i - mu[curBNLayer][feat]) / (minibatch_size);
					}
					lock_guard<mutex> guard(mtx_a[feat]);
					delta_mu[curBNLayer][feat] += sum;
				}
			}

			//put delta_mu, delta_sigma2, delta_gamma, and delta_beta on gpu
			// printf("%d of %d\n", thread_count + 1, num_threads);
			barrier->count_down_and_wait();
			if(thread_num == 0)
			{
				CheckError(clEnqueueWriteBuffer(queue, delta_mu_cl[curBNLayer], CL_FALSE, 0, sizeof(double) * delta_mu[curBNLayer].size(),
					delta_mu[curBNLayer].data(), 0, nullptr, nullptr));
				CheckError(clEnqueueWriteBuffer(queue, delta_sigma2_cl[curBNLayer], CL_FALSE, 0, sizeof(double) * delta_sigma2[curBNLayer].size(),
					delta_sigma2[curBNLayer].data(), 0, nullptr, nullptr));
				CheckError(clEnqueueWriteBuffer(queue, delta_gamma_cl[curBNLayer], CL_FALSE, 0, sizeof(double) * delta_gamma[curBNLayer].size(),
					delta_gamma[curBNLayer].data(), 0, nullptr, nullptr));
				CheckError(clEnqueueWriteBuffer(queue, delta_beta_cl[curBNLayer], CL_FALSE, 0, sizeof(double) * delta_beta[curBNLayer].size(),
					delta_beta[curBNLayer].data(), 0, nullptr, nullptr));
				clFinish(queue);
			}

			//calc delta_x on gpu
			barrier->count_down_and_wait();

			for(int a = 0; a < amount; a++)
				CheckError(clEnqueueWriteBuffer(queue, bn_x_cl[a], CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
					bn_x[thread_num][a][curBNLayer].data(), 0, nullptr, nullptr));

			int adjusted_minibatch_size = batch->byFeatureMap ? minibatch_size * __neuronDims[i][0] * __neuronDims[i][1] : minibatch_size;
			int adjusted_depth = batch->byFeatureMap ? __neuronDims[i][2] : 0; // number < 1 means by activation instead of by feature map. use number < 1 when previous layer is non-convLayers
			for(int a = 0; a < amount; a++)
			{	
				// printf("back a: %d\n", a);
				CheckError(clSetKernelArg(k.batchNormBackKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 1, sizeof(cl_mem), (*neurons)[a]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 2, sizeof(int), &adjusted_depth));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 3, sizeof(cl_mem), &gamma[curBNLayer]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 4, sizeof(cl_mem), &mu_cl[curBNLayer]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 5, sizeof(cl_mem), &sigma_squared_cl[curBNLayer]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 6, sizeof(cl_mem), &delta_mu_cl[curBNLayer]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 7, sizeof(cl_mem), &delta_sigma2_cl[curBNLayer]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 8, sizeof(cl_mem), &bn_x_cl[a]));
				CheckError(clSetKernelArg(k.batchNormBackKernel, 9, sizeof(int), &adjusted_minibatch_size));
				globalWorkSize[0] = __neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, k.batchNormBackKernel, 1, nullptr,
					globalWorkSize, nullptr, 0, nullptr, nullptr));
				clFinish(queue);
			}
			// getchar();

			//swap pointers
			for(int a = 0; a < amount; a++)
			{
				temp = (*neurons)[a];
				(*neurons)[a] = (*prevNeurons)[a];
				(*prevNeurons)[a] = temp;
			}

			curBNLayer--;
			// auto cendtime = chrono::system_clock::now();
			// auto elapsed = chrono::duration_cast<chrono::milliseconds>(cendtime - cstarttime);
			// printf("Time back BN: %lld\n",elapsed.count());
		}
		else
		{
			// auto cstarttime = chrono::system_clock::now();
			for(int a = 0; a < amount; a++)
			{
				#ifdef _DEBUG
				printf("Thread %d.%d: Backprop Layer %d Type %d\n", thread_num,a,i,__layers[i]->layerType);
				#endif
				// printf("layer %d\n", i);
				if(__layers[i]->layerType == ACTIV_LAYER)
				{
					int type = ((ActivLayer*)__layers[i])->activationType;
					globalWorkSize[0] = (size_t)__neuronSizes[i-1];
					if(type == RELU)
					{
						// cout << "running reluBackKernel " << endl;
						clSetKernelArg(k.reluBackKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
						clSetKernelArg(k.reluBackKernel, 1, sizeof(cl_mem), (*neurons)[a]);
						clSetKernelArg(k.reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[a][i])); // dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, k.reluBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
					else if(type == LEAKY_RELU)
					{
						// cout << "running leakyReluBackKernel " << endl;
						clSetKernelArg(k.leakyReluBackKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
						clSetKernelArg(k.leakyReluBackKernel, 1, sizeof(cl_mem), (*neurons)[a]);
						clSetKernelArg(k.leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[a][i])); // dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, k.leakyReluBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
				}
				else if(__layers[i]->layerType == MAX_POOL_LAYER)
				{
					// cout << "running maxPoolBackKernel " << endl;
					int numIndexes = __neuronSizes[i];
					clSetKernelArg(k.maxPoolBackKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
					clSetKernelArg(k.maxPoolBackKernel, 1, sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[a][i]));
					clSetKernelArg(k.maxPoolBackKernel, 3, sizeof(int), &numIndexes);
					clSetKernelArg(k.maxPoolBackKernel, 4, sizeof(int), &(__neuronDims[i][2]));
					// cout << "args set" << endl;
					globalWorkSize[0] = (size_t)__neuronSizes[i-1];
					// printf("globalWorkSize %lu\n", globalWorkSize[0]);
					CheckError(clEnqueueNDRangeKernel(queue, k.maxPoolBackKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					// cout << "kernel started" << endl;
				}
				else if(__layers[i]->layerType == CONV_LAYER)
				{
					auto cstarttime = chrono::system_clock::now();
					#ifdef _DEBUG
					printf("Thread %d.%d: back conv neuron\n", thread_num,a);
					#endif
					//backprop neurons					
					// cout << "running convBackNeuronsKernel" << endl;
					ConvLayer* conv = (ConvLayer*)__layers[i];
					clSetKernelArg(k.convBackNeuronsKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
					clSetKernelArg(k.convBackNeuronsKernel, 1, sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.convBackNeuronsKernel, 2, sizeof(cl_mem), &(clWeights[curConvLayer]));
					clSetKernelArg(k.convBackNeuronsKernel, 3, sizeof(int), &(conv->numBiases)); //numFilters
					clSetKernelArg(k.convBackNeuronsKernel, 4, sizeof(int), &(conv->filterSize)); //filterSize
					clSetKernelArg(k.convBackNeuronsKernel, 5, sizeof(int), &(conv->stride)); //stride
					clSetKernelArg(k.convBackNeuronsKernel, 6, sizeof(int), &(conv->paddedNeuronWidth));
					clSetKernelArg(k.convBackNeuronsKernel, 7, sizeof(int), &(__neuronDims[i-1][2])); //depth

					globalWorkSize[0] = (size_t) conv->paddedNeuronSize;					
					CheckError(clEnqueueNDRangeKernel(queue, k.convBackNeuronsKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

					clFinish(queue); //MUST finish before weights start getting updated
					auto cendtime = chrono::system_clock::now();
					auto elapsed = chrono::duration_cast<chrono::microseconds>(cendtime - cstarttime);
					// printf("Time back conv neurons: %lld\n",elapsed.count());
					cstarttime = chrono::system_clock::now();

					#ifdef _DEBUG
					printf("Thread %d.%d: back conv biases\n", thread_num,a);
					#endif

					//backprop and update biases
					// cout << "running convBackBiasesKernel " << endl;
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 0, sizeof(cl_mem), &(clBiases[curConvLayer]));
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 1, sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 2, sizeof(int), &(__neuronSizes[i]));
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 3, sizeof(int), &(conv->numBiases)); // numFilters = dneuronsDepth
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 4, sizeof(double), &(__learningRate));
					clSetKernelArg(k.convBackBiasesNoUpdateAccumKernel, 5, sizeof(cl_mem), &gradients_biases[curConvLayer]);
					globalWorkSize[0] = (size_t) conv->numBiases;
					//because of += we need to ensure mutual exclusion
					gb_mtx.lock();
						CheckError(clEnqueueNDRangeKernel(queue, k.convBackBiasesNoUpdateAccumKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
						clFinish(queue);
					gb_mtx.unlock();

					clFinish(queue);
					cendtime = chrono::system_clock::now();
					elapsed = chrono::duration_cast<chrono::microseconds>(cendtime - cstarttime);
					// printf("Time back conv biases: %lld\n",elapsed.count());
					cstarttime = chrono::system_clock::now();
				// cout << "ConvLayer back biases " << curConvLayer << endl;
				// CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
				// 	test.data(), 0, nullptr, nullptr));
				// printArray(test.data(), globalWorkSize[0]);
				// getchar();

					#ifdef _DEBUG
					printf("Thread %d.%d: back conv weights\n", thread_num,a);
					#endif

					//backprop and update weights					
					// cout << "running convBackWeightsKernel" << endl;
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 0,  sizeof(cl_mem), &(clWeights[curConvLayer]));
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 1,  sizeof(cl_mem), &(layerNeeds[a][i]));
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 2,  sizeof(cl_mem), (*neurons)[a]);
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 3,  sizeof(int), &(__neuronDims[i-1][2])); // depth
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 4,  sizeof(int), &(conv->stride)); // stride
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 5,  sizeof(int), &(conv->paddedNeuronWidth));
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 6,  sizeof(int), &(conv->filterSize)); // filterSize
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 7,  sizeof(int), &(conv->numBiases)); // numFilters
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 8,  sizeof(double), &__learningRate);
					clSetKernelArg(k.convBackWeightsNoUpdateAccumKernel, 9, sizeof(cl_mem), &gradients_weights[curConvLayer]);
					globalWorkSize[0] = (size_t)conv->numWeights;
					//because of += we need to ensure mutual exclusion
					gw_mtx.lock();
						CheckError(clEnqueueNDRangeKernel(queue, k.convBackWeightsNoUpdateAccumKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
						clFinish(queue);
					gw_mtx.unlock();

					clFinish(queue);
					cendtime = chrono::system_clock::now();
					elapsed = chrono::duration_cast<chrono::microseconds>(cendtime - cstarttime);
					// printf("Time back conv weights: %lld\n",elapsed.count());
					
				// cout << "ConvLayer back weights " << curConvLayer << endl;
				// cout << "numWeights " << conv->numWeights << endl;
				// CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
				// 	test.data(), 0, nullptr, nullptr));
				// printArray(test.data(), conv->numWeights);
				// getchar();

					//backprop zeroPad if necessary
					if(conv->padding != 0) 
					{
						cstarttime = chrono::system_clock::now();
						#ifdef _DEBUG
						printf("Thread %d.%d: back conv padding\n", thread_num,a);
						#endif
						clFinish(queue); //so the weights finish updating before zeroPadBackKernel starts changing prevNeurons and neurons

						//swap prev and cur neuron pointers
						temp = (*neurons)[a];
						(*neurons)[a] = (*prevNeurons)[a];
						(*prevNeurons)[a] = temp;

						clSetKernelArg(k.zeroPadBackKernel, 0, sizeof(cl_mem), (*prevNeurons)[a]);
						clSetKernelArg(k.zeroPadBackKernel, 1, sizeof(cl_mem), (*neurons)[a]);
						clSetKernelArg(k.zeroPadBackKernel, 2, sizeof(int), &(conv->padding)); //padding
						clSetKernelArg(k.zeroPadBackKernel, 3, sizeof(int), &(__neuronDims[i-1][0])); //prevWidth (non-padded)
						clSetKernelArg(k.zeroPadBackKernel, 4, sizeof(int), &(__neuronDims[i-1][1])); //prevHeight(non-padded)
						clSetKernelArg(k.zeroPadBackKernel, 5, sizeof(int), &(__neuronDims[i-1][2])); //depth
						// cout << "Running zeroPadBackKernel" << endl;
						globalWorkSize[0] = (size_t)conv->paddedNeuronSize;
						CheckError(clEnqueueNDRangeKernel(queue,k.zeroPadBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

						clFinish(queue);
						cendtime = chrono::system_clock::now();
						elapsed = chrono::duration_cast<chrono::microseconds>(cendtime - cstarttime);
						// printf("Time back conv zero pad: %lld\n",elapsed.count());
					}
					#ifdef _DEBUG
					printf("Thread %d.%d: back conv done\n", thread_num,a);
					#endif

					if(a == amount - 1)
						curConvLayer--;
				} // end if-elseif for backprop of single layer

				clFinish(queue);
				// cout << "post finish" << endl;

				// cout << "Backprop Layer " << i << endl;
				// CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
				// 	test.data(), 0, nullptr, nullptr));
				// for(int j=0; j< __neuronSizes[i]; j++)
				// {
				// 	cout << test[j] << ", ";
				// }
				// cout << endl << endl;
				// getchar();

				temp = (*neurons)[a];
				(*neurons)[a] = (*prevNeurons)[a];
				(*prevNeurons)[a] = temp;

				#ifdef _DEBUG
				printf("Thread %d.%d: swap\n", thread_num,a);
				#endif
			}
			// auto cendtime = chrono::system_clock::now();
			// auto elapsed = chrono::duration_cast<chrono::milliseconds>(cendtime - cstarttime);
			// printf("Time back layerType %d: %lld\n",__layers[i]->layerType, elapsed.count());
		}
	}// end for loop for backprop
	#ifdef _DEBUG
	printf("Thread %d: very last thing in backprop\n", thread_num);
	#endif
}


void Net::updateWeights(vector<cl_mem>& gradients_weights, vector<cl_mem>& gradients_biases, vector<cl_mem>& velocities)
{
	int curConvLayer = 0;
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];
			size_t sizeBias[] = {(size_t) conv->numBiases,0,0};
			size_t sizeWeight[] = {(size_t) conv->numWeights,0,0};

			clSetKernelArg(updateBiasesKernel, 0, sizeof(cl_mem), &clBiases[curConvLayer]);
			clSetKernelArg(updateBiasesKernel, 1, sizeof(cl_mem), &gradients_biases[curConvLayer]);
			clSetKernelArg(updateBiasesKernel, 2, sizeof(double), &__learningRate);
			CheckError(clEnqueueNDRangeKernel(queue, updateBiasesKernel, 1,
				nullptr, sizeBias, nullptr, 0, nullptr, nullptr));

			if(__useMomentum)
			{
				clSetKernelArg(updateWeightsMomentKernel, 0, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(updateWeightsMomentKernel, 1, sizeof(cl_mem), &gradients_weights[curConvLayer]);
				clSetKernelArg(updateWeightsMomentKernel, 2, sizeof(double), &__learningRate);
				clSetKernelArg(updateWeightsMomentKernel, 3, sizeof(cl_mem), &velocities[curConvLayer]);
				CheckError(clEnqueueNDRangeKernel(queue, updateWeightsMomentKernel, 1,
					nullptr, sizeWeight, nullptr, 0, nullptr, nullptr));
			}
			else
			{
				clSetKernelArg(updateWeightsKernel, 0, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(updateWeightsKernel, 1, sizeof(cl_mem), &gradients_weights[curConvLayer]);
				clSetKernelArg(updateWeightsKernel, 2, sizeof(double), &__learningRate);
				CheckError(clEnqueueNDRangeKernel(queue, updateWeightsKernel, 1,
					nullptr, sizeWeight, nullptr, 0, nullptr, nullptr));
			}
			clFinish(queue);

			curConvLayer++;
		}
	}	
}

void Net::zeroMem(vector<cl_mem>& mem, const vector<size_t>& sizes)
{
	for(int i = 0; i < mem.size(); i++)
	{
		size_t globalWorkSize[] = {sizes[i],0,0};
		clSetKernelArg(zeroMemKernel, 0, sizeof(cl_mem), &(mem[i]));
		CheckError(clEnqueueNDRangeKernel(queue, zeroMemKernel, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);
	}
}

/***************************
*
* A miniBatch train function training with backprop.
* TODO: Parallelize so multiple stuff feed forwards at once. Use DETrain or AntTrain as a reference
* Public
*
***************************/
void Net::miniBatchTrain(int batchSize, int epochs)
{
	if(batchSize < 0)
	{
		printf("MiniBatch size must be positive. Aborting train.\n");
		return;
	}


	printf("MINIBATCH GRADIENT DESCENT\n");

	if(!__isFinalized)
	{
		finalize(); // this also sets __maxNeuronSize
	}

	vector<cl_mem> layerNeeds(0);
	vector<cl_mem> velocities(0);
	trainSetup(layerNeeds, velocities);

	// vector<vector<double> > confidences;
	double prevError = numeric_limits<double>::max();
	double curError;

	if(epochs == -1)
 	{
 		// if(__testData.size() == 0)
 		// 	epochs = 1;
 		// else
 			epochs = 9999; // if you need more than this set it yourself
 	}

 	string ep = to_string(epochs);
	int epSize = ep.size();

	vector<double> soft(__neuronSizes.back());

	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	bool pullWeights = true;

	time_t starttime, endtime;

	cl_int error;

	vector<cl_mem> gradients_weights, gradients_biases;
	vector<size_t> gradients_weights_sizes, gradients_biases_sizes;

	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];

			gradients_weights_sizes.push_back(conv->numWeights);
			gradients_weights.push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * conv->numWeights, nullptr, &error));
			CheckError(error);

			gradients_biases_sizes.push_back(conv->numBiases);
			gradients_biases.push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * conv->numBiases, nullptr, &error));
			CheckError(error);
		}
	}

	////////////////////////////
	// start of training
	// start of epochs
	////////////////////////////
	setbuf(stdout, NULL);
	printf("Run for %d epochs\n", epochs);
	for(int e = 0; e < epochs; e++)
	{
		starttime = time(NULL);
		//adjust learning rate
		if(e % 10 == 0 && e != 0)
		{
			__learningRate *= .5;
			printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e);
		}
		cout << "Epoch: ";
	 	cout << setw(epSize) << e;
	 	cout << ". ";
	 	// printf("Epoch: %d",e);

		getTrainingData(trainingData, trueVals); // this gets the training data for this epoch
		if(batchSize > trainingData.size())
		{
			printf("MiniBatch size is larger than the amount of training data used in an epoch. Aborting train.\n");
			pullWeights = false;
			break;
		}
		
		int numCorrect = 0;
		int imsDone = 0;

		for(int r = 0; r < trainingData.size() - batchSize; r += batchSize) // the ones that don't nicely fit in a batch will be discarded
		{
			//reset the accumulated derivatives to 0
			zeroMem(gradients_weights, gradients_weights_sizes);
			zeroMem(gradients_biases, gradients_biases_sizes);

			for(int b = r; b < r+batchSize; b++)
			{
				//put in the next image
		 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
						sizeof(double) * __neuronSizes[0],
						trainingData[b]->data(), 0, nullptr, nullptr));
				clFinish(queue);

				//do the feedForward and see if it was right
				feedForward(layerNeeds);
				softmaxForward();

				CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
				 	soft.data(), 0, nullptr, nullptr));
	            clFinish(queue);
				if(getMaxElementIndex(soft) == trueVals[b])
					numCorrect++;
				imsDone++;

				// get the derivatives
				softmaxBackprop(trueVals[b]);
				backprop_noUpdate(layerNeeds, gradients_weights, gradients_biases);

			} // end of batch
			// cout << "end of batch" << endl;

			clFinish(queue);

			//update weights
			// backprop(layerNeeds, velocities);
			updateWeights(gradients_weights, gradients_biases, velocities);

		} // end of single epoch. end of training data to go through this epoch

		endtime = time(NULL);
	 	//beginning of this line is at the top of the epoch loop
	 	cout << "Accuracy on training data: " << numCorrect << " out of " << imsDone << " (" << trueVals.size() << "). " << numCorrect/(double)imsDone*100 << "%  " << convnet::secondsToString(endtime-starttime) << " seconds" << endl;
	 	if(__testData.size() != 0)
	 	{
	 		printf("\tTest Set. ");
	 		run(); //this will update __confidences
	 		curError = 0;
	 		int testCorrect = 0;
	 		for(int c = 0; c < __confidences.size(); c++)
	 		{
	 			for(int im = 0; im < __confidences[c].size(); im++)
	 				curError += __confidences[c][im];
		 		if(getMaxElementIndex(__confidences[c]) == __testTrueIndexes[c])
		 			testCorrect++;
		 	}
	 		double testAccuracy = testCorrect/(double)__testData.size() * 100.0;
	 		printf("Accuracy on test data: %d out of %lu. %lf%%\n",testCorrect,__testData.size(), testAccuracy);

	 		//stop training when the curError is greater than the previous data and accuracy is more than 80
	 		if(curError > prevError && testAccuracy >= 80.0)
	 			break;
	 		prevError = curError;
	 	}
	}// end of all epochs
	if(pullWeights) // only false if batchSize > trainingData.size()
		pullCLWeights();
	//clean up anything we may have messed with in order to use run.
	__confidences.resize(__data.size());
	__isTraining = false;
}

void Net::setupBatchNormCLMems(int num_threads, const vector<int>& thread_sizes, vector<vector<cl_mem> >& bn_x_cl)
{
	if(setupBatchNormCLMems_done)
		return;
	#ifdef _DEBUG
	printf("Start setupBatchNormCLMems [");
	printf("%d", thread_sizes[0]);
	for(int i = 1; i < thread_sizes.size(); i++)
		printf(",%d", thread_sizes[i]);
	printf("]... ");
	#endif
	cl_int error;
	int numBNLayers = 0;
	for(int i=0; i < __layers.size(); i++)
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
			numBNLayers++;

	//these are all vector<cl_mem>
	gamma.resize(numBNLayers);
	beta.resize(numBNLayers);
	delta_gamma.resize(numBNLayers);
	delta_beta.resize(numBNLayers);
	delta_gamma_cl.resize(numBNLayers);
	delta_beta_cl.resize(numBNLayers);

	//have same number of mus and sigmas as we do gammas and betas
	mu.resize(numBNLayers);
	mu_cl.resize(numBNLayers);
	sigma_squared.resize(numBNLayers);
	sigma_squared_cl.resize(numBNLayers);

	delta_mu.resize(numBNLayers);
	delta_mu_cl.resize(numBNLayers);
	delta_sigma2.resize(numBNLayers);
	delta_sigma2_cl.resize(numBNLayers);

	bn_x.resize(num_threads);
	bn_xhat.resize(num_threads);
	bn_x_cl.resize(num_threads);
	for(int t = 0; t < num_threads; t++)
	{
		bn_x[t].resize(thread_sizes[t]);
		bn_xhat[t].resize(thread_sizes[t]);
		bn_x_cl[t].resize(thread_sizes[t]);
		for(int a = 0; a < thread_sizes[t]; a++)
		{
			bn_x[t][a].resize(numBNLayers);
			bn_xhat[t][a].resize(numBNLayers);
			bn_x_cl[t][a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error); CheckError(error);
		}
	}

	int curBNLayer = 0;
	int maxsize = 0;
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			BatchNormLayer *batch = (BatchNormLayer*)__layers[i];
			vector<double> zeros(batch->gamma.size(),0);
			// if(batch->gamma.size() > maxsize)
			// 	maxsize = batch->gamma.size();
			if(__neuronSizes[i] > maxsize)
				maxsize = __neuronSizes[i];
			//size of gamma and beta is given by the BatchNormLayer->gamma/beta.size()
			gamma[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->gamma.size(), batch->gamma.data(), &error); CheckError(error);
			beta[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->beta.size(), batch->beta.data(), &error); CheckError(error);
			delta_gamma[curBNLayer].resize(batch->gamma.size());
			delta_beta[curBNLayer].resize(batch->gamma.size());
			delta_gamma_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->gamma.size(), zeros.data(), &error); CheckError(error);
			delta_beta_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->beta.size(), zeros.data(), &error); CheckError(error);

			mu[curBNLayer].resize(batch->gamma.size());
			sigma_squared[curBNLayer].resize(batch->gamma.size());
			mu_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * batch->gamma.size(), nullptr, &error); CheckError(error);
			sigma_squared_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * batch->gamma.size(), nullptr, &error); CheckError(error);

			delta_mu[curBNLayer].resize(batch->gamma.size());
			delta_sigma2[curBNLayer].resize(batch->gamma.size());
			delta_mu_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * batch->gamma.size(), nullptr, &error); CheckError(error);
			delta_sigma2_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * batch->gamma.size(), nullptr, &error); CheckError(error);

			for(int t = 0; t < num_threads; t++)
			{
				for(int a = 0; a < thread_sizes[t]; a++)
				{
					bn_x[t][a][curBNLayer].resize(__neuronSizes[i]);
					bn_xhat[t][a][curBNLayer].resize(__neuronSizes[i]);
				}
			}

			curBNLayer++;
		}
	}
	vector<mutex> temp(maxsize);
	mtx_a.swap(temp);

	
	setupBatchNormCLMems_done = true;
	#ifdef _DEBUG
	printf("done\n");
	printf("num mutexes in mtx_a is %d\n", maxsize);
	#endif
}

void Net::destroyBatchNormCLMems()
{
	if(setupBatchNormCLMems_running_done || setupBatchNormCLMems_done)
	{
		destroyVectorCLMems(mu_cl);
		destroyVectorCLMems(sigma_squared_cl);
		destroyVectorCLMems(gamma);
		destroyVectorCLMems(beta);

	}
	if(setupBatchNormCLMems_done)
	{
		destroyVectorCLMems(delta_mu_cl);
		destroyVectorCLMems(delta_sigma2_cl);
		destroyVectorCLMems(delta_gamma_cl);
		destroyVectorCLMems(delta_beta_cl);
	}
	setupBatchNormCLMems_done = false;
	setupBatchNormCLMems_running_done = false;
}

void Net::setupBatchNormCLMems_running(int num_threads, const vector<int>& thread_sizes)
{
	if(setupBatchNormCLMems_running_done)
	{
		// printf("aleady set up\n");
		return;
	}
	#ifdef _DEBUG
	printf("Start setupBatchNormCLMems [");
	printf("%d", thread_sizes[0]);
	for(int i = 1; i < thread_sizes.size(); i++)
		printf(",%d", thread_sizes[i]);
	printf("]... ");
	#endif
	cl_int error;
	int numBNLayers = 0;
	for(int i=0; i < __layers.size(); i++)
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
			numBNLayers++;

	//these are all vector<cl_mem>
	gamma.resize(numBNLayers);
	beta.resize(numBNLayers);

	//have same number of mus and sigmas as we do gammas and betas
	mu.resize(numBNLayers);
	mu_cl.resize(numBNLayers);
	sigma_squared.resize(numBNLayers);
	sigma_squared_cl.resize(numBNLayers);

	bn_x.resize(num_threads);
	for(int t = 0; t < num_threads; t++)
	{
		bn_x[t].resize(thread_sizes[t]);
		for(int a = 0; a < thread_sizes[t]; a++)
		{
			bn_x[t][a].resize(numBNLayers);
		}
	}

	int curBNLayer = 0;
	int maxsize = 0;
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			BatchNormLayer *batch = (BatchNormLayer*)__layers[i];
			vector<double> zeros(batch->gamma.size(),0);
			// if(batch->gamma.size() > maxsize)
			// 	maxsize = batch->gamma.size();
			if(__neuronSizes[i] > maxsize)
				maxsize = __neuronSizes[i];
			//size of gamma and beta is given by the BatchNormLayer->gamma/beta.size()
			gamma[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->gamma.size(), batch->gamma.data(), &error); CheckError(error);
			beta[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->beta.size(), batch->beta.data(), &error); CheckError(error);

			mu[curBNLayer].resize(batch->gamma.size());
			sigma_squared[curBNLayer].resize(batch->gamma.size());
			mu_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->gamma.size(), batch->e.data(), &error); CheckError(error);
			sigma_squared_cl[curBNLayer] = clCreateBuffer(__context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * batch->gamma.size(), batch->var.data(), &error); CheckError(error);

			for(int t = 0; t < num_threads; t++)
			{
				for(int a = 0; a < thread_sizes[t]; a++)
				{
					bn_x[t][a][curBNLayer].resize(__neuronSizes[i]);
				}
			}

			curBNLayer++;
		}
	}
	vector<mutex> temp(maxsize);
	mtx_a.swap(temp);

	setupBatchNormCLMems_running_done = true;

	#ifdef _DEBUG
	printf("done\n");
	printf("num mutexes in mtx_a is %d\n", maxsize);
	#endif
}

void Net::pullGammaAndBeta()
{
	cl_int error;
	int curBNLayer = 0;
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			BatchNormLayer *batch = (BatchNormLayer*)__layers[i];
			
			CheckError(clEnqueueReadBuffer(queue,gamma[curBNLayer],CL_TRUE,0,sizeof(double) * batch->gamma.size(), batch->gamma.data(), 0, nullptr, nullptr));
			CheckError(clEnqueueReadBuffer(queue,beta[curBNLayer],CL_TRUE,0,sizeof(double) * batch->beta.size(), batch->beta.data(), 0, nullptr, nullptr));

			curBNLayer++;
		}
	}
}

void Net::updateGammaAndBeta()
{
	int curBNLayer = 0;
	size_t globalWorkSize[] = {0,0,0};
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
		{
			// printf("update gamma/beta layer %d\n", i);
			CheckError(clSetKernelArg(updateGammaAndBetaKernel, 0, sizeof(cl_mem), &gamma[curBNLayer]));
			CheckError(clSetKernelArg(updateGammaAndBetaKernel, 1, sizeof(cl_mem), &beta[curBNLayer]));
			CheckError(clSetKernelArg(updateGammaAndBetaKernel, 2, sizeof(cl_mem), &delta_gamma_cl[curBNLayer]));
			CheckError(clSetKernelArg(updateGammaAndBetaKernel, 3, sizeof(cl_mem), &delta_beta_cl[curBNLayer]));
			CheckError(clSetKernelArg(updateGammaAndBetaKernel, 4, sizeof(double), &__learningRate));
			globalWorkSize[0] = delta_gamma[curBNLayer].size();
			CheckError(clEnqueueNDRangeKernel(queue, updateGammaAndBetaKernel, 1, nullptr,
				globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);

			curBNLayer++;
		}
	}
	// clFinish(queue);
}

int Net::getNumBatchNormLayers()
{
	int n = 0;
	for(int i = 0; i < __layers.size(); i++)
		if(__layers[i]->layerType == BATCH_NORM_LAYER)
			n++;
	return n;
}

/***************************
*
* A batchNorm train function training does minibatch training with batch normalization.
* Public
*
***************************/
void Net::batchNormTrain(int batchSize, int epochs)
{
	setbuf(stdout, NULL);
	// if(getNumBatchNormLayers() > 0)
	// 	__preprocessType = __PREPROCESS_BATCH_NORM;
	if(batchSize < 0)
	{
		printf("MiniBatch size must be positive. Aborting train.\n");
		return;
	}

	// makeTrainingGreyscale();

	printf("MINIBATCH GRADIENT DESCENT WITH BATCH NORMALIZATION\n");

	bnClassCorrect.resize(__neuronSizes.back());
	bnClassTotal.resize(__neuronSizes.back());

	int numThreads = 1;
	if(batchSize < numThreads)
		numThreads = batchSize;
	vector<int> thread_sizes(numThreads);

	for(int mini = 0, i = 0; mini < batchSize; mini++, i = (i+1)%numThreads)
	{
		thread_sizes[i]++;
	}


	// for(int i = 0; i < thread_sizes.size(); i++)
	// 	printf("size %d: %d\n", i,thread_sizes[i]);
	#ifdef _DEBUG
	printf("thread sizes set, %d\n",__isFinalized);
	#endif

	if(!__isFinalized)
	{
		finalize(); // this also sets __maxNeuronSize
	}

	trainSetup();

	vector<vector<cl_mem> > bn_x_cl;

	setupBatchNormCLMems(numThreads, thread_sizes, bn_x_cl);


	//layerNeeds -> 3D
	//prevNeurons, neurons -> 2D
	//kernels, queue, denom, velocities -> 1D
	//barrier -> 1

	vector<vector<vector<cl_mem> > > layerNeeds(numThreads);
	vector<vector<cl_mem> > p(numThreads), n(numThreads);
	vector<vector<cl_mem*> > prevNeurons(numThreads), neurons(numThreads);
	vector<Kernels> kernels(numThreads);
	vector<cl_command_queue> queues(numThreads);
	vector<cl_mem> denoms(numThreads), velocities(0);
	initVelocities(velocities);
	cl_int error;

	spinlock_barrier barrier(numThreads);
	spinlock_barrier* barrier_ptr = &barrier;

	for(int t = 0; t < numThreads; t++)
	{
		//2D and 3D inits
		layerNeeds[t].resize(thread_sizes[t]);
		p[t].resize(thread_sizes[t]);
		n[t].resize(thread_sizes[t]);
		prevNeurons[t].resize(thread_sizes[t]);
		neurons[t].resize(thread_sizes[t]);

		for(int s = 0; s < thread_sizes[t]; s++)
		{
			setupLayerNeeds(layerNeeds[t][s]);
			//finalize sets __maxNeuronSize
			p[t][s] = clCreateBuffer(__context, CL_MEM_READ_WRITE,
				sizeof(double) * __maxNeuronSize, nullptr, &error); CheckError(error);
			n[t][s] = clCreateBuffer(__context, CL_MEM_READ_WRITE,
				sizeof(double) * __maxNeuronSize, nullptr, &error); CheckError(error);
			prevNeurons[t][s] = &(p[t][s]);
			neurons[t][s] = &(n[t][s]);
		}

		//1D inits
		buildKernels(kernels[t],__device);
		queues[t] = clCreateCommandQueue(__context, __deviceIds[__device], 0, &error); CheckError(error);
		denoms[t] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error); CheckError(error);
	}

	// vector<vector<double> > confidences;
	double bestAccuracy = 0;

	if(epochs == -1)
 	{
 		epochs = 9999; // if you need more than this set it yourself
 	}

 	string ep = to_string(epochs);
	int epSize = ep.size();

	vector<double> soft(__neuronSizes.back());

	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	bool pullWeights = true;

	time_t starttime, endtime;

	vector<cl_mem> gradients_weights, gradients_biases;
	vector<size_t> gradients_weights_sizes, gradients_biases_sizes;

	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];

			gradients_weights_sizes.push_back(conv->numWeights);
			gradients_weights.push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * conv->numWeights, nullptr, &error));
			CheckError(error);

			gradients_biases_sizes.push_back(conv->numBiases);
			gradients_biases.push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * conv->numBiases, nullptr, &error));
			CheckError(error);
		}
	}

	////////////////////////////
	// start of training
	// start of epochs
	////////////////////////////
	
	printf("Run for %d epochs on %d threads\n", epochs,numThreads);
	int lastTrainIndex;
	for(int e = 0; e < epochs; e++)
	{
		// printf("Press ENTER\n");getchar();
		starttime = time(NULL);
		bnNumCorrect = 0;
		for(int i = 0; i < bnClassCorrect.size(); i++)
		{
			bnClassCorrect[i] = 0;
			bnClassTotal[i] = 0;
		}
		//adjust learning rate
		if(e % 1 == 0 && e != 0)
		{
			printf("\tChanged learning rate from %.3e ",__learningRate);
			__learningRate *= .75;
			printf("to %.3e before starting epoch %d\n",__learningRate,e);
		}
		cout << "Epoch: ";
	 	cout << setw(epSize) << e;
	 	cout << ". ";

		getTrainingData(trainingData, trueVals); // this gets the training data for this epoch
		if(batchSize > trainingData.size())
		{
			printf("MiniBatch size is larger than the amount of training data used in an epoch. Aborting train.\n");
			pullWeights = false;
			break;
		}
		
		int numCorrect = 0;
		int imsDone = 0;
		vector<int> starts(numThreads);
		lastTrainIndex = trainingData.size() - trainingData.size() % batchSize;
		int batchesDone = 0;
		for(int r = 0; r < lastTrainIndex; r += batchSize, batchesDone++) // the ones that don't nicely fit in a batch will be discarded
		{
			bnTotalError = 0;
			//reset the accumulated derivatives to 0
			zeroMem(gradients_weights, gradients_weights_sizes);
			zeroMem(gradients_biases, gradients_biases_sizes);

			pullGammaAndBeta(); //to calc xhat in feedForward_BN

			//do the feedForward and see if it was right
			int start = r, end;
			vector<thread> thr(numThreads);
			//feedforward
			#ifdef _DEBUG
			printf("Start threadify feedforward...");
			#endif
			auto cstarttime = chrono::system_clock::now();
			// printf("\33[2K\r");
			// printf("Feed forward items %d of %d", start, lastTrainIndex);
			for(int t = 0; t < numThreads; t++) //thread-ify the batch
			{
				end = start + thread_sizes[t];
				starts[t] = start;
				vector<cl_mem*> *pptr = &(prevNeurons[t]), *nptr = &(neurons[t]);
				thr[t] = thread([=, &kernels, &trainingData, &trueVals, &layerNeeds, &queues, &denoms]
				 {feedForward_BN(numThreads, batchSize, t, ref(trainingData), ref(trueVals), start, end, pptr, nptr,//prevNeurons[t], neurons[t],
					ref(layerNeeds[t]), ref(queues[t]), ref(denoms[t]), ref(kernels[t]), barrier_ptr);});
				start = end;
			}
			// printf("%d", end);
			for(int t = 0; t < numThreads; t++)
				thr[t].join();
			auto cendtime = chrono::system_clock::now();
			auto elapsed = chrono::duration_cast<chrono::milliseconds>(cendtime - cstarttime);
			// printf("Time feed forward: %lld\n",elapsed.count());
			#ifdef _DEBUG
			printf(" feedforward joined\n");
			#endif

			// printf("Total Error: %lf\n", bnTotalError);


			//backprop
			#ifdef _DEBUG
			printf("Start threadify backprop... ");
			#endif
			cstarttime = chrono::system_clock::now();
			for(int t = 0; t < numThreads; t++)
			{
				vector<cl_mem*> *pptr = &(prevNeurons[t]), *nptr = &(neurons[t]);
				thr[t] = thread([=, &kernels, &layerNeeds, &queues, &gradients_weights, &gradients_biases, &bn_x_cl, &trueVals]
					{backprop_noUpdate_BN(numThreads, batchSize, t, starts[t], thread_sizes[t], trueVals, pptr, nptr,
					ref(layerNeeds[t]), queues[t], ref(kernels[t]), barrier_ptr, ref(gradients_weights), ref(gradients_biases), bn_x_cl[t]);});
			}
			// printf("post for\n");
			for(int t = 0; t < numThreads; t++)
				thr[t].join();

			cendtime = chrono::system_clock::now();
			elapsed = chrono::duration_cast<chrono::milliseconds>(cendtime - cstarttime);
			// cout << "Time backprop: " << elapsed.count() << endl;
			#ifdef _DEBUG
			printf("backprop joined\n");
			#endif

			clFinish(queue);

			//update weights
			updateWeights(gradients_weights, gradients_biases, velocities);
			updateGammaAndBeta();

		} // end of single epoch. end of training data to go through this epoch

		endtime = time(NULL);
	 	//beginning of this line is at the top of the epoch loop
	 	double accuracy = 100.0*bnNumCorrect/(lastTrainIndex-1);
	 	// printf("\33[2K\r");
	 	cout << "Accuracy on training data: " << bnNumCorrect << " out of " << lastTrainIndex - 1 << " (" << trueVals.size() << "). " << accuracy << "%  " << convnet::secondsToString(endtime-starttime) << " seconds" << endl;
	 	for(int i = 0; i < bnClassCorrect.size(); i++)
	 	{
	 		printf("\tClass %d (%10s) - %d of %d, %lf%%\n", i, __trueNames[i].c_str(), bnClassCorrect[i],bnClassTotal[i], 100.0 * bnClassCorrect[i]/bnClassTotal[i]);
	 	}
	 	// getchar();

	 	if(accuracy > bestAccuracy)
	 	{
	 		bestAccuracy = accuracy;
	 		if(__saveNet)
			{
				pullGammaAndBeta();
				pullCLWeights();
				save(__saveName.c_str());
			}
	 	}

	 	if(__testData.size() != 0)
	 	{
	 		printf("\tTest Set. ");
	 		run(); //this will update __confidences
	 		int testCorrect = 0;
	 		for(int c = 0; c < __confidences.size(); c++)
	 		{
		 		if(getMaxElementIndex(__confidences[c]) == __testTrueIndexes[c])
		 			testCorrect++;
		 	}
	 		double testAccuracy = testCorrect/(double)__testData.size() * 100.0;
	 		printf("Accuracy on test data: %d out of %lu. %lf%%\n",testCorrect,__testData.size(), testAccuracy);
	 	}
	}// end of all epochs

	//clean up anything we may have messed with in order to use run.
	__confidences.resize(__data.size());
	__isTraining = false;

	if(__saveNet)
		load(__saveName.c_str());

	for(int i = 0; i < layerNeeds.size(); i++)
		for(int j = 0; j < layerNeeds[i].size(); j++)
			destroyVectorCLMems(layerNeeds[i][j]);

	prevNeurons.clear();
	neurons.clear(); // so no dangling pointers
	for(int i = 0; i < p.size(); i++)
	{
		destroyVectorCLMems(p[i]);
		destroyVectorCLMems(n[i]);
	}

	for(int i = 0; i < numThreads; i++)
	{
		clReleaseCommandQueue(queues[i]);
		clReleaseMemObject(denoms[i]);
	}
	destroyVectorCLMems(velocities);
}

/***************************
*
* A batchNorm train function training does minibatch training with batch normalization.
* Public
*
***************************/
void Net::batchNormRun()
{
	setbuf(stdout, NULL);

	if(!__isTraining)
 	{
 		__dataPointer = &__data;
 	}

 	__confidences.resize(__dataPointer->size());

	int batchSize = __dataPointer->size();

	int numThreads = 4;
	if(batchSize < numThreads)
		numThreads = batchSize;
	vector<int> thread_sizes(numThreads);

	for(int mini = 0, i = 0; mini < batchSize; mini++, i = (i+1)%numThreads)
		thread_sizes[i]++;

	#ifdef _DEBUG
	for(int i = 0; i < thread_sizes.size(); i++)
		printf("size %d: %d\n", i,thread_sizes[i]);
	printf("thread sizes set, %d\n",__isFinalized);
	#endif

	if(!__isFinalized)
	{
		finalize(); // this also sets __maxNeuronSize
	}

 	if(!__dataPreprocessed)
 	{
 		// if(__preprocessIndividual)
 		if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 			preprocessDataIndividual();
 		else if(__preprocessType == __PREPROCESS_COLLECTIVE)
 			preprocessDataCollective();
 		else
			printf("NO PREPROCESSING OF TRAINING DATA\n");
 	}
	
	setupBatchNormCLMems_running(numThreads, thread_sizes);

	if(numThreads != bnrunmems.getNumThreads())
	{
		bnrunmems.load(numThreads, this); // this takes care of creation of mem. The class destructor will destroy it.
	}

	////////////////////////////
	// start of run
	// start of epochs
	////////////////////////////
	
	vector<int> starts(numThreads);

	//do the feedForward
	int start = 0, end;
	vector<thread> thr(numThreads);
	//feedforward
	#ifdef _DEBUG
	printf("Start threadify feedforward... ");
	#endif
	// printf("Starting run with %d threads\n", numThreads);
	for(int t = 0; t < numThreads; t++) //thread-ify the batch
	{
		end = start + thread_sizes[t];
		starts[t] = start;
		cl_mem **pptr = &(bnrunmems.prevNeurons[t]), **nptr = &(bnrunmems.neurons[t]);
		thr[t] = thread([=]// &kernels, &queues, &denoms]
		 {feedForward_BN_running(numThreads, batchSize, t, start, end, __dataPointer, pptr, nptr,
			ref(bnrunmems.queues[t]), ref(bnrunmems.denoms[t]), ref(bnrunmems.kernels[t]));});
		start = end;
	}
	for(int t = 0; t < numThreads; t++)
		thr[t].join();

	//the threads set __confidences

	#ifdef _DEBUG
	printf("feedforward joined\n");
	#endif
}

/*****************************
*
* Main training function. Stochiastic gradient descent backprop used.
* Saves trained net to file specified by setSaveName() every time new best net is found.
* Has auto learning rate adjustment and auto cutoff functionality.
* After 3 "stale" (accuracy not better than best) epochs, learning rate cut by 1/2 and weights
* reset to best weights (get deeper in that valley). If better accuracy found, # times stale resets to 0.
* If total 5 consecutive stale, cut again by 1/2 and go back to best. If total 7 consecutive stale, end.
* Public
*
*****************************/
void Net::train(int epochs)
{
	printf("STOCHASTIC GRADIENT DESCENT\n");

	for(int i = 0; i < __trueNames.size(); i++)
		printf("index %d is name %s\n", i,__trueNames[i].c_str());

	//set up all the layerNeeds and velocities for training.
 	vector<cl_mem> layerNeeds(0);
	vector<cl_mem> velocities(0); // still needs to exist even if not using momentum so it compiles.
	trainSetup(layerNeeds, velocities);

 	//set up stuff so we can exit based on error on test data
 	double prevError = numeric_limits<double>::max();
 	double curError;
 	int timesStale = 0;

 	WeightHolder holder;

 // 	double prevTrainError = numeric_limits<double>::max();
	// double curTrainError;

 	if(epochs == -1)
 	{
 		// if(__testData.size() == 0)
 		// 	epochs = 1;
 		// else
 			epochs = 9999; // if you need more than this set it yourself
 	}
    
	string ep = to_string(epochs);
	int epSize = ep.size();

	vector<double> soft(__neuronSizes.back());

	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	time_t starttime, endtime;

	////////////////////////////
	// start of training
	// start of epochs
	////////////////////////////
	setbuf(stdout,NULL);
	for(int e = 1; e <= epochs; e++)
	{   
		starttime = time(NULL);
		//adjust learning rate
		// if(e % 10 == 0 && e != 0)
		// {
		// 	__learningRate *= .5;
		// 	printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e);
		// }
		cout << "Epoch: ";
	 	cout << setw(epSize) << e;
	 	cout << ". ";

		getTrainingData(trainingData, trueVals); // this gets the training data for this epoch

		#ifdef _DEBUG
		printf("Training size = %lu\n",trainingData.size());
		#endif

		int numCorrect = 0;
		double totalError = 0;

	 	for(int r = 0; r < trainingData.size(); r++)
	 	{
	 		#ifdef _DEBUG
	 		printf("Starting image %d\n",r);
	 		#endif
	 		//put in the next image
	 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
					sizeof(double) * __neuronSizes[0],
					trainingData[r]->data(), 0, nullptr, nullptr));
			clFinish(queue);

			////////////////////////////
			// start forwardprop
			////////////////////////////
			#ifdef _DEBUG
			printf("Starting feed forward.\n");
			#endif
			feedForward(layerNeeds); //output goes into prevNeurons
			if(usesSoftmax)
			{
				#ifdef _DEBUG
				cout << "softmax forward" << endl;
				#endif

				softmaxForward();

				//get the output and see if it was right
				CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
				 	soft.data(), 0, nullptr, nullptr));
	            clFinish(queue);
				if(getMaxElementIndex(soft) == trueVals[r])
					numCorrect++;
			}
			else if(isApproximator)
			{
				// printf("is approx\n");
				//get the output and see if it was right
				CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
				 	soft.data(), 0, nullptr, nullptr));
	            clFinish(queue);

	            double singleError = trueVals[r] - soft[0];

	            // for(int s = 0; s < soft.size(); s++)
	            	totalError += abs(singleError);

	            //set it up for backprop. Essentially approximatorBackprop
	            CheckError(clEnqueueWriteBuffer(queue, *neurons, CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
	            	soft.data(), 0, nullptr, nullptr));
			}

			// printf("Image %d:  %lf", r, soft[0]);
			// for(int v = 1; v < soft.size(); v++)
			// 	printf(", %lf", soft[v]);
			// printf("\n");

			////////////////////////////
			// start backprop
			////////////////////////////
			if(usesSoftmax)
			{
				#ifdef _DEBUG
				printf("softmax back\n");
				#endif
				softmaxBackprop(trueVals[r]);
			}
			// printf("rest back\n");
			backprop(layerNeeds, velocities);
	 	}// end for loop for training data (meaning the epoch has finished)
	 	// printf("post epoch\n");	
 	 	endtime = time(NULL);
 	 	//beginning of this line is at the top of the epoch loop
 	 	double accuracy = 100.0 * numCorrect/trueVals.size();
 	 	printf("Accuracy on training data: %d out of %lu. %lf%% %s\n", numCorrect, trueVals.size(), accuracy,convnet::secondsToString(endtime-starttime).c_str());
 	 	if(__testData.size() != 0)
 	 	{
 	 		starttime = time(NULL);
 	 		printf("\tTest Set. ");
 	 		run(); //this will update __confidences
 	 		curError = 0;
 	 		int testCorrect = 0;
 	 		double testError = 0;
 	 		double maxError = 0;
 	 		double was = 0, found = 0;
 	 		for(int c = 0; c < __confidences.size(); c++)
 	 		{
 	 			for(int im = 0; im < __confidences[c].size(); im++)
 	 				curError += __confidences[c][im];
 	 			if(isApproximator)
 	 			{
 	 				double absError = abs(__testTrueIndexes[c] - __confidences[c][0]);
 	 				testError += absError;
 	 				if(absError > maxError)
 	 				{
 	 					maxError = absError;
 	 					was = __testTrueIndexes[c];
 	 					found  = __confidences[c][0];
 	 				}
 	 			}
 	 			else //is classifier
 	 			{
 		 			if(getMaxElementIndex(__confidences[c]) == __testTrueIndexes[c])
 		 				testCorrect++;
 		 		}
 		 	}
 		 	endtime = time(NULL);
 		 	double testAccuracy = 0;
 		 	if(isApproximator)
 		 	{
 		 		printf("Total error on test data: %lf Max single error: %lf - Was: %lf Found: %lf\n", testError, maxError, was, found);
 		 	}
 		 	else //is classifier
 		 	{
 	 			testAccuracy = testCorrect/(double)__testData.size() * 100.0;
 	 			printf("Accuracy on test data: %d out of %lu. %lf%% %s\n",testCorrect,__testData.size(), testAccuracy,convnet::secondsToString(endtime-starttime).c_str());
 	 		}
 

 	 		if((testAccuracy > holder.testAccuracy && !isApproximator) || (totalError < holder.trainError && isApproximator))
 	 		{
 	 			pullCLWeights();
 	 			if(__saveNet)
 	 				save(__saveName.c_str());
 	 			storeWeightsInHolder(holder);
 	 			holder.testAccuracy = testAccuracy;
 	 			holder.trainAccuracy = accuracy;
 	 			holder.trainError = totalError;
 	 			timesStale = 0;
 	 		}
 	 		else
 	 		{
 	 			timesStale++;
 	 			if(timesStale == 3)
 	 			{
 	 				loadWeightsFromHolder(holder);
 	 				__learningRate *= .5;
 					printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e+1);
 	 			}
 	 			else if(timesStale == 5)
 	 			{
 	 				loadWeightsFromHolder(holder);
 	 				__learningRate *= .5;
 	 				printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e+1);
 	 			}
 	 			else if(timesStale == 7)
 	 			{
 	 				printf("We don't seem to be learning anymore. Exiting.\n");
 	 				break;
 	 			}
 	 		}
	 	 	
 	 		prevError = curError;
 	 	}
 	 	else // no test data. get the one with the best training data
 	 	{
 	 		if(!isApproximator)
 	 		{
	 	 		if((accuracy > holder.trainAccuracy && !isApproximator) || (totalError < holder.trainError && isApproximator))
	 	 		{
	 	 			pullCLWeights();
	 	 			if(__saveNet)
	 	 			{
	 	 				save(__saveName.c_str());
	 	 				storeWeightsInHolder(holder);
	 	 			}
	 	 			holder.trainAccuracy = accuracy;
	 	 			holder.trainError = totalError;
	 	 			timesStale = 0;
	 	 		}
	 	 		else
	 	 		{
	 	 			timesStale++;
	 	 			if(timesStale == 3)
	 	 			{
	 	 				loadWeightsFromHolder(holder);
	 	 				__learningRate *= .5;
	 					printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e+1);
	 	 			}
	 	 			else if(timesStale == 5)
	  				{
	  					loadWeightsFromHolder(holder);
	 	 				__learningRate *= .5;
	 					printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e+1);
	 	 			}
	 	 			else if(timesStale == 7)
	 	 			{
	 	 				printf("We don't seem to be learning anymore. Exiting.\n");
	 	 				break;
	 	 			}
	 	 		}
	 	 	}
 	 	}
 	}
 	// end} of all epochs
	//get best weights and they should already be saved to the disk if we are saving them
 	if(__saveNet)
		loadWeightsFromHolder(holder);
	//clean up anything we may have messed with in order to use run.
	// vector<cl_mem> layerNeeds(0);
	// vector<cl_mem> velocities(0);
	destroyVectorCLMems(layerNeeds);
	destroyVectorCLMems(velocities);

	__confidences.resize(__data.size());
	__isTraining = false;
}

/*****************************
*
* Returns the total amount of weights (not biases) in all layers.
* Public
*
*****************************/
uint Net::getTotalWeights() const
{
	uint numWeights = 0;
	for(uint i = 1; i < __layers.size(); i++) //layer 0 is input
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			numWeights += ((ConvLayer*)(__layers[i]))->numWeights;
		}
	}
	return numWeights;
}

/*****************************
*
* Returns the total amount of biases in all layers.
* Public
*
*****************************/
uint Net::getTotalBiases() const
{
	uint numBiases = 0;
	for(uint i = 1; i < __layers.size(); i++) //layer 0 is input
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			numBiases += ((ConvLayer*)(__layers[i]))->numBiases;
		}
	}
	return numBiases;
}

/*****************************
*
* Trains CNN using Ant algorithms. Can choose what kind of ant algo to use
* and parameters only by changing in code and recompiling. 
* Does not work as well as backprop right now.
* The parallelization of feed forward is good though.
* Public
*
*****************************/
void Net::antTrain(uint maxIterations, uint population, int dataBatchSize)
{
	/******************************
	*
	*  set up all parameters
	*
	******************************/
	double rho = 0.5; //evaporation constant
	double Q;
	double oneMinusRho = 1 - rho;
	double alpha = 1; //phermone exponent
	double weightMin = -1;//-__MAX_NORM_CAP/2; //-3
	double weightMax = 1;// __MAX_NORM_CAP/2; // 3
	double weightDiff = weightMax - weightMin;
	double granularity = 0.005;
	uint numWeightOptions = ((weightMax - weightMin)/granularity); //also number of bias options
	double weightPerOption = weightDiff/numWeightOptions;
	Q = numWeightOptions/10.0;
	printf("Num weight options: %u, granularity %lf\n", numWeightOptions, granularity);

	//leak types: ANT_LEAK_NONE, ANT_LEAK_LINEAR_DECREASE, ANT_LEAK_EXPONENTIAL_DECREASE
	double leakRange = .1; //this is how far on each side pheromone should leak.
	const int leakRange_options = (int)(leakRange/weightPerOption);
	int leakType = ANT_LEAK_LINEAR_DECREASE;
	printf("leakRange options %d\n", leakRange_options);

	//init types: ANT_INIT_FANT, ANT_INIT_MMAS
	int pheromone_init_type = ANT_INIT_MMAS;

	//update types: ANT_UPDATE_SIMPLE, ANT_UPDATE_BEST, ANT_UPDATE_FANT, ANT_UPDATE_ACS
	int pheromone_update_type = ANT_UPDATE_ACS;

	//weights and stuff for FANT
	double w1 = numWeightOptions / 10.0;
	double w2 = numWeightOptions / 10.0;
	if(pheromone_update_type == ANT_UPDATE_FANT)
	{
		rho = 0;
		oneMinusRho = 1;
		// population = 1;
	}

	//stuff for MMAS
	double PHER_MAX, PHER_MIN;
	double rho1_ACS = 0.7, rho2_ACS = 0.5;
	double tau_0 = 1 / numWeightOptions;
	bool limitPheromone = true;
	if(pheromone_update_type == ANT_UPDATE_ACS)
	{
		rho = rho1_ACS;
		oneMinusRho = 1 - rho;
	}

	/******************************
	*
	*  get training data and check dataBatchSize
	*
	******************************/

	// if(__preprocessIndividual)
	if(__preprocessType == __PREPROCESS_INDIVIDUAL)
 	{
	 	if(!__trainingDataPreprocessed)
	 		//preprocessTrainingDataCollective();
	        preprocessTrainingDataIndividual();
	    if(__testData.size() != 0 && !__testDataPreprocessed)
	    	preprocessTestDataIndividual();
	}
	else if(__preprocessType == __PREPROCESS_COLLECTIVE) // __preprocessCollective
	{
		if(!__trainingDataPreprocessed)
			preprocessTrainingDataCollective();
		if(!__testDataPreprocessed && __testData.size() != 0)
			preprocessTestDataCollective();
	}
	else if(__preprocessType == __PREPROCESS_BATCH_NORM)
	{

	}

	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	getTrainingData(trainingData, trueVals);
	int curTrainDataIndex = 0;
	if(dataBatchSize <= 0)
		dataBatchSize = trainingData.size();

	/******************************
	*
	*  get total numWeights/Biases and initialize phermones
	*
	******************************/
	uint numWeights = getTotalWeights();
	uint numBiases = getTotalBiases();

	//tau_0 = 1 / (numWeights + numBiases);

	vector<vector<double> > pheromoneWeights(numWeights);
	vector<vector<double> > pheromoneBiases(numBiases);

	vector<vector<double> > weightProbs(numWeights);
	vector<vector<double> > biasProbs(numBiases);

	for(uint i = 0; i < numWeights; i++)
	{
		weightProbs[i].resize(numWeightOptions);
		pheromoneWeights[i].resize(numWeightOptions);
		if(pheromone_init_type == ANT_INIT_FANT)
		{
			for(uint j = 0; j < numWeightOptions; j++)
				pheromoneWeights[i][j] = 1; //got from FANT in notes. might change
		}
		else if(pheromone_init_type == ANT_INIT_MMAS)
		{
			for(uint j = 0; j < numWeightOptions; j++)
				pheromoneWeights[i][j] = 1;
		}
	}
	for(uint i = 0; i < numBiases; i++)
	{
		biasProbs[i].resize(numWeightOptions);
		pheromoneBiases[i].resize(numWeightOptions);
		if(pheromone_init_type == ANT_INIT_FANT)
		{
			for(uint j = 0; j < numWeightOptions; j++)
				pheromoneBiases[i][j] = 1;
		}
		else if(pheromone_init_type == ANT_INIT_MMAS)
		{
			for(uint j = 0; j < numWeightOptions; j++)
				pheromoneBiases[i][j] = 1;
		}
	}

	/******************************
	*
	*  set up ants as Nets and set up antfit
	*
	******************************/

	vector<Net*> ants(population);
	vector<double> antfit(population);
	vector<uint> antCorrect(population);
	vector<uint> prevCorrect(population,0);

	for(uint i = 0; i < population; i++)
		ants[i] = new Net(*this); //what the weights start at doesn't matter

	Net* bestNet = nullptr;
	uint bestCorrect = 0;
	double bestFit = numeric_limits<double>::max();
	uint bestEpoch;


	/******************************
	*
	*  Set up everything for OpenCL on all ants
	*
	******************************/
	cl_int error;
	uint clBatchSize = 10;
	vector<vector<cl_mem> > layerNeeds(clBatchSize), clWeights(clBatchSize), clBiases(clBatchSize), velocities(population);
	vector<cl_mem> p(clBatchSize), n(clBatchSize);
	vector<cl_mem*> prevNeurons(clBatchSize), neurons(clBatchSize);
	vector<cl_command_queue> queues(clBatchSize);
	vector<cl_mem> denoms(clBatchSize);
	vector<Kernels> kerns(clBatchSize);

	int q = __device == -1 ? 0: __device;
	int __maxNeuronSize = 0;
	for(int a = 0; a< clBatchSize; a++)
	{
		setupLayerNeeds(layerNeeds[a]);
		for(int i=1; i < __layers.size(); i++)
		{
			if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer *conv = (ConvLayer*) __layers[i];

				clWeights[a].push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE,
					sizeof(double) * conv->numWeights, nullptr, &error));
				CheckError(error);
				clBiases[a].push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE,
					sizeof(double) * conv->numBiases, nullptr, &error));
				CheckError(error);
				//none of the other layers have the ability to increase the size from the layer before
				if(conv->maxSizeNeeded > __maxNeuronSize)
				{
					__maxNeuronSize = conv->maxSizeNeeded;
				}
			}
		}

		p[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		prevNeurons[a] = &(p[a]);
		n[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		neurons[a] = &(n[a]);

		queues[a] = clCreateCommandQueue(__context, __deviceIds[q], 0, &error);
		CheckError(error);

		denoms[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);
		CheckError(error);

		buildKernels(kerns[a], q);
	}
	for(int a = 0; a < population; a++)
	{
		initVelocities(velocities[a]);
	}


	//set some softmax related args that won't change
	for(int i = 0; i < kerns.size(); i++)
	{
	 	clSetKernelArg(kerns[i].maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
	 	clSetKernelArg(kerns[i].vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	}

 	vector<double> soft(__neuronSizes.back());
 	vector<thread> thr(clBatchSize);

	/******************************
	*
	*  MAIN LOOP
	*
	******************************/

	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	//set up random gen for choosing start weightOption. Used for both weights and biases
	uniform_int_distribution<uint> optionStart(0,numWeightOptions-1);
	//set up random gen for choosing weight choice
	uniform_real_distribution<double> prob(0.0,1.0);
	uint epoch = 0;
	time_t epochStartTime  = time(NULL);
	printf("Starting %d iterations\n", maxIterations);
	for(uint iteration = 1; iteration <= maxIterations; iteration++)
	{

		/******************************
		*
		*  Determine probabilities
		*
		******************************/
		// printf("Determine probabilities\n");
		//weights
		double maxprob = 0, minprob = 100;
		double avgprob = 0;
		for(uint i = 0; i < numWeights; i++)
		{
			//get sum
			double sum = 0;
			double* wptr = pheromoneWeights[i].data();
			for(uint j = 0; j < numWeightOptions; j++)
			{
				if(*wptr < 0)
				{
					printf("ERROR: Pheromone should not be negative. Found: %lf\n", *wptr);
				}
				sum += pow(*(wptr++),alpha);
			}
			//calc prob for each weightOption in weights
			for(uint j = 0; j < numWeightOptions; j++)
			{
				double prob = pow(pheromoneWeights[i][j],alpha)/sum;
				weightProbs[i][j] = prob;
				if(prob < minprob) minprob = prob;
				if(prob > maxprob) maxprob = prob;
				avgprob += prob;
			}
		}
		avgprob /= numWeights * numWeightOptions;
		printf("Probs: Min %lf Max %lf Range %lf\n", minprob,maxprob, maxprob-minprob);

		//biases
		for(uint i = 0; i < numBiases; i++)
		{
			//get sum
			double sum = 0;
			double* bptr = pheromoneBiases[i].data();
			for(uint j = 0; j < numWeightOptions; j++)
				sum += pow(*(bptr++),alpha);

			//calc prob for each weightOption in biases
			for(uint j = 0; j < numWeightOptions; j++)
				biasProbs[i][j] = pow(pheromoneBiases[i][j],alpha)/sum;
		}

		/******************************
		*
		*  Make paths
		*
		******************************/
		// printf("Make paths\n");
		double deltaTau_acs_local = rho2_ACS * tau_0;
		for(uint i = 0; i < population; i++) //parallelize this later?
		{
			uint curWeight = 0; // used with weightProbs
			uint curBias   = 0; // used with biasProbs
			//get every weight
			for(int l = 0; l < ants[i]->__layers.size(); l++)
			{
				if(ants[i]->__layers[l]->layerType == CONV_LAYER)
				{
					ConvLayer* conv = (ConvLayer*)ants[i]->__layers[l];
					//weights
					for(int w = 0; w < conv->numWeights; w++)
					{
						double chosenWeight;
						//get chosenWeight using probs;
						// start randomly for uniformness
						for(uint p = optionStart(gen); true; p = (p+1)%numWeightOptions) //loop until found
						{
							if(prob(gen) < weightProbs[curWeight][p]) //p is a weightOption location
							{
								//chosenWeight = weightMin + ((double)(weightMax - weightMin) * p/numWeightOptions); //this is what is happening
								chosenWeight = weightMin + weightPerOption * p; //p is option number
								break;
							}
						}
						conv->weights[w] = chosenWeight;

						if(pheromone_update_type == ANT_UPDATE_ACS)
						{
							int optionIndex = (int)((conv->weights[w] - weightMin)/weightPerOption);
							pheromoneWeights[curWeight][optionIndex] = pheromoneWeights[curWeight][optionIndex] /* rho2_ACS*/ + deltaTau_acs_local;

							if(leakType == ANT_LEAK_LINEAR_DECREASE)
							{
								//go left leaking
								for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
									pheromoneWeights[curWeight][opt] = pheromoneWeights[curWeight][opt] /* rho2_ACS*/ + deltaTau_acs_local * (1 - (optionIndex - opt)/leakRange_options);
								//go right leaking
								for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
									pheromoneWeights[curWeight][opt] = pheromoneWeights[curWeight][opt] /* rho2_ACS*/ + deltaTau_acs_local * (1 - (opt - optionIndex)/leakRange_options);
							}
						}

						curWeight++;
					}

					//biases
					for(int b = 0; b < conv->numBiases; b++)
					{
						double chosenBias;
						for(uint p = optionStart(gen); true; p = (p+1)%numWeightOptions)
							if(prob(gen) < biasProbs[curBias][p])
							{
								chosenBias = weightMin + weightPerOption * p;
								break;
							}
						conv->biases[b] = chosenBias;

						if(pheromone_update_type == ANT_UPDATE_ACS)
						{
							int optionIndex = (int)((conv->biases[b] - weightMin)/weightPerOption);
							pheromoneBiases[curBias][optionIndex] = pheromoneBiases[curBias][optionIndex] /* rho2_ACS*/ + deltaTau_acs_local;

							if(leakType == ANT_LEAK_LINEAR_DECREASE)
							{
								//go left leaking
								for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
									pheromoneBiases[curBias][opt] = pheromoneBiases[curBias][opt] /* rho2_ACS*/ + deltaTau_acs_local * (1 - (optionIndex - opt)/leakRange_options);
								//go right leaking
								for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
									pheromoneBiases[curBias][opt] = pheromoneBiases[curBias][opt] /* rho2_ACS*/ + deltaTau_acs_local * (1 - (opt - optionIndex)/leakRange_options);
							}
						}
						curBias++;
					}
				}
			}
		}

		/******************************
		*
		*  Get fitnesses in parallel
		*
		******************************/

		// if(curTrainDataIndex % 1000 == 0)

		for(uint i = 0; i < population; i++)
		{
			antfit[i] = 0;
			antCorrect[i] = 0;
		}
		// printf("Get fitnesses\n");
		//determine amount of data for this iteration. is dataBatchSize unless not that much data left
		int mydataBatchSize = curTrainDataIndex + dataBatchSize >= trainingData.size() ? trainingData.size() - curTrainDataIndex : dataBatchSize;
		printf("Training examples %d - %d\n", curTrainDataIndex,curTrainDataIndex + mydataBatchSize -1 );
		for(uint i = 0; i < population; i++)
			antfit[i] = 0;
		for(uint i = 0; i < population;) //no i++ b/c myclBatchSize
		{
			//determine amount of parallelism. is clBatchSize unless not that many ants left
			int myclBatchSize = i+clBatchSize >= population ? population-i : clBatchSize;
			if(myclBatchSize == 0)
			{
				printf("Error: myclBatchSize == 0. Infinite loop. Aborting.\n");
				return;
			}
			//push weights
			for(int t = 0; t < myclBatchSize; t++)
				pushCLWeights(ants[i+t]->__layers, clWeights[t], clBiases[t], queues[t], CL_TRUE);
			//for our data batch size
			for(int d = 0; d < mydataBatchSize; d++)
			{
				for(int t = 0; t < myclBatchSize; t++)
				{
					//write training data piece
					CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, 
						sizeof(double) * __neuronSizes[0], trainingData[curTrainDataIndex+d]->data(), 0, nullptr, nullptr));

					//run threads
					cl_mem **prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]); //need because const rid
					thr[t] = thread([=] {feedForward(prevptr, nptr,ref(ants[i+t]->__neuronDims), ref(ants[i+t]->__layers), ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
				}
				for(int t = 0; t < myclBatchSize; t++)
				{
					thr[t].join();
					CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double)*__neuronSizes.back(),
						soft.data(), 0, nullptr, nullptr));
					antfit[i+t] += getFitness(soft,trueVals[curTrainDataIndex + d],ants[i+t]);
					if(getMaxElementIndex(soft) == trueVals[d])
							antCorrect[i+t]++;
				}
			}
			i += myclBatchSize;
		}
		//we are done using data for this iteration so increase it for next iteration
		curTrainDataIndex += mydataBatchSize;

		/******************************
		*
		*  If we hit all data, then let's see how we're doing
		*
		******************************/
		if(curTrainDataIndex == trainingData.size())
		{
			epoch++;
			printf("---------------------------------------------------------------\n");
			curTrainDataIndex = 0;
			if(dataBatchSize != trainingData.size())
			{
				for(uint i = 0; i < population; i++)
				{
					antfit[i] = 0;
					antCorrect[i] = 0;
				}
				for(uint i = 0; i < population;) //no i++ b/c myclBatchSize
				{
					//determine amount of parallelism. is clBatchSize unless not that many ants left
					int myclBatchSize = i+clBatchSize >= population ? population-i : clBatchSize;
					if(myclBatchSize == 0)
					{
						printf("Error: myclBatchSize == 0. Infinite loop. Aborting.\n");
						return;
					}
					//push weights
					for(int t = 0; t < myclBatchSize; t++)
						pushCLWeights(ants[i+t]->__layers, clWeights[t], clBiases[t], queues[t], CL_TRUE);
					//for our data batch size
					for(int d = 0; d < trainingData.size(); d++)
					{
						for(int t = 0; t < myclBatchSize; t++)
						{
							//write training data piece
							CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, 
								sizeof(double) * __neuronSizes[0], trainingData[d]->data(), 0, nullptr, nullptr));

							//run threads
							cl_mem **prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]); //need because const rid
							thr[t] = thread([=] {feedForward(prevptr, nptr,ref(ants[i+t]->__neuronDims), ref(ants[i+t]->__layers), ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
						}
						for(int t = 0; t < myclBatchSize; t++)
						{
							thr[t].join();
							CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double)*__neuronSizes.back(),
								soft.data(), 0, nullptr, nullptr));
							antfit[i+t] += getFitness(soft,trueVals[d],ants[i+t]);
							if(getMaxElementIndex(soft) == trueVals[d])
								antCorrect[i+t]++;
						}
					}
					i += myclBatchSize;
				}
			}
			//get new (shuffled) training data for next epoch
			getTrainingData(trainingData,trueVals);

			//print out how we're doing
			int curBestC = 0, curBestF = 0;
			for(int i = 0; i < population; i++)
			{
				//finding best net
				if(antfit[i] < antfit[curBestF])
					curBestF = i;
				if(antCorrect[i] > antCorrect[curBestC])
					curBestC = i;

				//printing stuff for this net
				char better = ' ';
				if(antCorrect[i] > prevCorrect[i])
					better = '*'; //nicely display this was better than last time
				prevCorrect[i] = antCorrect[i];
				printf("%cAnt: %3d. Acc: %7.3lf. Error: %9.2lf\n",better,i,100.0*antCorrect[i]/trainingData.size(),antfit[i]);
			}
			char better = ' ';
			//see if we have a new best!
			//if(antCorrect[curBestC] > bestCorrect)
			if(antfit[curBestF] < bestFit)
			{
				better = '*';
				if(bestNet != nullptr)
					delete bestNet;
				bestNet = new Net(*(ants[curBestF]));
				ants[curBestF]->save(__saveName.c_str());
				// bestNet->save(__saveName.c_str()); //save as we go so we don't lose it all on Ctrl+C
				printf("Saving best net with %u out of %lu correct.\n", antCorrect[curBestF],trainingData.size());
				bestCorrect = antCorrect[curBestF];
				bestFit = antfit[curBestF];
				bestEpoch = epoch;

				if(pheromone_update_type == ANT_UPDATE_FANT)
				{
					//reset pheromones
					for(int i = 0; i < pheromoneWeights.size(); i++)
						for(int j = 0; j < numWeightOptions; j++)
							pheromoneWeights[i][j] = 1;

					for(int i = 0; i < pheromoneBiases.size(); i++)
						for(int j = 0; j < numWeightOptions; j++)
							pheromoneBiases[i][j] = 1;

					//increase w1?
				}
			}
			printf("End EPOCH %u ITERATION %u - Time: %s\n", epoch,iteration,convnet::secondsToString(time(NULL) - epochStartTime).c_str());
			printf("%cAll time best: Acc: %7.3lf, Error: %9.3lf, epoch %u\n", better,100.0*bestCorrect/trainingData.size(),bestFit,bestEpoch);
			printf( " Current Best : Acc: %7.3lf, Error: %9.3lf, Ant %d\n", 100.0*antCorrect[curBestF]/trainingData.size(),antfit[curBestF],curBestF);
			printf("---------------------------------------------------------------\n");
			epochStartTime = time(NULL);
		}

		/******************************
		*
		*  Pheromone evaporation
		*
		******************************/
		if(oneMinusRho != 1.0)
		{
			//weights
			for(uint i = 0; i < numWeights; i++)
				for(uint j = 0; j < numWeightOptions; j++)
					pheromoneWeights[i][j] *= oneMinusRho;

			//biases
			for(uint i = 0; i < numBiases; i++)
				for(uint j = 0; j < numWeightOptions; j++)
					pheromoneBiases[i][j] *= oneMinusRho;
		}

		/******************************
		*
		*  Phermone updates
		*
		******************************/

		uint best = 0;
		for(int i = 1; i < population; i++)
		{
			if(antfit[i] < antfit[best])
				best = i;
		}

		PHER_MAX = 1/(rho * antfit[best]);
		PHER_MIN = 0.25 * PHER_MAX / numWeightOptions;
		printf("Pher range [%lf, %lf]\n", PHER_MIN, PHER_MAX);

		vector<Net*> uants;
		vector<double> deltaTau;
		if(pheromone_update_type == ANT_UPDATE_SIMPLE)
		{
			for(uint i = 0; i < ants.size(); i++)
			{
				uants.push_back(ants[i]);
				deltaTau.push_back(Q/antfit[i]);
			}
		}
		else if(pheromone_update_type == ANT_UPDATE_BEST || pheromone_update_type == ANT_UPDATE_FANT
			|| pheromone_update_type == ANT_UPDATE_ACS)
		{
			/*uint best = 0;
			for(int i = 1; i < population; i++)
			{
				if(antfit[i] < antfit[best])
					best = i;
			}

			PHER_MAX = 1/(rho * antfit[best]);
			PHER_MIN = 0.5 * PHER_MAX / numWeightOptions;*/

			if(pheromone_update_type == ANT_UPDATE_BEST)
			{
				uants.push_back(ants[best]);
				deltaTau.push_back(Q/antfit[best]);
			}
			else if(pheromone_update_type == ANT_UPDATE_FANT)
			{
				uants.push_back(ants[best]);
				deltaTau.push_back(w1);
				if(bestNet != nullptr)
				{
					uants.push_back(bestNet);
					deltaTau.push_back(w2);
				}
			}
			else if(pheromone_update_type == ANT_UPDATE_ACS) // this does the global part
			{
				if(bestNet != nullptr)
				{
					uants.push_back(bestNet);
					deltaTau.push_back(rho1_ACS * Q/bestFit);
				}
				else
				{
					uants.push_back(ants[best]);
					deltaTau.push_back(rho1_ACS * Q/antfit[best]);
				}
			}
		}

		for(uint i = 0; i < uants.size(); i++)
		{
			uint curWeight = 0, curBias = 0;
			for(uint l = 0; l < uants[i]->__layers.size(); l++)
			{
				if(uants[i]->__layers[l]->layerType == CONV_LAYER)
				{
					ConvLayer* conv = (ConvLayer*)uants[i]->__layers[l];
					
					//weights
					for(int w = 0; w < conv->numWeights; w++)
					{
						int optionIndex = (int)((conv->weights[w] - weightMin)/weightPerOption);
						pheromoneWeights[curWeight][optionIndex] += deltaTau[i];

						if(leakType == ANT_LEAK_LINEAR_DECREASE)
						{
							//go left leaking
							for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
								pheromoneWeights[curWeight][opt] += deltaTau[i] * (1 - (optionIndex - opt)/leakRange_options);
							//go right leaking
							for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
								pheromoneWeights[curWeight][opt] += deltaTau[i] * (1 - (opt - optionIndex)/leakRange_options);
						}

						curWeight++;
					}

					//biases
					for(int b = 0; b < conv->numBiases; b++)
					{
						int optionIndex = (int)((conv->biases[b] - weightMin)/weightPerOption);
						pheromoneBiases[curBias][optionIndex] += deltaTau[i];

						if(leakType == ANT_LEAK_LINEAR_DECREASE)
						{
							//go left leaking
							for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
								pheromoneBiases[curBias][opt] += deltaTau[i] * (1 - (optionIndex - opt)/leakRange_options);
							//go right leaking
							for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
								pheromoneBiases[curBias][opt] += deltaTau[i] * (1 - (opt - optionIndex)/leakRange_options);
						}

						curBias++;
					}
				}
			}

		}

		if(limitPheromone)
		{	
			//weights
			for(int w = 0; w < numWeights; w++)
			{
				for(int optionIndex = 0; optionIndex < numWeightOptions; optionIndex++)
					if(pheromoneWeights[w][optionIndex] < PHER_MIN)
						pheromoneWeights[w][optionIndex] = PHER_MIN;
					else if(pheromoneWeights[w][optionIndex] > PHER_MAX)
						pheromoneWeights[w][optionIndex] = PHER_MAX;
			}

			//biases
			for(int b = 0; b < numBiases; b++)
			{
				for(int optionIndex = 0; optionIndex < numWeightOptions; optionIndex++)
					if(pheromoneBiases[b][optionIndex] < PHER_MIN)
						pheromoneBiases[b][optionIndex] = PHER_MIN;
					else if(pheromoneBiases[b][optionIndex] > PHER_MAX)
						pheromoneBiases[b][optionIndex] = PHER_MAX;
			}
		
		}
		// if(pheromone_update_type == ANT_UPDATE_SIMPLE)
		// {
		// 	for(uint i = 0; i < population; i++)
		// 	{
		// 		uint curWeight = 0;
		// 		uint curBias   = 0;

		// 		//make antfit the average error per example
		// 		antfit[i] /= mydataBatchSize;
		// 		double deltaTau = Q/antfit[i];

		// 		for(uint l = 0; l < ants[i]->__layers.size(); l++)
		// 		{
		// 			if(ants[i]->__layers[l]->layerType == CONV_LAYER)
		// 			{
		// 				ConvLayer* conv = (ConvLayer*)ants[i]->__layers[l];
						
		// 				//weights
		// 				for(int w = 0; w < conv->numWeights; w++)
		// 				{
		// 					int optionIndex = (int)((conv->weights[w] - weightMin)/weightPerOption);
		// 					pheromoneWeights[curWeight][optionIndex] += deltaTau;

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * (optionIndex - opt)/leakRange_options;
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * (opt - optionIndex)/leakRange_options;
		// 					}
		// 					else if(leakType == ANT_LEAK_EXPONENTIAL_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * pow((optionIndex - opt)/leakRange_options,2);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * pow((opt - optionIndex)/leakRange_options,2);
		// 					}

		// 					curWeight++;
		// 				}

		// 				for(int b = 0; b < conv->numBiases; b++)
		// 				{
		// 					int optionIndex = (int)((conv->biases[b] - weightMin)/weightPerOption);
		// 					pheromoneBiases[curBias][optionIndex] += deltaTau;

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
		// 							pheromoneBiases[curBias][opt] += deltaTau * (optionIndex - opt)/leakRange_options;
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
		// 							pheromoneBiases[curBias][opt] += deltaTau * (opt - optionIndex)/leakRange_options;
		// 					}
		// 					else if(leakType == ANT_LEAK_EXPONENTIAL_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
		// 							pheromoneBiases[curBias][opt] += deltaTau * pow((optionIndex - opt)/leakRange_options,2);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
		// 							pheromoneBiases[curBias][opt] += deltaTau * pow((opt - optionIndex)/leakRange_options,2);
		// 					}

		// 					curBias++;
		// 				}
		// 			}
		// 		}
		// 	}
		// }
		// else if(pheromone_update_type == ANT_UPDATE_BEST)
		// {
		// 	uint best = 0;
		// 	for(int i = 1; i < population; i++)
		// 	{
		// 		if(antfit[i] < antfit[best])
		// 			best = i;
		// 	}
		// 	antfit[best] /= mydataBatchSize; //make it avg per example. more scalable this way

		// 	double deltaTau = Q/antfit[best];
		// 	// printf("avgfit: %lf Q: %lf deltaTau: %lf\n", antfit[best],Q,deltaTau);

		// 	uint curWeight = 0, curBias = 0;
		// 	for(uint l = 0; l < ants[best]->__layers.size(); l++)
		// 		{
		// 			if(ants[best]->__layers[l]->layerType == CONV_LAYER)
		// 			{
		// 				ConvLayer* conv = (ConvLayer*)ants[best]->__layers[l];
						
		// 				//weights
		// 				for(int w = 0; w < conv->numWeights; w++)
		// 				{
		// 					int optionIndex = (int)((conv->weights[w] - weightMin)/weightPerOption);
		// 					pheromoneWeights[curWeight][optionIndex] += deltaTau;

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * (1 - (optionIndex - opt)/leakRange_options);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
		// 							pheromoneWeights[curWeight][opt] += deltaTau * (1 - (opt - optionIndex)/leakRange_options);
		// 					}

		// 					curWeight++;
		// 				}

		// 				for(int b = 0; b < conv->numBiases; b++)
		// 				{
		// 					int optionIndex = (int)((conv->biases[b] - weightMin)/weightPerOption);
		// 					pheromoneBiases[curBias][optionIndex] += deltaTau;

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
		// 							pheromoneBiases[curBias][opt] += deltaTau * (1 - (optionIndex - opt)/leakRange_options);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
		// 							pheromoneBiases[curBias][opt] += deltaTau * (1 - (opt - optionIndex)/leakRange_options);
		// 					}

		// 					curBias++;
		// 				}
		// 			}
		// 		}
		// }
		// else if(pheromone_update_type == ANT_UPDATE_FANT)
		// {
		// 	// printf("Start update fant\n");
		// 	uint best = 0;
		// 	for(int i = 1; i < population; i++)
		// 	{
		// 		if(antfit[i] < antfit[best])
		// 			best = i;
		// 	}

		// 	double delta[] = {w1,w2};
		// 	Net* fants[] = {ants[best],bestNet};
			
		// 	for(int i = 0; i < 2; i++)
		// 	{
		// 		uint curWeight = 0, curBias = 0;
		// 		if(fants[i] == nullptr)
		// 			continue;
		// 		// printf("i: %d\n", i);
		// 		// printf("size = %lu\n", fants[i]->__layers.size());
		// 		// printf("here\n");
		// 		for(uint l = 0; l < fants[i]->__layers.size(); l++)
		// 		{
		// 			if(fants[i]->__layers[l]->layerType == CONV_LAYER)
		// 			{
		// 				ConvLayer* conv = (ConvLayer*)fants[i]->__layers[l];
						
		// 				//weights
		// 				// printf("layer %u\n", l);
		// 				// printf("numWeight %d\n", conv->numWeights);
		// 				for(int w = 0; w < conv->numWeights; w++)
		// 				{
		// 					int optionIndex = (int)((conv->weights[w] - weightMin)/weightPerOption);
		// 					pheromoneWeights[curWeight][optionIndex] += delta[i];

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt >= optionIndex - leakRange_options; opt--)
		// 							pheromoneWeights[curWeight][opt] += delta[i] * (1 - (optionIndex - opt)/leakRange_options);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt <= optionIndex + leakRange_options; opt++)
		// 							pheromoneWeights[curWeight][opt] += delta[i] * (1 - (opt - optionIndex)/leakRange_options);
		// 					}

		// 					curWeight++;
		// 				}

		// 				// printf("weights done starting biases\n");
		// 				// printf("numBiases %d\n", conv->numBiases);
		// 				for(int b = 0; b < conv->numBiases; b++)
		// 				{
		// 					int optionIndex = (int)((conv->biases[b] - weightMin)/weightPerOption);
		// 					pheromoneBiases[curBias][optionIndex] += delta[i];

		// 					if(leakType == ANT_LEAK_LINEAR_DECREASE)
		// 					{
		// 						//go left leaking
		// 						for(int opt = optionIndex - 1; opt >= 0 && opt > optionIndex - leakRange_options; opt--)
		// 							pheromoneBiases[curBias][opt] += delta[i] * (1 - (optionIndex - opt)/leakRange_options);
		// 						//go right leaking
		// 						for(int opt = optionIndex + 1; opt < numWeightOptions && opt < optionIndex + leakRange_options; opt++)
		// 							pheromoneBiases[curBias][opt] += delta[i] * (1 - (opt - optionIndex)/leakRange_options);
		// 					}

		// 					curBias++;
		// 				}
		// 			}
		// 		}
		// 	}
		// 	// printf("end update fant\n");
		// }
	} //end iterations	
}


/*****************************
*
* DO NOT USE
* Fancy implementation of training structure and weights of CNN by DE. Never finished, doesn't work.
* Public
*
*****************************/
//If you use DETrain, only the input size and location of maxpools matter
/*void Net::DETrain(int generations, int population, double mutationScale, int crossMethod, double crossProb, bool BP)
{
	printf("DE training\n");

	//We'll need some stuff to hold our training data
	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	//preprocess the data, training and test
 	if(__preprocessIndividual)
 	{
	 	if(!__trainingDataPreprocessed)
	 		//preprocessTrainingDataCollective();
	        preprocessTrainingDataIndividual();
	    if(__testData.size() != 0 && !__testDataPreprocessed)
	    	preprocessTestDataIndividual();
	}
	else // __preprocessCollective
	{
		if(!__trainingDataPreprocessed)
			preprocessTrainingDataCollective();
		if(!__testDataPreprocessed && __testData.size() != 0)
			preprocessTestDataCollective();
	}

	getTrainingData(trainingData, trueVals);
	int curTrainDataIndex = 0;


	//We'll have a vector of nets to hold the population
	vector<Net*> nets(population);
	vector<double> netfit(population, -1);

	//set up all the nets
	setupRandomNets(nets);
	for(int i = 0; i < nets.size(); i++)
	{
		nets[i]->setDevice(__device);
		nets[i]->setTrainingType(TRAIN_AS_IS);
		nets[i]->__dataPreprocessed = true;
		nets[i]->__isTraining = true;
		if(!__preprocessIndividual) //this means preprocess collective
		{
			//store the mean and stddev for when we choose the best
			nets[i]->preprocessCollectively();
			nets[i]->__mean = __mean;
			nets[i]->__stddev = __stddev;
		}
		nets[i]->finalize();
	}

	vector<vector<double> > curTrainData(1);
	double curTrueVal;
	vector<vector<double> > curConfid;
	vector<Net*> helperParents(3); //[0] is target, [1,2] are parameter
	vector<int> calcedClasses;
	int curGen = 0;
	vector<vector<double> > fullEpochData(trainingData.size());
	while(curGen < generations)
	{
		if(curTrainDataIndex == trainingData.size())
		{
			//end of an epoch
			for(int f = 0; f < fullEpochData.size(); f++)
			{
				fullEpochData[f] = *(trainingData[f]);
			}

			//run all indivs over train set w/out updating weights to see how their doing
			double maxAcc = 0.0;
			for(int i = 0; i < nets.size(); i++)
			{
				int numRight = 0;
				// nets[i]->finalize(); //allocate memory
				nets[i]->__dataPointer = &fullEpochData;
				nets[i]->run();
				nets[i]->getCalculatedClasses(calcedClasses);
				for(int j = 0; i < calcedClasses.size(); j++)
					if(calcedClasses[j] == trueVals[j])
						numRight++;
				double accuracy = 100.0 * numRight/calcedClasses.size();
				if(accuracy > maxAcc)
					maxAcc = accuracy;
			}

			//get new training data
			getTrainingData(trainingData, trueVals);
			curTrainDataIndex = 0;
		}

// struct task_basic_info t_info;
// mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

// if (KERN_SUCCESS != task_info(mach_task_self(),
//                               TASK_BASIC_INFO, (task_info_t)&t_info, 
//                               &t_info_count))
// {
//     return;
// }
// cout << "Mem before getFitness " << t_info.virtual_size << endl;

		//run nets over a piece of train data (get fitness)
		curTrainData[0] = *(trainingData[curTrainDataIndex]);
		printf("Training Data %d of %lu. Gen %d\n", curTrainDataIndex, trainingData.size(),curGen);
		curTrueVal = trueVals[curTrainDataIndex];
		curTrainDataIndex++;
		for(int i = 0; i < nets.size(); i++)
		{
			nets[i]->__dataPointer = &curTrainData;
			nets[i]->run();
			nets[i]->getConfidences(curConfid);
			netfit[i] = getFitness(curConfid[0], curTrueVal,nets[i]);
		}

		//get indivs for mutation
		for(int i = 0; i < nets.size(); i++)
		{
			//get target and parameter vectors
			int target = getTargetVector(__targetSelectionMethod,netfit,i);
			getHelperVectors(nets, target, i, helperParents); //fills helperParents

			//make donor vector. will be same structure as target
			Net* donor = makeDonor(helperParents,mutationScale);

			//apply crossover to get trial vector of same structure as original
			Net* trial = crossover(nets[i],donor, crossMethod, crossProb);

			//apply selection
			trial->setDevice(__device);
			trial->setTrainingType(TRAIN_AS_IS);
			trial->__dataPreprocessed = true;
			trial->__isTraining = true;
			if(!__preprocessIndividual) //this means preprocess collective
			{
				//store the mean and stddev for when we choose the best
				trial->preprocessCollectively();
				trial->__mean = __mean;
				trial->__stddev = __stddev;
			}
			trial->finalize(); //allocate memory
			trial->__dataPointer = &curTrainData;
			trial->run();
			trial->getConfidences(curConfid);
			double trialfit = getFitness(curConfid[0], curTrueVal,trial);

			if(trialfit < netfit[i]) //trial is better. backprop it over training data
			{
				delete nets[i];
				nets[i] = trial;
				nets[i]->__trainingData.resize(1);
				nets[i]->__trainingData[0].resize(1);
				nets[i]->__trainingData[0][0] = trainingData[curTrainDataIndex];
				nets[i]->__trueNames.clear();
				nets[i]->__trueNames.push_back(trueVals[curTrainDataIndex]);
				nets[i]->train(1);
				nets[i]->__trainingData.clear(); //don't want to delete the pointer
				//no need to copy the fitness b/c won't matter next time anyway
			}
			else 
			{
				delete trial;
			}
			delete donor;

			//only set up layerNeeds and velocities as needed for backprop cause
			//all the nets are different sizes
		}
	}

	//run all indivs over train set w/out updating weights to see how their doing
	double maxAcc = 0.0;
	int maxIndex = 0;
	setTrainingType(TRAIN_AS_IS); // so we get whole thing to test on
	getTrainingData(trainingData, trueVals);
	fullEpochData.resize(trainingData.size());
	for(int f = 0; f < fullEpochData.size(); f++)
	{
		fullEpochData[f] = *(trainingData[f]);
	}
	for(int i = 0; i < nets.size(); i++)
	{
		int numRight = 0;
		nets[i]->finalize(); //allocate memory
		nets[i]->__dataPointer = &fullEpochData;
		nets[i]->run();
		nets[i]->getCalculatedClasses(calcedClasses);
		for(int j = 0; i < calcedClasses.size(); j++)
			if(calcedClasses[j] == trueVals[j])
				numRight++;
		double accuracy = 100.0 * numRight/calcedClasses.size();
		if(accuracy > maxAcc)
		{
			maxAcc = accuracy;
			maxIndex = i;
		}
	}

}*/

/*****************************
*
* Trains CNNs using Differential Evolution. 
* Doesn't work as well as backprop or AntTrain
* @parameter dataBatchSize int If positive, that many examples per generation. If negative, all examples per generation (preferred).
* Public
*
*****************************/
/*void Net::DETrain_sameSize(int mutationType, int generations, int dataBatchSize, int population, double mutationScale, int crossMethod, double crossProb, bool BP)
{
	double mutationMax = 0.1, mutationMin = 0.1;
	double crossMax = 0.8, crossMin = 0.1;

	printf("DE training same size\n");
	BP = false;
	cl_int error;

	// int mutationType = mutationType;
	int numDiffVec;
	if(mutationType == DE_BEST || mutationType == DE_RAND)
		numDiffVec = 1;
	else if(mutationType == DE_CURRENT_TO_BEST)
		numDiffVec = 2;
	else if(mutationType == DE_QUIN_AND_SUGANTHAN)
		numDiffVec = -1; //set in makeDonor
	else
	{
		printf("Unknown mutationType: %d\n",mutationType);
		return;
	}

	//We'll need some stuff to hold our training data
	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	getTrainingData(trainingData, trueVals);
	int curTrainDataIndex = 0;
	if(dataBatchSize <= 0)
		dataBatchSize = trainingData.size();

	double percentTotalData = (double)dataBatchSize/trainingData.size();
	printf("percentTotalData = %lf\n",percentTotalData);
	// if(mutationMin*percentTotalData < 0.01)
	// {
	// 	mutationMin = .01;
	// 	mutationMax = mutationMax/mutationMin * mutationMin;
	// }
	// else
	// {
		mutationMax *= percentTotalData;
		mutationMin *= percentTotalData;
	// }
	// if(crossProb * percentTotalData < 0.01)
	// {
	// 	crossProb = 0.01;
	// }
	// else
	// {
		crossProb *= percentTotalData;
	// }
	

	//preprocess the data, training and test
 	if(__preprocessIndividual)
 	{
	 	if(!__trainingDataPreprocessed)
	 		//preprocessTrainingDataCollective();
	        preprocessTrainingDataIndividual();
	    if(__testData.size() != 0 && !__testDataPreprocessed)
	    	preprocessTestDataIndividual();
	}
	else // __preprocessCollective
	{
		if(!__trainingDataPreprocessed)
			preprocessTrainingDataCollective();
		if(!__testDataPreprocessed)
			preprocessTestDataCollective();
	}

	//We'll have a vector of weight holders to hold the population
	vector<Net*> nets(population);
	// vector<Net*>* netsPtr = &nets;
	vector<double> netfit(population, 0);
	vector<int> netCorrect(population, 0);
	vector<int> prevCorrect(population, 0);
	vector<double> prevfit(population, 0);
	vector<Net*> trials(population);
	vector<double> trialfit(population, 0);

	setupEquivalentNets(nets);
	vector<vector<double> > curConfid;
	vector<Net*> helperParents(3); //[0] is target, [1,2] are parameter
	vector<int> calcedClasses;
	int curGen = 0;

	//set up clBatchSize stuff
	printf("set up clBatchSize stuff\n");
	int clBatchSize = 15;
	vector<vector<cl_mem> > layerNeeds(clBatchSize), clWeights(clBatchSize), clBiases(clBatchSize), velocities(population);
	vector<cl_mem> p(clBatchSize), n(clBatchSize);
	vector<cl_mem*> prevNeurons(clBatchSize), neurons(clBatchSize);
	vector<cl_command_queue> queues(clBatchSize);
	vector<cl_mem> denoms(clBatchSize);
	vector<Kernels> kerns(clBatchSize);

	int q = __device == -1 ? 0: __device;



	int __maxNeuronSize = 0;
	for(int a = 0; a< clBatchSize; a++)
	{
		setupLayerNeeds(layerNeeds[a]);
		for(int i=1; i < __layers.size(); i++)
		{
			if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer *conv = (ConvLayer*) __layers[i];

				clWeights[a].push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE,
					sizeof(double) * conv->numWeights, nullptr, &error));
				CheckError(error);
				clBiases[a].push_back(clCreateBuffer(__context, CL_MEM_READ_WRITE,
					sizeof(double) * conv->numBiases, nullptr, &error));
				CheckError(error);
				//none of the other layers have the ability to increase the size from the layer before
				if(conv->maxSizeNeeded > __maxNeuronSize)
				{
					__maxNeuronSize = conv->maxSizeNeeded;
				}
			}
		}

		p[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		prevNeurons[a] = &(p[a]);
		n[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize, nullptr, &error);
		CheckError(error);
		neurons[a] = &(n[a]);

		queues[a] = clCreateCommandQueue(__context, __deviceIds[q], 0, &error);
		CheckError(error);
		// if(a % 2 == 0)
		// 	queues[a] = clCreateCommandQueue(__context, __deviceIds[2], 0, &error);
		// else
		// 	queues[a] = clCreateCommandQueue(__context, __deviceIds[1], 0, &error);
		// CheckError(error);

		denoms[a] = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);
		CheckError(error);

		buildKernels(kerns[a], q);
	}
	for(int a = 0; a < population; a++)
	{
		initVelocities(velocities[a]);
	}


	//set some softmax related args that won't change
	for(int i = 0; i < kerns.size(); i++)
	{
	 	clSetKernelArg(kerns[i].maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
	 	clSetKernelArg(kerns[i].vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	}

 	vector<double> soft(__neuronSizes.back());
 	vector<thread> thr(clBatchSize);

 	printf("Starting gens\n");
 	time_t genstarttime = time(NULL);
 	int starttime = genstarttime;
 	bool firstTime = true;
	while(curGen <= generations)
	{
		******************************
		*
		*  END OF AN EPOCH/GENERATION
		*
		******************************//*
		if(curTrainDataIndex == trainingData.size() || curGen == generations || firstTime)
		// if(true)
		{
			if(firstTime)
				firstTime = false;
			else
			{
				curGen++;
				printf("Time for a generation: %s\n", convnet::secondsToString(time(NULL) - genstarttime).c_str());
			}
			printf("---------------------------------------------------------------\n");
			time_t starttime = time(NULL);
			
			//end of an epoch
			for(int i =0; i < netCorrect.size(); i++)
			{
				netCorrect[i] = 0;
				netfit[i] = 0;
			}

			//run all indivs over train set w/out updating weights to see how their doing
			for(int i = 0; i < nets.size();)
			{
				int myclBatchSize;
				if(i + clBatchSize >= nets.size())
					myclBatchSize = nets.size() - i;
				else
					myclBatchSize = clBatchSize;
				if(myclBatchSize == 0)
				{
					printf("Error: myclBatchSize == 0. Infinite loop. Aborting.\n");
					return;
				}
				
				for(int t = 0; t < myclBatchSize; t++)
				{
					pushCLWeights(nets[i+t]->__layers, clWeights[t], clBiases[t], queues[t], CL_FALSE);
				}
				for(int data = 0; data < trainingData.size(); data++)
				{
					for(int t = 0; t < myclBatchSize; t++)
					{
						// continue;
						CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes[0], trainingData[data]->data(),
							0, nullptr, nullptr));
						
						clFinish(queues[t]);
						//this does both feedForward and softmaxForward
						cl_mem ** prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
						thr[t] = thread([=] {feedForward(prevptr,nptr,ref(nets[i+t]->__neuronDims), ref(nets[i+t]->__layers), ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), std::ref(kerns[t]));});

					}
					for(int t = 0; t < myclBatchSize; t++)
						thr[t].join();

					for(int t = 0; t < myclBatchSize; t++)
					{
						//get output and see if right
						CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
						 	soft.data(), 0, nullptr, nullptr));
						netfit[i+t] += getFitness(soft, trueVals[data],nets[i+t]);
						if(getMaxElementIndex(soft) == trueVals[data])
							netCorrect[i+t]++;
					}
				}
				for(int t = 0; t < myclBatchSize; t++)
				{
					if(netCorrect[i+t] > prevCorrect[i+t]) 	printf("*");
					else 									printf(" ");
					if(netfit[i+t] < prevfit[i+t] - 0.0001)	printf("*");
					else 									printf(" ");
					prevCorrect[i+t] = netCorrect[i+t];
					prevfit[i+t] = netfit[i+t];
					printf("Net: %3d. Acc: %7.3lf. Error %lf\n", i+t, 100.0*netCorrect[i+t]/trainingData.size(), netfit[i+t]);
				}

				i+=myclBatchSize;
			}

			int best = getMaxElementIndex(netCorrect);
			int bestErrorIndex = 0;
			double bestError = netfit[0];
			for(int i = 1; i < netfit.size(); i++)
				if(netfit[i] < bestError)
				{
					bestError = netfit[i];
					bestErrorIndex = i;
				}
			printf("End Gen %4d. Best accuracy is: Net %d, %lf, Error %lf\n", curGen, best, 100.0*netCorrect[best]/trainingData.size(),netfit[best]);
			printf("              Best error is:    Net %d, %lf, Error %lf |", bestErrorIndex, 100.0*netCorrect[bestErrorIndex]/trainingData.size(),netfit[bestErrorIndex]);
			printf("Time to run all training over all indivs: %s\n", convnet::secondsToString(time(NULL)-starttime).c_str());
			printf("---------------------------------------------------------------\n");

			if(curGen == generations)
			{
				if(__saveNet)
					nets[best]->save(__saveName.c_str());
				break;
			}
			//get new training data
			getTrainingData(trainingData, trueVals);
			curTrainDataIndex = 0;
			genstarttime = time(NULL);

		}

		******************************
		*
		*  START OF DE TRAINING PER IMAGE
		*
		******************************//*

		if(curTrainDataIndex % 1000 == 0 && curTrainDataIndex != 0)
		{
			printf("Training Data %d of %lu. Gen %d | ", curTrainDataIndex, trainingData.size(),curGen);
			printf("Time for 1000: %s\n", convnet::secondsToString(time(NULL)-starttime).c_str());
			starttime = time(NULL);
		}
		else if(curTrainDataIndex == 0)
			starttime = time(NULL); //cause we had to wait for it to run over all the images

		int mydataBatchSize = curTrainDataIndex + dataBatchSize >= trainingData.size() ? trainingData.size() - curTrainDataIndex : dataBatchSize;

		//run nets over a piece of train data (get fitness)
		for(int i = 0; i < netfit.size(); i++)
			netfit[i] = 0;
		for(int i = 0; i < nets.size();)
		{
			int myclBatchSize;
			if(i + clBatchSize >= nets.size())
				myclBatchSize = nets.size() - i;
			else
				myclBatchSize = clBatchSize;

			vector<thread> thr(myclBatchSize);
			// printf("myclBatchSize = %d\n",myclBatchSize);
			if(myclBatchSize == 0)
			{
				printf("Error: myclBatchSize == 0. Infinite loop. Aborting.\n");
				return;
			}
			//put current weights in OpenCL
			for(int t = 0; t < myclBatchSize; t++)
				pushCLWeights(nets[i+t]->__layers, clWeights[t], clBiases[t], queues[t], CL_TRUE);
			for(int d = 0; d < mydataBatchSize; d++)
			{
				for(int t = 0; t < myclBatchSize; t++)
				{
					// continue;
					CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes[0], trainingData[curTrainDataIndex+d]->data(),
						0, nullptr, nullptr));
					clFinish(queues[t]);
					//this does both feedForward and softmaxForward
					cl_mem ** prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
					thr[t] = thread([=] {feedForward(prevptr, nptr,ref(nets[i+t]->__neuronDims), ref(nets[i+t]->__layers), ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
				}
				for(int t = 0; t < myclBatchSize; t++)
					thr[t].join();

				for(int t = 0; t < myclBatchSize; t++)
				{
					//get output and see if right
					CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
					 	soft.data(), 0, nullptr, nullptr));
					netfit[i+t] += getFitness(soft, trueVals[curTrainDataIndex + d],nets[i+t]);
				}
			}

			i+=myclBatchSize;
		}

		******************************
		*
		*  GETTING DONOR AND TRIAL VECTORS SEQUENTIALLY
		*
		******************************//*
		double mutationCurrent = mutationMin + (mutationMax - mutationMin)*(generations - curGen)/generations;
		crossProb = crossMin + (crossMax - crossMin) * (generations - curGen)/generations;
		if(mutationType != DE_QUIN_AND_SUGANTHAN)
		{
			for(int i = 0; i < nets.size(); i++)
			{
				trialfit[i] = 0;
				// printf("starting donor\n");
				Net* donor = makeDonor(mutationType, nets, netfit, i, numDiffVec, mutationCurrent);
				// printf("starting crossover\n");
				trials[i] = crossover(nets[i],donor, crossMethod, crossProb);

				delete donor;
			}
			for(int i = 0; i < nets.size();)
			{
				int myclBatchSize;
				if(i + clBatchSize >= nets.size())
					myclBatchSize = nets.size() - i;
				else
					myclBatchSize = clBatchSize;

				vector<thread> thr(myclBatchSize);

				for(int t = 0; t < myclBatchSize; t++)
					pushCLWeights(trials[i+t]->__layers, clWeights[t], clBiases[t], queues[t], CL_TRUE);
				for(int d = 0; d < mydataBatchSize; d++)
				{
					for(int t = 0; t < myclBatchSize; t++)
					{
						// continue;
						CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes[0], trainingData[curTrainDataIndex+d]->data(),
							0, nullptr, nullptr));
						clFinish(queues[t]);
						//this does both feedForward and softmaxForward
						cl_mem ** prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
						thr[t] = thread([=] {feedForward(prevptr, nptr,ref(trials[i+t]->__neuronDims), ref(trials[i+t]->__layers), ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
					}
					for(int t = 0; t < myclBatchSize; t++)
						thr[t].join();

					for(int t = 0; t < myclBatchSize; t++)
					{
						//get output and see if right
						CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
						 	soft.data(), 0, nullptr, nullptr));
						trialfit[i+t] += getFitness(soft, trueVals[curTrainDataIndex + d],trials[i+t]);
					}
				}
				// printf("trial ran\n");
				i+= myclBatchSize;

			} //got all trials

			// printf("donors made:\n");
			// for(int i = 0; i < r1v.size(); i++)
			// 	printf("%2d %2d\n",r1v[i],r2v[i]);
			// r1v.clear();
			// r2v.clear();
			// getchar();
			//figure out next generation individuals / apply selection
			for(int i = 0; i < nets.size(); i++)
			{
				
				if(trialfit[i] < netfit[i]) //trial is better. backprop it over training data
				{
					delete nets[i];
					nets[i] = trials[i];
					// printf("trial better\n");
					if(BP)
					{
						cl_mem ** prevptr = &(prevNeurons[0]), **nptr = &(neurons[0]);
						backprop(trueVals[curTrainDataIndex], prevptr, nptr, nets[i], layerNeeds[0], velocities[i], clWeights[0], clBiases[0], queues[0], kerns[0]);
						pullCLWeights(nets[i], clWeights[0], queues[0]);
					}
					// printf("*");			
				}
				else 
				{
					// printf("trial not better\n");
					delete trials[i];
					// printf(" ");
				}
				// printf("Net %d netfit %lf trialfit %lf\n", i, netfit[i], trialfit[i]);
				// printf("nets[%d] @ %#x\n", i,nets[i]);
			}
		}
		else //mutationType == DE_QUIN_AND_SUGANTHAN
		{
			vector<double> randfit(population,0), curbestfit(population, 0);
			vector<Net*> randtrial(population), curbesttrial(population);
			//get both sets of donor vectors and trial vectors
			// printf("getting rand and curbest donors and trials\n");
			for(int i = 0; i < nets.size(); i++)
			{
				Net* randDonor = makeDonor(DE_RAND,nets,netfit,i,1,mutationCurrent);
				Net* curBestDonor = makeDonor(DE_CURRENT_TO_BEST,nets,netfit,i,2,mutationCurrent);

				randtrial[i] = crossover(nets[i],randDonor,crossMethod,crossProb);
				curbesttrial[i] = crossover(nets[i],curBestDonor,crossMethod,crossProb);

				delete randDonor;
				delete curBestDonor;
			}
			//run both sets of trial vectors
			// printf("running rand trials\n");
			for(int i = 0; i < nets.size();)
			{
				int myclBatchSize;
				if(i + clBatchSize >= nets.size())
					myclBatchSize = nets.size() - i;
				else
					myclBatchSize = clBatchSize;

				vector<thread> thr(myclBatchSize);
				//rand
				for(int t = 0; t < myclBatchSize; t++)
					pushCLWeights(randtrial[i+t]->__layers, clWeights[0], clBiases[0], queues[0], CL_TRUE);
				for(int d = 0; d < mydataBatchSize; d++)
				{
					for(int t = 0; t < myclBatchSize; t++)
					{
						// continue;
						CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes[0], trainingData[curTrainDataIndex+d]->data(),
							0, nullptr, nullptr));
						//run trial over data and get fitness of trial
						cl_mem ** prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
						thr[t] = thread([=] {feedForward(prevptr,nptr,ref(randtrial[i+t]->__neuronDims), ref(randtrial[i+t]->__layers), 
								ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
					}
					for(int t = 0; t < myclBatchSize; t++)
						thr[t].join();
					for(int t = 0; t < myclBatchSize; t++)
					{
						CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
							 	soft.data(), 0, nullptr, nullptr));
						randfit[i+t] += getFitness(soft, trueVals[curTrainDataIndex+d],randtrial[i+t]);
					}
				}
				i+=myclBatchSize;
			}
			// printf("running curbest trials\n");
			for(int i = 0; i < nets.size();)
			{
				int myclBatchSize;
				if(i + clBatchSize >= nets.size())
					myclBatchSize = nets.size() - i;
				else
					myclBatchSize = clBatchSize;

				vector<thread> thr(myclBatchSize);
				//curbest
				pushCLWeights(curbesttrial[i]->__layers, clWeights[0], clBiases[0], queues[0], CL_TRUE);
				for(int d = 0; d < mydataBatchSize; d++)
				{
					for(int t = 0; t < myclBatchSize; t++)
					{
						// continue;
						CheckError(clEnqueueWriteBuffer(queues[t], *(prevNeurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes[0], trainingData[curTrainDataIndex+d]->data(),
							0, nullptr, nullptr));
						//run trial over data and get fitness of trial
						cl_mem ** prevptr = &(prevNeurons[t]), **nptr = &(neurons[t]);
						thr[t] = thread([=]{feedForward(prevptr,nptr,ref(curbesttrial[i+t]->__neuronDims), ref(curbesttrial[i+t]->__layers), 
								ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(queues[t]), ref(denoms[t]), ref(kerns[t]));});
					}
					for (int t = 0; t < myclBatchSize; ++t)
						thr[t].join();
					for(int t = 0; t < myclBatchSize; t++)
					{
						CheckError(clEnqueueReadBuffer(queues[t], *(neurons[t]), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
							 	soft.data(), 0, nullptr, nullptr));
						curbestfit[i+t] += getFitness(soft, trueVals[curTrainDataIndex+d],curbesttrial[i+t]);
					}
				}
				i+=myclBatchSize;
			}
			//calculate survivals and deaths of offspring
			// printf("calc survivals\n");
			int nsrand = 0, nscurbest = 0;// don't think we actually need these: nfrand = 0, nfcurbest = 0;
			for(int i = 0; i < nets.size(); i++)
			{
				if(randfit[i] < netfit[i])
					nsrand++;
				if(curbestfit[i] < netfit[i])
					nscurbest++;
			}
			//calculate probability of choosing DE/rand/1/z
			double probRand = nsrand * population/((double)nscurbest * population + nsrand * population);
			uniform_real_distribution<double> dis(0.0, 1.0);
			default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			if(probRand < dis(gen))
			{
				printf("rand survive\n");
				for(int i = 0; i < curbesttrial.size(); i++)
					delete curbesttrial[i];
				for(int i = 0; i < nets.size(); i++)
					if(randfit[i] < netfit[i])
					{
						delete nets[i];
						nets[i] = randtrial[i];
						// printf("trial better\n");
						if(BP)
						{
							cl_mem ** prevptr = &(prevNeurons[0]), **nptr = &(neurons[0]);
							backprop(trueVals[curTrainDataIndex], prevptr, nptr, nets[i], layerNeeds[0], velocities[i], clWeights[0], clBiases[0], queues[0], kerns[0]);
							pullCLWeights(nets[i], clWeights[0], queues[0]);
						}	
					}
					else
						delete randtrial[i];
			}
			else
			{
				printf("cur2best survive\n");
				for(int i = 0; i < randtrial.size(); i++)
					delete randtrial[i];
				for(int i = 0; i < nets.size(); i++)
					if(curbestfit[i] < netfit[i])
					{
						delete nets[i];
						nets[i] = curbesttrial[i];
						// printf("trial better\n");
						if(BP)
						{
							cl_mem ** prevptr = &(prevNeurons[0]), **nptr = &(neurons[0]);
							backprop(trueVals[curTrainDataIndex], prevptr, nptr, nets[i], layerNeeds[0], velocities[i], clWeights[0], clBiases[0], queues[0], kerns[0]);
							pullCLWeights(nets[i], clWeights[0], queues[0]);
						}	
					}
					else
						delete curbesttrial[i];
			}
		}



		******************************
		 *
		 *  RUNNING TRIAL VECTORS CONCURRENTLY
		 *
		 ******************************//*
		// for(int i = 0; i < nets.size();)
		// {
		// 	//threaded
		// 	int myclBatchSize;
		// 	if(i + clBatchSize >= nets.size())
		// 		myclBatchSize = nets.size() - i;
		// 	else
		// 		myclBatchSize = clBatchSize;

		// 	vector<thread> thr(myclBatchSize);
		// 	// printf("myclBatchSize = %d\n",myclBatchSize);
		// 	if(myclBatchSize == 0)
		// 	{
		// 		printf("Error: myclBatchSize == 0. Infinite loop. Aborting.\n");
		// 		return;
		// 	}
		// 	for(int t = 0; t < myclBatchSize; t++)
		// 	{
		// 		thr[t] = thread([=] {trial_thread(i+t, netsPtr, netfit[i+t], trials[i+t], trainingData[curTrainDataIndex]->data(), curTrueVal, prevNeurons[t],
		// 			neurons[t], ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(velocities[t]), ref(queues[t]), ref(denoms[t]), 
		// 			ref(kerns[t]), BP);});
		// 	}
		// 	for(int t = 0; t < myclBatchSize; t++)
		// 		thr[t].join();
		// 	i+=myclBatchSize;

		 // try this sequential next
			//sequential
			// int t = 0;
			// trial_thread(i+t, netsPtr, netfit[i+t], trials[i+t], trainingData[curTrainDataIndex]->data(), curTrueVal, prevNeurons[t],
			// 		neurons[t], ref(layerNeeds[t]), ref(clWeights[t]), ref(clBiases[t]), ref(velocities[t]), ref(queues[t]), ref(denoms[t]), 
			// 		ref(kerns[t]), BP);
			// i++;
		// }
		// curTrainDataIndex++;
		curTrainDataIndex += mydataBatchSize;
	}


	for(int i = 0; i < clBatchSize; i++) //they are all the same size
	{	
		for(int j = 0; j < layerNeeds[i].size(); j++)
			clReleaseMemObject(layerNeeds[i][j]);
		for(int j = 0; j < clWeights[i].size(); j++)
			clReleaseMemObject(clWeights[i][j]);
		for(int j = 0; j < clBiases[i].size(); j++)
			clReleaseMemObject(clBiases[i][j]);
		clReleaseMemObject(p[i]);
		clReleaseMemObject(n[i]);
		clReleaseCommandQueue(queues[i]);
		clReleaseMemObject(denoms[i]);
		//releaseKernels(kerns[i]);
	}
}*/

/*****************************
*
* DO NOT USE
* Meant to be used with DETrain_sameSize to create and run trial vectors concurrently
* Has weird bug in it that will slow down system the longer it runs. The slow down is persistant until machine restart.
* Private
*
*****************************/
/*void Net::trial_thread(int netIndex, vector<Net*>* nets, double netfit, Net* trial, double* trainDataPtr, int curTrueVal, cl_mem** prevNeurons, 
	cl_mem** neurons, const vector<cl_mem>& layerNeeds, const vector<cl_mem>& clWeights, const vector<cl_mem>& clBiases, 
	const vector<cl_mem>& velocities, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k, bool BP)
{
	// printf("trial_thread method\n");
	vector<double> soft(trial->__neuronSizes.back());

	// printf("write buf\n");
	CheckError(clEnqueueWriteBuffer(queue, **prevNeurons, CL_TRUE, 0, sizeof(double) * trial->__neuronSizes[0], trainDataPtr,
		0, nullptr, nullptr));
	// printf("pull cl \n");
	pushCLWeights(trial->__layers, clWeights, clBiases, queue, CL_TRUE);
	clFinish(queue);
	//run trial over data and get fitness of trial
	// printf("feed forward\n");
	feedForward(prevNeurons,neurons,trial->__neuronDims, trial->__layers, 
			layerNeeds, clWeights, clBiases, queue, denom, k);
	// printf("read buf\n");
	// printf("neurons is nullptr: %d\n", neurons == nullptr);
	// printf("soft size: %lu\n", soft.size());
	CheckError(clEnqueueReadBuffer(queue, **neurons, CL_TRUE, 0, sizeof(double) * trial->__neuronSizes.back(),
		 	soft.data(), 0, nullptr, nullptr));
	// printf("get fit\n");
	double trialfit = getFitness(soft, curTrueVal,(*nets)[netIndex]);
	// printf("trial ran\n");

	
	//apply selection
	if(trialfit < netfit) //trial is better. backprop it over training data
	{
		// printf("trial better\n");
		delete (*nets)[netIndex];
		nets->at(netIndex) = trial;
		//backprop
		if(BP)
		{
			// printf("netIndex BP\n");
			//update weights using backprop
			backprop(curTrueVal, prevNeurons, neurons, (*nets)[netIndex], layerNeeds, velocities, clWeights, clBiases, queue, k);
			// printf("end BP\n");
			//pull new weights from cl_mem back into Net
			pullCLWeights((*nets)[netIndex], clWeights, queue);
			// printf("end pull\n");
		}
	}
	else 
	{
		// printf("parent better\n");
		delete trial;
	}
}*/

/*****************************
*
* Builds all the CNN kernels into a Kernels object for specified device.
* Private
*
*****************************/
void Net::buildKernels(Kernels& k, int device)
{
	#ifdef _DEBUG
	printf("Start buildKernels train(%s) run(%s)... ",CNTrainingPath.c_str(), CNForwardPath.c_str());
	#endif
	cl_int error;
	if(k.built)
		return;
	// const char* foptions = "-cl-mad-enable";
	// const cl_device_id* deviceToBuild = &(__deviceIds[device]);
	__program_creation_mutex.lock();
	if(!__programs_already_created)
	{
		CNForward = CreateProgram(LoadKernel(CNForwardPath.c_str()), __context, RUNNING_PROGRAM);
		CheckError(clBuildProgram(CNForward, __deviceIds.size(), __deviceIds.data(), nullptr, nullptr, nullptr));
		//training
		CNTraining = CreateProgram(LoadKernel(CNTrainingPath.c_str()), __context, TRAINING_PROGRAM);
		CheckError(clBuildProgram(CNTraining, __deviceIds.size(), __deviceIds.data(), nullptr, nullptr, nullptr));
		__programs_already_created = true;
	}
	__program_creation_mutex.unlock();

	k.reluKernelF = clCreateKernel(CNForward, "relu", &error); CheckError(error);
	k.leakyReluKernelF = clCreateKernel(CNForward, "leakyRelu", &error); CheckError(error);
	k.convKernelF = clCreateKernel(CNForward, "convolve", &error); CheckError(error);
	k.convKernelFC = clCreateKernel(CNForward, "convolveConstant", &error); CheckError(error);
	k.zeroPadKernelF = clCreateKernel(CNForward, "zeroPad", &error); CheckError(error);
	k.maxPoolKernelF = clCreateKernel(CNForward, "maxPool", &error); CheckError(error);
	k.softmaxKernelF = clCreateKernel(CNForward, "softmax_allCL", &error); CheckError(error);
	k.reluKernel = clCreateKernel(CNTraining, "relu", &error); CheckError(error);
	k.reluBackKernel = clCreateKernel(CNTraining, "relu_back", &error); CheckError(error);
	k.leakyReluKernel = clCreateKernel(CNTraining, "leakyRelu", &error); CheckError(error);
	k.leakyReluBackKernel = clCreateKernel(CNTraining, "leakyRelu_back", &error); CheckError(error);
	k.convKernel = clCreateKernel(CNTraining, "convolve", &error); CheckError(error);
	k.convBackNeuronsKernel = clCreateKernel(CNTraining, "convolve_back_neurons", &error); CheckError(error);
	k.convBackBiasesKernel = clCreateKernel(CNTraining, "convolve_back_biases", &error); CheckError(error);
	k.convBackWeightsKernel = clCreateKernel(CNTraining, "convolve_back_weights", &error); CheckError(error);
	k.convBackWeightsMomentKernel = clCreateKernel(CNTraining, "convolve_back_weights_moment", &error); CheckError(error);
	k.zeroPadKernel = clCreateKernel(CNTraining, "zeroPad", &error); CheckError(error);
	k.zeroPadBackKernel = clCreateKernel(CNTraining, "zeroPad_back", &error); CheckError(error);
	k.maxPoolKernel = clCreateKernel(CNTraining, "maxPool", &error); CheckError(error);
	k.maxPoolBackKernel = clCreateKernel(CNTraining, "maxPool_back", &error); CheckError(error);
	k.softmaxKernel = clCreateKernel(CNTraining, "softmax", &error); CheckError(error);
	k.softmaxBackKernel = clCreateKernel(CNTraining, "softmax_back", &error); CheckError(error);
	k.copyArrayKernel = clCreateKernel(CNTraining, "copyArray", &error); CheckError(error);
	k.maxSubtractionKernel = clCreateKernel(CNTraining, "maxSubtraction", &error); CheckError(error);
	k.vectorESumKernel = clCreateKernel(CNTraining, "vectorESum", &error); CheckError(error);
	k.plusEqualsKernel = clCreateKernel(CNTraining, "plusEquals", &error); CheckError(error);
	k.divideEqualsKernel = clCreateKernel(CNTraining, "divideEquals", &error); CheckError(error);
	k.zeroMemKernel = clCreateKernel(CNTraining, "zero_out", &error); CheckError(error);
	k.convBackWeightsNoUpdateAccumKernel = clCreateKernel(CNTraining, "convolve_back_weights_no_update_accum", &error); CheckError(error);
	k.convBackBiasesNoUpdateAccumKernel = clCreateKernel(CNTraining, "convolve_back_biases_no_update_accum", &error); CheckError(error);
	k.updateWeightsKernel = clCreateKernel(CNTraining, "update_weights", &error); CheckError(error);
	k.updateWeightsMomentKernel = clCreateKernel(CNTraining, "update_weights_moment", &error); CheckError(error);
	k.updateBiasesKernel = clCreateKernel(CNTraining, "update_biases", &error); CheckError(error);
	k.batchNormRunKernel = clCreateKernel(CNTraining, "batch_norm_run", &error); CheckError(error);
	k.batchNormKernel = clCreateKernel(CNTraining, "batch_norm", &error); CheckError(error);
	k.batchNormBackKernel = clCreateKernel(CNTraining, "batch_norm_back", &error); CheckError(error);
	k.updateGammaAndBetaKernel = clCreateKernel(CNTraining, "update_gamma_and_beta", &error); CheckError(error);

	k.built = true;
	#ifdef _DEBUG
	printf("done\n");
	#endif
}

/*****************************
*
* Releases all kernels in a given Kernels object. MUST be called when done in order to prevent memory leak
* Private
*
*****************************/
// void Net::releaseKernels(Kernels& k)
// {
// 	clReleaseKernel(k.reluKernelF);
// 	clReleaseKernel(k.leakyReluKernelF);
// 	clReleaseKernel(k.convKernelF);
// 	clReleaseKernel(k.convKernelFC);
// 	clReleaseKernel(k.zeroPadKernelF);
// 	clReleaseKernel(k.maxPoolKernelF);
// 	clReleaseKernel(k.softmaxKernelF);
// 	clReleaseKernel(k.reluKernel);
// 	clReleaseKernel(k.reluBackKernel);
// 	clReleaseKernel(k.leakyReluKernel);
// 	clReleaseKernel(k.leakyReluBackKernel);
// 	clReleaseKernel(k.convKernel);
// 	clReleaseKernel(k.convBackNeuronsKernel);
// 	clReleaseKernel(k.convBackBiasesKernel);
// 	clReleaseKernel(k.convBackWeightsKernel);
// 	clReleaseKernel(k.convBackWeightsMomentKernel);
// 	clReleaseKernel(k.zeroPadKernel);
// 	clReleaseKernel(k.zeroPadBackKernel);
// 	clReleaseKernel(k.maxPoolKernel);
// 	clReleaseKernel(k.maxPoolBackKernel);
// 	clReleaseKernel(k.softmaxKernel);
// 	clReleaseKernel(k.softmaxBackKernel);
// 	clReleaseKernel(k.copyArrayKernel);
// 	clReleaseKernel(k.maxSubtractionKernel);
// 	clReleaseKernel(k.vectorESumKernel);
// 	clReleaseKernel(k.plusEqualsKernel);
// 	clReleaseKernel(k.divideEqualsKernel);
// 	clReleaseKernel(k.zeroMemKernel);
// 	clReleaseKernel(k.convBackWeightsNoUpdateAccumKernel);
// 	clReleaseKernel(k.convBackBiasesNoUpdateAccumKernel);
// 	clReleaseKernel(k.updateWeightsKernel);
// 	clReleaseKernel(k.updateWeightsMomentKernel);
// 	clReleaseKernel(k.updateBiasesKernel);
// 	clReleaseKernel(k.batchNormKernel);
// 	clReleaseKernel(k.batchNormBackKernel);
// }

Net::Kernels::~Kernels()
{
	if(!built)
		return;

	// releaseKernels(*this);
	clReleaseKernel(reluKernelF);
	clReleaseKernel(leakyReluKernelF);
	clReleaseKernel(convKernelF);
	clReleaseKernel(convKernelFC);
	clReleaseKernel(zeroPadKernelF);
	clReleaseKernel(maxPoolKernelF);
	clReleaseKernel(softmaxKernelF);
	clReleaseKernel(reluKernel);
	clReleaseKernel(reluBackKernel);
	clReleaseKernel(leakyReluKernel);
	clReleaseKernel(leakyReluBackKernel);
	clReleaseKernel(convKernel);
	clReleaseKernel(convBackNeuronsKernel);
	clReleaseKernel(convBackBiasesKernel);
	clReleaseKernel(convBackWeightsKernel);
	clReleaseKernel(convBackWeightsMomentKernel);
	clReleaseKernel(zeroPadKernel);
	clReleaseKernel(zeroPadBackKernel);
	clReleaseKernel(maxPoolKernel);
	clReleaseKernel(maxPoolBackKernel);
	clReleaseKernel(softmaxKernel);
	clReleaseKernel(softmaxBackKernel);
	clReleaseKernel(copyArrayKernel);
	clReleaseKernel(maxSubtractionKernel);
	clReleaseKernel(vectorESumKernel);
	clReleaseKernel(plusEqualsKernel);
	clReleaseKernel(divideEqualsKernel);
	clReleaseKernel(zeroMemKernel);
	clReleaseKernel(convBackWeightsNoUpdateAccumKernel);
	clReleaseKernel(convBackBiasesNoUpdateAccumKernel);
	clReleaseKernel(updateWeightsKernel);
	clReleaseKernel(updateWeightsMomentKernel);
	clReleaseKernel(updateBiasesKernel);
	clReleaseKernel(batchNormKernel);
	clReleaseKernel(batchNormBackKernel);
	clReleaseKernel(batchNormRunKernel);
	clReleaseKernel(updateGammaAndBetaKernel);
}

/*****************************
*
* Creates a number of new Net objects equivalent to *this except having different weights. Meant for DETrain.
* The size of the input vector is how many nets it will make. 
* new is called, so function user should call delete when done with Nets to prevent memory leak.
* Private
*
*****************************/
void Net::setupEquivalentNets(vector<Net*>& nets)
{
	// printf("setupEquivalentNets\n");
	for(int i = 0; i < nets.size(); i++)
	{
		// printf("i %d\n", i);
		nets[i] = new Net(*this);
		if(nets[i] == nullptr)
		{
			printf("nullptr! setupEquivalentNets\n");
			getchar();
		}
		// printf("= done\n");
		for(int l = 1; l < nets[i]->__layers.size();l++)
		{
			// printf("l %d\n", l);
			if(nets[i]->__layers[l]->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)nets[i]->__layers[l];
				delete conv->weights;
				delete conv->biases;
				initRandomWeights(conv,nets[i]->__neuronDims[l-1][2]);
			}
		}
	}
	// printf("end setupEquivalentNets\n");
}

/*****************************
*
* Probably don't use unless doing stuff with DETrain (not sameSize one)
* Creates a number of Net objects with different structure and weights than *this. Created Nets will be mappable to *this.
* The size of the input vector is how many nets will be made.
* new is called, so you (or calling function) should call delete yourself
* Private
*
*****************************/
void Net::setupRandomNets(vector<Net*>& nets)
{
	vector<MaxPoolLayer*> maxs;
	for(int i = 0; i < __layers.size(); i++)
		if(__layers[i]->layerType == MAX_POOL_LAYER)
			maxs.push_back((MaxPoolLayer*)__layers[i]);
	uniform_real_distribution<double> dis(0.0, 1.0);
	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> filsize_dis(0,4); //need *2 + 1. filterSizes are 1-9

	for(int i = 0; i < nets.size(); i++)
	{
		nets[i] = new Net(__neuronDims[0][0],__neuronDims[0][1],__neuronDims[0][2]);
		//make a bunch of conv layers then the maxpool
		for(int j = 0; j < maxs.size(); j++)
		{
			double prob = .7;
			while(dis(gen) < prob)
			{
				//keep less filters in shallower layers b/c faster
				uniform_int_distribution<int> numFils_dis(1,32 * (i+1)); 
				int numfils = numFils_dis(gen);

				//the next two numbers are set so the width/height aren't changed
				int pad = filsize_dis(gen);
				int filsize = pad * 2 + 1;
				
				nets[i]->addConvLayer(numfils, 1, filsize, pad); // always has stride of 1
				prob *= .8;
			}
			nets[i]->addMaxPoolLayer(maxs[j]->poolSize, maxs[j]->stride);
		}
		double prob = .5;
		uniform_int_distribution<int> numFils_dis(1,1024);
		while(dis(gen) < prob)
		{
			nets[i]->addFullyConnectedLayer(numFils_dis(gen));
		}
		nets[i]->addFullyConnectedLayer(__neuronDims.back()[2]);
	}
}

/*****************************
*
* Gets fitness (Mean Squared Error (MSE)) of the net prediction on one training data
* @param prediction The output of the CNN (in same format as __confidences[i] for the ith output)
* @param trueVal The actual output (index of true class or the true numeric value being estimated)
* @param net The Net object associated with the prediction. Was used for testing, not needed for MSE.
* Private
*
*****************************/
double Net::getFitness(vector<double>& prediction, double trueVal, Net* net)
{
	//minimize weights (for testing)
	// double fit = 0;
	// unsigned long numWeights = 0;
	// for(int i = 0; i < net->__layers.size(); i++)
	// {
	// 	if(__layers[i]->layerType == CONV_LAYER)
	// 	{
	// 		ConvLayer* conv = (ConvLayer*)net->__layers[i];
	// 		// numWeights += conv->numWeights;
	// 		for(int j = 0; j < conv->numWeights; j++)
	// 			fit += fabs(conv->weights[j]);
	// 	}
	// }
	// // printf("%lf %lu\n", fit, numWeights);
	// return fit;
	//MSE
	double sum = 0;
	// printf("getfit\n");
	for(int i = 0; i < prediction.size(); i++)
	{
		// printf("%lf\n", prediction[i]);
		double diff;
		if(i == trueVal)
			diff = 1 - prediction[i]; //should be 1, is prediction[i]
		else
			diff = prediction[i]; //should be 0, is prediction[i]
		sum += diff;
	}
	return sum*sum;
}

/*****************************
*
* NOTE: This method might not contain newer Target selection methods.
* Sets the target selection method needed for DETrain_sameSize. For options see ConvNetCL.h under //defines for DE
* Returns true if the method number is valid, false otherwise
* Private
*
*****************************/
bool Net::setDETargetSelectionMethod(int method)
{
	if(method == DE_RAND || method == DE_BEST)
	{
		__targetSelectionMethod = method;
		return true;
	}
	return false;
}

/*****************************
*
* Returns the index of target vector to be used based on the fitness of the nets
* @param method The target selection method to be used.
* @param fits The fitness of the nets being used. Is a parallel vector to nets in DETrain_sameSize
* @param curNet Index of the current net that the target vector will be used against
* Private
*
*****************************/
int Net::getTargetVector(int method, const vector<double>& fits, int curNet)
{
	if(method == DE_BEST)
	{
		int bestIndex = (curNet == 0) ? 1 : 0; //don't want to count curNet in the running
		double bestFit = fits[bestIndex];
		for(int i = 0; i < fits.size(); i++)
			if(fits[i] < bestFit)
			{
				bestFit = fits[i];
				bestIndex = i;
			}
		return bestIndex;
	}
	else if(method == DE_RAND)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		uniform_int_distribution<int> dis(0,fits.size() - 2);
		default_random_engine gen(seed);
		int index = dis(gen);
		return (index >= curNet) ? index + 1 : index;
	}
	else
	{
		printf("Unknown method number %d\n", method);
		return -1;
	}
}

/*****************************
*
* Gets the 3 needed helper vectors for mutation in DETrain_sameSize and puts them into the helpers parameter
* @param nets Vector of Nets being used.
* @param target Index of target vector as gotten from getTargetVectors(...)
* @param curNet Index of current net helpers are being got for
* @param helpers Vector of Net* that the helper found will go into.
* Private
*
*****************************/
void Net::getHelperVectors(const vector<Net*>& nets, int target, int curNet, vector<Net*>& helpers)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	if(helpers.size() != 3)
	{
		printf("resizing helpers\n");
		helpers.resize(3);
	}
	if(nets.size() < 4)
	{
		printf("Should have at least 4 individuals for donor vector\n");
		helpers[0] = nets[target];
		helpers[1] = nets[0];
		helpers[2] = nets[0];
		return;
	}
	helpers[0] = nets[target];
	uniform_int_distribution<int> dis(0,nets.size()-1);
	default_random_engine gen(seed);
	int index1;
	do
	{
		index1 = dis(gen);
	} while(index1 == target || index1 == curNet);
	int index2;
	do
	{
		index2 = dis(gen);
	} while(index2 == target || index2 == curNet || index2 == index1);

	helpers[0] = nets[target];
	helpers[1] = nets[index1];
	helpers[2] = nets[index2];
	// printf("indexes %d %d %d\n", target, index1, index2);
}

/*****************************
*
* Maps Conv Layers to each other between different (or same) structured Nets.
* Used with Nets created by setupEquivalentNets() and/or setupRandomNets()
* @param orig Net* of original net
* @param layerNum The layer number of the layer to be mapped (0 is input layer, 1 is first hidden layer, etc)
* @param dest Net* of net to be mapped to
* @return int The layer number in the dest net that the layerNum layer in the orig net maps to
* Private
*
*****************************/
int Net::mapConvLayer(Net* orig, int layerNum, Net* dest)
{
	//find relative location in original net
	int maxsHit = 0, convsHit = 0;
	for(int i = 1; i <= layerNum; i++)
	{
		if(orig->__layers[i]->layerType == CONV_LAYER)
			convsHit++;
		else if(orig->__layers[i]->layerType == MAX_POOL_LAYER)
		{
			maxsHit++;
			convsHit = 0;
		}
	}

	//find corresponding location in new net
	int i = 1;
	int dmaxsHit = 0, dconvsHit = 0;
	while(dmaxsHit < maxsHit)
	{
		if(dest->__layers[i]->layerType == MAX_POOL_LAYER)
			dmaxsHit++;
 		i++;
	}
	while(dconvsHit < convsHit)
	{
		if(i >= dest->__layers.size())
		{
			return -1;
		}
		if(dest->__layers[i]->layerType == CONV_LAYER)
			dconvsHit++;
		i++;
	}
	i--;
	// printf("mapConvLayer: orig: %d. Mapped: %d\n", layerNum, i);
	return i;
}

/*****************************
*
* Makes and returns a donor vector for DETrain_sameSize.
* @param mutType The mutation type to be used (defines from the ConvNetCL.h)
* @param nets Vector of Net* containing nets from the current generation. const
* @param netfit Vector of doubles parallel to nets that contain fitnesses of nets from current generation. const
* @param curIndex Index of currently used net in nets
* @param n number of vectors to use for mutation. From DE/x/n/z
* @param scaleFactor The scaling factor, beta, that the mutation will be scaled by.
* @return A pointer to the donor vector made. Must be deleted by user to prevent memory leaks.
* Private
*
*****************************/
Net* Net::makeDonor(int mutType, const vector<Net*>& nets, const vector<double>& netfit, int curIndex, int n, double scaleFactor)
{
	if(mutType == DE_BEST)
	{
		vector<Net*> helpers(3);
		int target = getTargetVector(DE_BEST, netfit, curIndex);
		getHelperVectors(nets,target,curIndex,helpers);
		return makeDonor(helpers,scaleFactor);
	}
	else if(mutType == DE_RAND)
	{
		int target = getTargetVector(DE_RAND, netfit, curIndex);
		uniform_int_distribution<int> first(0,nets.size()-2);
		default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		int r1 = first(gen);
		int r2 = first(gen);
		if(r1 >= curIndex) r1++;
		if(r2 >= curIndex) r2++;
		// printf("i %d target %d r1 %d r2 %d\n", curIndex,target, r1,r2);
		// r1v.push_back(r1);
		// r2v.push_back(r2);
		return makeDonor({nets[target],nets[r1],nets[r2]},scaleFactor,false); //deep copy
	}
	else if(mutType == DE_CURRENT_TO_BEST)
	{
		// printf("current to best\n");
		// u = x_current + beta(x_hat - x_current) + beta*sum_1^n(x_k1 - x_k2)
		//first: x_current + beta(x_hat - x_i)
		int bestIndex = getTargetVector(DE_BEST, netfit, curIndex);
		Net* donor = makeDonor({nets[curIndex],nets[bestIndex],nets[curIndex]},scaleFactor); //this one is deep copy

		default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		uniform_int_distribution<int> first(0,nets.size()-2);
		uniform_int_distribution<int> second(0,nets.size()-3);
		for(int i = 0; i < n-1; i++)
		{
			// printf("curTobest i: %d\n", i);
			int in1 = first(gen);
			if(in1 >= curIndex) in1++;
			int in2 = second(gen);
			if(in2 >= in1)
			{
				in2++;
				if(in2 >= curIndex) in2++;
			}
			else if(in2 >= curIndex)
			{
				in2++;
				if(in2 >= in1) in2++;
			}
			// printf("in1 %d in2 %d\n", in1,in2);
			makeDonor({donor,nets[in1],nets[in2]},scaleFactor,true); //shallow
		}
		return donor;
	}
	// else if(mutType == DE_QUIN_AND_SUGANTHAN)
	// {

	// }
	else
	{
		printf("Unknown mutation type: %d\n", mutType);
		return nullptr;
	}
}

Net* Net::makeDonor(const vector<Net*>& helpers, double scaleFactor, bool shallow)
{
	if(helpers.size() != 3)
	{
		if(helpers.size() < 3)
		{
			printf("Helpers should be sized 3. It is less than that. Exiting.\n");
			exit(1);
		}
		else
			printf("Helpers should be sized 3. It is more than that. The extra nets are ignored.\n");
	}
	Net* donor;
	if(shallow)
		donor = helpers[0];
	else
		donor = new Net(*(helpers[0])); // copy construct target to donor
	if(donor == nullptr)
	{
		printf("nullptr! donor\n");
		getchar();
	}
	if(helpers[1] == helpers[2]) //the subtraction would be 0
		return donor;
	// printf("donor copied\n");
	//u = target + scaleFactor * (x2 - x3)
	for(int i = 1; i < donor->__layers.size(); i++) //start at 1 b/c 0 is input
	{
		if(donor->__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)donor->__layers[i];
			ConvLayer *helperConvs[2];
			int helperDepths[2];

			int filsize = conv->filterSize;
			int prevDepth = donor->__neuronDims[i-1][2]; // = conv->numWeights/(conv->filsize^2 * numFilters)

			int index1 = mapConvLayer(donor, i, helpers[1]);
			int index2 = mapConvLayer(donor, i, helpers[2]);

			if(index1 != -1)
			{
				helperConvs[0] = (ConvLayer*)(helpers[1]->__layers[index1]);
				helperDepths[0] = helpers[1]->__neuronDims[index1-1][2];
			}
			else
			{
				printf("NULL helper 0\n");
				helperConvs[0] = NULL;
			}

			if(index2 != -1)
			{
				helperConvs[1] = (ConvLayer*)(helpers[2]->__layers[index2]);
				helperDepths[1] = helpers[2]->__neuronDims[index2-1][2];
			}
			else
			{
				printf("NULL helper 1\n");
				helperConvs[1] = NULL;
			}
			//go through all filters
			for(int f = 0; f < conv->numBiases; f++)
			{
				//go through all of the weights in a 3 dim filter
				for(int a = 0; a < filsize; a++)
					for(int b = 0; b < filsize; b++)
						for(int c = 0; c < prevDepth; c++)
						{
							int mypos[] = {a,b,c};
							double otherNums[2];
							for(int o = 0; o < 2; o++)
							{
								if(helperConvs[o] == NULL)
									otherNums[o] = 0;
								else
								{
									int pos[3];
									mapPosIndexes(conv, mypos, helperConvs[o], pos);
									otherNums[o] = helperConvs[o]->weights[POSITION(f % helperConvs[o]->numBiases,pos[0],pos[1],pos[2],helperConvs[o]->filterSize,helperDepths[o])];
								}
							}
							conv->weights[POSITION(f,a,b,c,conv->filterSize, prevDepth)]
								+= scaleFactor * (otherNums[0] - otherNums[1]);
							// printf("scaleFactor %lf [0] %lf [1] %lf\n", scaleFactor, otherNums[0],otherNums[1]);
							// printf("target: %lf donor: %lf\n", conv->weights[POSITION(f,a,b,c,conv->filterSize, prevDepth)],((ConvLayer*)(helpers[0]->__layers[i]))->weights[POSITION(f,a,b,c,conv->filterSize, prevDepth)]);
							
						}
			}
		}
	}
	return donor;
}

inline int Net::POSITION(int filter, int x, int y, int z, int filsize, int prevDepth)
{
	return filter * filsize * filsize * prevDepth + x * filsize * prevDepth + y * prevDepth + z;
}

void Net::mapPosIndexes(ConvLayer* origConv, int* origpos, ConvLayer* destConv, int* destpos)
{
	// int origdepth = origConv->numWeights / (origConv->numBiases * origConv->filterSize * origConv->filterSize);
	int destdepth = destConv->numWeights / (destConv->numBiases * destConv->filterSize * destConv->filterSize);
	destpos[2] = origpos[2] % destdepth;
	if(destConv->filterSize == 1) // don't have a lot of options here
	{
		destpos[0] = 0; 
		destpos[1] = 0; 
	}
	else if(destConv->filterSize == origConv->filterSize) // best case. always case in fc layers, thankfully
	{
		destpos[0] = origpos[0];
		destpos[1] = origpos[1];
	}
	// else if(origConv->filterSize == 1) //map to middle point.
	// {
	// 	destpos[0] = destConv->filterSize / 2;
	// 	destpos[1] = destpos[0];
	// }
	else if(destConv->filterSize > origConv->filterSize) // dest bigger than me. should work if I am 1
	{
		// if the dest is bigger than me, I map to the middle of it
		int onesideDiff = destConv->filterSize - origConv->filterSize;
		destpos[0] = onesideDiff + origpos[0];
		destpos[1] = onesideDiff + origpos[1];
	}
	else // I am bigger than dest
	{
		//if I bigger than dest, grab random index from dest in same geospatial region
		int mycornersize = origConv->filterSize / 3; //5->1, 7->2, 9->3
		int geoRegion[2];
		if(origpos[0] < mycornersize)
			geoRegion[0] = 0;
		else if(origpos[0] >= origConv->filterSize - mycornersize)
			geoRegion[0] = 2;
		else 
			geoRegion[0] = 1;
		if(origpos[1] < mycornersize)
			geoRegion[1] = 0;
		else if(origpos[1] >= origConv->filterSize - mycornersize)
			geoRegion[1] = 2;
		else 
			geoRegion[1] = 1;

		int theircornersize = destConv->filterSize / 3;
		int geoIndexes[2][2];
		for(int g = 0; g < 2; g++)
		{
			if(geoRegion[g] == 0) //top or left
			{
				geoIndexes[g][0] = 0;
				geoIndexes[g][1] = theircornersize - 1;
			}
			else if(geoRegion[g] == 1) //middle
			{
				geoIndexes[g][0] = theircornersize;
				geoIndexes[g][1] = destConv->filterSize - theircornersize - 1;
			}
			else //bottom or right
			{
				geoIndexes[g][0] = destConv->filterSize - theircornersize;
				geoIndexes[g][1] = destConv->filterSize - 1; 
			}
		}
		uniform_int_distribution<int> xdis(geoIndexes[0][0], geoIndexes[0][1]);
		uniform_int_distribution<int> ydis(geoIndexes[1][0], geoIndexes[1][1]);
		default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		destpos[0] = xdis(gen);
		destpos[1] = ydis(gen);
	}
	// printf("mapPosIndexes: orig: %d %d %d dest: %d %d %d\n", origpos[0], origpos[1], origpos[2], destpos[0], destpos[1], destpos[2]);
}

Net* Net::crossover(Net* parent, Net* donor, int method, double prob)
{
	Net* trial = new Net(*parent);
	if(trial == nullptr)
	{
		printf("nullptr! trial\n");
		getchar();
	}
	uniform_real_distribution<double> dis(0.0,1.0);
	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	int donorPos[3];
	if(method == DE_BINOMIAL_CROSSOVER)
	{
		// printf("binomial\n");
		for(int i = 0; i < trial->__layers.size(); i++)
		{
			if(trial->__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* theirconv;
				int theirprevDepth;
				int theirconvIndex = mapConvLayer(trial,i,donor);
				if(theirconvIndex != -1)
				{
					theirconv = (ConvLayer*)donor->__layers[theirconvIndex];
					theirprevDepth = donor->__neuronDims[theirconvIndex-1][2];
				}

				ConvLayer* myconv = (ConvLayer*)trial->__layers[i];				
				int filsize = myconv->filterSize;
				int prevdepth = trial->__neuronDims[i-1][2];
				int numfils = myconv->numBiases;

							
				for(int f = 0; f < numfils; f++)
				{
					for(int a = 0; a < filsize; a++)
						for(int b = 0; b < filsize; b++)
							for(int c = 0; c < prevdepth; c++)
							{
								// int mypos[] = {a,b,c};
								// mapPosIndexes(myconv, mypos, theirconv, donorPos);
								// printf("My weight: %lf Their weight: %lf\n", myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)],theirconv->weights[POSITION(f % theirconv->numBiases,donorPos[0],donorPos[1],donorPos[2],theirconv->filterSize,theirprevDepth)]);
								if(dis(gen) < prob)
								{
									// printf("U < prob: myConv %d theirConv %d. \n", i, theirconvIndex);
									if(theirconvIndex != -1)
									{
										int mypos[] = {a,b,c};
										mapPosIndexes(myconv, mypos, theirconv, donorPos);
										// printf("My weight: %lf Their weight: %lf\n", myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)],theirconv->weights[POSITION(f % theirconv->numBiases,donorPos[0],donorPos[1],donorPos[2],theirconv->filterSize,theirprevDepth)]);
										myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)]
											 = theirconv->weights[POSITION(f /*% theirconv->numBiases*/,donorPos[0],donorPos[1],donorPos[2],theirconv->filterSize,theirprevDepth)];
										
									}
									else
									{
										myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)] = 0;
									}
								}
							}
				}
			}
		}

	}
	else if(method == DE_EXPONENTIAL_CROSSOVER)
	{
		default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		//need a random start point
		int numConvLayers = 0;
		for(int i = 1; i < trial->__layers.size(); i++)
			if(trial->__layers[i]->layerType == CONV_LAYER)
				numConvLayers++;

		uniform_int_distribution<int> convStart_dis(1,numConvLayers);
		int convStart = convStart_dis(gen);
		int istart = -1;

		int curConvLayer = 0;
		for(int i = 1; i < trial->__layers.size(); i++)
		{
			if(trial->__layers[i]->layerType == CONV_LAYER)
				curConvLayer++;
			if(curConvLayer == convStart)
			{
				istart = i;
				break;
			}
		}
		if(istart == -1)
		{
			printf("Break in DE_EXPONENTIAL_CROSSOVER. exiting.\n");
			exit(-1);
		}

		bool changedOne = false;

		int i = istart;
		do
		{
			if(trial->__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* myconv = (ConvLayer*)trial->__layers[i];				
				int filsize = myconv->filterSize;
				int prevdepth = trial->__neuronDims[i-1][2];
				int numFilters = myconv->numBiases;

				ConvLayer* theirconv;
				int theirprevDepth;
				int theirconvIndex = mapConvLayer(trial,i,donor);
				if(theirconvIndex != -1)
				{
					theirconv = (ConvLayer*)donor->__layers[theirconvIndex];
					theirprevDepth = donor->__neuronDims[theirconvIndex-1][2];
				}

				int fstart = 0, astart = 0, bstart = 0, cstart = 0;

				if(i == istart)
				{
					uniform_int_distribution<int> f_dis(0,numFilters-1);
					uniform_int_distribution<int> ab_dis(0,filsize-1);
					uniform_int_distribution<int> c_dis(0,prevdepth-1);

					fstart = f_dis(gen); 
					astart = ab_dis(gen); 
					bstart = ab_dis(gen);
					cstart = c_dis(gen);
				}

				for(int f = fstart; f < numFilters; f++)
				{
					for(int a = astart; a < filsize; a++)
					{
						for(int b = bstart; b < filsize; b++)
						{
							for(int c = cstart; c < prevdepth; c++)
							{
								if(dis(gen) < prob || !changedOne)
								{
									changedOne = true;
									if(theirconvIndex != -1)
									{
										int mypos[] = {a,b,c};
										mapPosIndexes(myconv, mypos, theirconv, donorPos);
										myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)]
											 = theirconv->weights[POSITION(f % theirconv->numBiases,donorPos[0],donorPos[1],donorPos[2],theirconv->filterSize,theirprevDepth)];
									}
									else
									{
										myconv->weights[POSITION(f,a,b,c,filsize,prevdepth)] = 0;
									}
								}
								else
									break;
							}
						}
					}
				}
			}
			i = (i + 1) % trial->__layers.size();


		}while(i != istart);

	}

	return trial;
}

void Net::setupLayerNeeds(vector<cl_mem>& layerNeeds)
{
	layerNeeds.clear();
	layerNeeds.resize(__layers.size());
	cl_int error = CL_SUCCESS;
	for(int i = 1; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == CONV_LAYER)
		{
			layerNeeds[i] = clCreateBuffer(__context, CL_MEM_READ_WRITE, 
				sizeof(double) * ((ConvLayer*)__layers[i])->paddedNeuronSize, nullptr, &error);
		}
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
		{
			layerNeeds[i] = clCreateBuffer(__context, CL_MEM_READ_WRITE,
				sizeof(double) * __neuronSizes[i], nullptr, &error);
		}
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			layerNeeds[i] = clCreateBuffer(__context, CL_MEM_READ_WRITE,
				sizeof(double) * __neuronSizes[i], nullptr, &error);
		}
		CheckError(error);
	}
}

void Net::destroyVectorCLMems(vector<cl_mem>& vect)
{
	for(size_t i = 0; i < vect.size(); i++)
		clReleaseMemObject(vect[i]);
	vect.clear();
	vect.resize(0);
}

void Net::getTrainingData(vector<vector<double>* >& trainingData, vector<double>& trueVals)
{
	if(__trainingType == TRAIN_AS_IS)
	{
		if(trainingData.size() != 0) // this means we've run this function at least once
		{
			shuffleTrainingData(trainingData, trueVals);
			return;
		}
		//if it's our first time through
		//add all data in order into the trainingData and trueVals
		for(int t = 0; t < __trainingData.size(); t++) // for all possible classes
		{
			int oldSize = trainingData.size();
			trainingData.resize(oldSize + __trainingData[t].size());
			trueVals.resize(oldSize + __trainingData[t].size());
			for(int i = 0; i < __trainingData[t].size(); i++) // for all training images in the class
			{
				trainingData[oldSize + i] = __trainingData[t][i];
				trueVals[oldSize + i] = getTrueValIndex(__trueNames[t]);
			}
		}
		//printf("Sizes! data %lu, true %lu\n",trainingData.size(), trueVals.size());
		//shuffle 10 times to start
		shuffleTrainingData(trainingData, trueVals, 10);
	}
	else if(__trainingType == TRAIN_EQUAL_PROP)
	{
		if(trainingData.size() == 0) //first time through. find the smallest class size
		{
			__smallestClassSize = __trainingData[0].size();
			for(int t = 1; t < __trainingData.size(); t++)
				if(__trainingData[t].size() < __smallestClassSize)
					__smallestClassSize = __trainingData[t].size();
			trainingData.clear();
			trainingData.resize(__smallestClassSize * __trainingData.size());
			trueVals.clear();
			trueVals.resize(__smallestClassSize * __trainingData.size());
		}
		int index = 0;
		//for each class, shuffle it (global __trainingData) and bring in the smallestClassSize to trainingData.
		for(int t = 0; t < __trainingData.size(); t++) // class
		{
			shuffleData(__trainingData[t],2);	//shuffling brings randomness without duplicates. It should be shuffleData 
			                                //not shuffleTrainingData because it doesn't have a vector of trueVals
			for(int i = 0; i < __smallestClassSize; i++) // image. take the first __smallestClassSize amount of images
			{
				trainingData[index] = __trainingData[t][i];
				trueVals[index] = getTrueValIndex(__trueNames[t]);
				index++;
			}
		}
		//shuffle trainingData to mix up the classes
		shuffleTrainingData(trainingData,trueVals,2);
	}
	else if(__trainingType == TRAIN_RATIO)
	{
		if(trainingData.size() == 0) //first time through. find the smallest class size
		{
			// printf("getTraining data: train ratio setup\n");
			__smallestClassSize = -1;
			__smallestClassIndex = -1;
			for(int t = 0; t < __trainingData.size(); t++)
				if(__trainingData[t].size() < __smallestClassSize && __trainRatioAmounts[t] != 0)
				{
					__smallestClassSize = __trainingData[t].size();
					__smallestClassIndex = t;
				}

			if(__smallestClassIndex == (unsigned int)-1)
			{
				printf("It seems that all train ratio amounts are 0? Exiting.\n");
				exit(-1);
			}

			int smallestClassRatio = __trainRatioAmounts[__smallestClassIndex];

			//make the smallestClassSize even divisible by the trainRatioAmount
			__smallestClassSize -= __smallestClassSize % smallestClassRatio; // if evenly divisible already, then minus 0

			bool good;
			int totalSize;
			__trainActualAmounts.resize(__trainRatioAmounts.size());
			do // if all goes well, this will only run once. If we ask for a ratio we can't do with the amount of data we have
			{  // then we try again with a smaller __smallestClassSize
				good = true;
				totalSize = 0;
				for(int t = 0; t < __trainingData.size(); t++ ) // for each class
				{
					//check if enough data for ratio based on __smallestClassSize
					int classAmount = (int)(__smallestClassSize * __trainRatioAmounts[t]/(double)smallestClassRatio);
					if(classAmount > __trainingData[t].size())
					{
						//this means we need more data than we have. Adjust accordingly
						__smallestClassSize = (int)(__trainingData[t].size() * (double)smallestClassRatio/__trainRatioAmounts[t]);

						//make the new smallestClassSize even divisible by the trainRatioAmount
						__smallestClassSize -= __smallestClassSize % smallestClassRatio; // if evenly divisible already, then minus 0
						
						//need to loop again
						good = false;
						break;
					}

					//add size 
					totalSize += classAmount;
					__trainActualAmounts[t] = classAmount;
				}

			} while(!good);
		
			trainingData.clear();
			trueVals.clear();
			trainingData.resize(totalSize);
			trueVals.resize(totalSize);

			// printf("totalSize = %d\n", totalSize);
			// for(int c = 0; c < __trainActualAmounts.size(); c++)
			// 	printf("act amount %d = %d\n", c,__trainActualAmounts[c]);
		} //end if(trainingData.size() == 0) //first time through. find the smallest class size


		//shuffle all the data
		for(int t = 0; t < __trainingData.size(); t++)
			shuffleData(__trainingData[t],2);

		//put the right amount of data in to be trained on
		int i = 0;
		for(int c = 0; c < __trainingData.size(); c++) // c is for class
		{
			for(int j = 0; j < __trainActualAmounts[c]; j++)
			{
				trainingData[i] = __trainingData[c][j];
				trueVals[i] = getTrueValIndex(__trueNames[c]);
				i++;
			}
		}

		shuffleTrainingData(trainingData,trueVals,2);
	}
}

void Net::shuffleData(vector<vector<double>* >& data, int times)
{
	if(times < 1)
		return;
	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> distr(0,data.size()-1);
	vector<double>* temp;
	for(int t=0; t < times; t++)
	{
		for(int i=0; i< data.size(); i++)
		{
			int swapIndex = distr(gen);
			temp = data[i];
			data[i] = data[swapIndex];
			data[swapIndex] = temp;
		}
	}
}

void Net::shuffleTrainingData(vector<vector<double>* >& trainingData, vector<double>& trueVals, int times)
{
	//if debugging, don't shuffle
	//return;
	// printf("start shuffle\n");
	if(times < 1)
		return;
	if(trainingData.size() != trueVals.size())
	{
		printf("In shuffle. Inconsistent sizes %lu vs %lu\n",trainingData.size(),trueVals.size());
	}
	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> distr(0,trainingData.size()-1);
	vector<double>* temp;
	int tempTrue;
	for(int t=0; t< times; t++)
	{
		for(int i=0; i< trainingData.size(); i++)
		{
			int swapIndex = distr(gen);

			temp  	 = trainingData[i];
			tempTrue = trueVals[i];

			trainingData[i]  = trainingData[swapIndex];
			trueVals[i] 	 = trueVals[swapIndex];

			trainingData[swapIndex] = temp;
			trueVals[swapIndex] 	= tempTrue;
		}
	}
	// printf("end shuffle\n");
}

bool Net::setTrainingType(int type, const vector<string>& params)
{
	if(type < 0 || 2 < type) // good up to TRAIN_RATIO
		return false;
	__trainingType = type;
	if(type == TRAIN_RATIO) // need params[0] "name1:name2:..." params[1] "ratio1:ratio2:..."
	{
		if(params.size() != 2)
		{
			printf("To use net.setTrainingType(TRAIN_RATIO) you need a second paramater 'params' \n");
			printf("   that is a vector of strings of size 2 with params[0] =\"className1:className2:...\"\n");
			printf("   and params[1] \"ratio1:ratio2:...\"\n");
			return false;
		}
		printf("splititng\n");
		vector<string> names = convnet::split(params[0],':');
		vector<string> string_ratios = convnet::split(params[1],':');

		if(names.size() != string_ratios.size())
		{
			printf("The 2 elements of setTrainingType 2nd paramater must be of contain the same number of \n");
			printf("   colon (:) delimited sub-elements. i.e. [0] -> name1:name2, [1] -> ratio1:ratio2\n");
			printf("   Not [0] -> name1:name2, [1] -> ratio1:ratio2:ratio3\n");
			return false;
		}

		printf("start for\n");
		for(int i = 0; i < names.size(); i++)
		{
			printf("for - %d\n", i);
			int index = getTrueValIndex(names[i], true); // this also takes care of resizing __trainRatioAmounts if needed
			printf("class name %s index %d ratio %s\n", names[i].c_str(),index,string_ratios[i].c_str());
			__trainRatioAmounts[index] = stoi(string_ratios[i]);
		}
	}
	printf("end setTrainingType\n");
	return true;
}

void Net::pullCLWeights()
{
 	int curConvLayer = 0;
	for(int i = 1; i < __layers.size(); i++)
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[i];
			CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
				conv->weights, 0, nullptr, nullptr));
			CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numBiases,
				conv->biases, 0, nullptr, nullptr));	
			curConvLayer++;	
		}
}

void Net::pullCLWeights(Net* net, const vector<cl_mem>& clWeights, const cl_command_queue& queue)
{
 	int curConvLayer = 0;
	for(int i = 1; i < net->__layers.size(); i++)
		if(__layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)net->__layers[i];
			CheckError(clEnqueueReadBuffer(queue, clWeights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numWeights,
				conv->weights, 0, nullptr, nullptr));
			CheckError(clEnqueueReadBuffer(queue, clBiases[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->numBiases,
				conv->biases, 0, nullptr, nullptr));	
			curConvLayer++;	
		}
}

void Net::pushCLWeights(vector<Layer*>& layers, const vector<cl_mem>& clWeights, const vector<cl_mem>& clBiases, const cl_command_queue& queue, cl_bool block)
{
	// printf("push\n");
	int curConvLayer = 0;
	for(int i=1; i < layers.size(); i++)
	{
		if(layers[i]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)layers[i];

			// printf("Layer %d: \n", i);
			// for(int j = 0 ; j < 10; j++)
			// 	printf("%lf ", conv->weights[j]);
			// printf("\n");

			CheckError(clEnqueueWriteBuffer(queue, clWeights[curConvLayer], block, 0, sizeof(double) * conv->numWeights,
				conv->weights, 0, nullptr, nullptr));
			CheckError(clEnqueueWriteBuffer(queue, clBiases[curConvLayer], block, 0, sizeof(double) * conv->numBiases,
				conv->biases, 0, nullptr, nullptr));
			curConvLayer++;
		}
	}
	// printf("end push\n");
}

void Net::setMomentum(bool useMomentum)
{
 	__useMomentum = useMomentum;
}

void Net::initVelocities(vector<cl_mem>& velocities)
{
 	cl_int error;
 	double *zeroVels = new double[__maxWeightSize];
	for(int i=0; i< __maxWeightSize; i++)
		zeroVels[i] = 0.0;
	for(int l=1; l < __layers.size(); l++)
	{
		if(__layers[l]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[l];
			velocities.push_back(clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR, sizeof(double) * conv->numWeights, zeroVels, &error));
			CheckError(error);
		}
	}
	delete zeroVels;
}

/*****************************************
 * Functions dealing with data
 *****************************************/

//running
void Net::addData(const vector<imVector>& data)
{
	printf("Adding %lu data of size %lu x %lu x %lu\n", data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size());
	int oldSize = __data.size();
	__data.resize(__data.size() + data.size());
	for(int d = 0, o = oldSize; d < data.size(); d++, o++)
	{
		__data[o].resize(__neuronSizes[0]);
		int dat = 0;
		for(int i=0; i < data[d].size(); i++)
			for(int j=0; j < data[d][i].size(); j++)
				for(int k=0; k < data[d][i][j].size(); k++)
					__data[o][dat++] = data[d][i][j][k];
	}
	printf("new size is %lu\n",__data.size());
	__confidences.resize(__confidences.size() + __data.size());
	__dataPreprocessed = false;
}

void Net::addData(const vector<Mat>& data, bool rgb)
{
	//__data is class member. data is parameter
	// cout << "Add data mat" << endl;
	int oldSize = __data.size();
	// printf("Old %d data.size %lu\n", oldSize, data.size());
	// printf("neuron[0] %d\n", __neuronSizes[0]);
	__data.resize(oldSize + data.size());
	int curIndex;
	for(int d = 0; d < data.size(); d++)
	{
		// printf("curIndex = %d\n",curIndex);
		curIndex = oldSize + d;
		__data[curIndex].resize(__neuronSizes[0]);
		int dat = 0; 
		for(int i = 0; i < data[d].rows; i++)
			for(int j = 0; j < data[d].cols; j++)
			{
				const Vec3b& curPixel = data[d].at<Vec3b>(i,j);
				//printf("%d,%d,%d", curPixel[0],curPixel[1],curPixel[2]);
				if(rgb)
				{
					__data[curIndex][dat++] = curPixel[2];
					__data[curIndex][dat++] = curPixel[1];
					__data[curIndex][dat++] = curPixel[0];
				}
				else
				{
					__data[curIndex][dat++] = curPixel[0];
					__data[curIndex][dat++] = curPixel[1];
					__data[curIndex][dat++] = curPixel[2];
				}
			}
		//printf("\n");
	}
	// printf("__data[0][0] = %lf\n",__data[0][0]);
	__confidences.resize(__confidences.size() + __data.size());
	__dataPreprocessed = false;
}

void Net::clearData()
{
	__data.resize(0);
	__confidences.resize(0);
}

void Net::setData(const vector<imVector>& data)
{
	clearData();
	addData(data);
}

void Net::setData(const vector<Mat>& data, bool rgb)
{
	// cout << "Set data mat" << endl;
	clearData();
	addData(data,rgb);
}

// void Net::setClassNames(vector<string> names, vector<int> trueVals)
// {
// 	printf("Setting Class Names.\n");
// 	__classes.resize(names.size());
// 	for(int i = 0; i < __classes.size(); i++)
// 	{
// 		__classes[i].name = names[i];
// 		__classes[i].trueVal = trueVals[i];

// 		printf("Class %d, %s\n", __classes[i].trueVal, __classes[i].name.c_str());
// 	}
// }

void Net::getClassNames(vector<string>& infos) const
{
	infos = __trueNames;
}

//training
bool Net::addTrainingData(const vector<imVector>& trainingData, const vector<string>& trueVals)
{
	if(trainingData.size() != trueVals.size())
		return false;

	#ifdef _DEBUG
	printf("Adding %lu training data of size %lu x %lu x %lu\n", trainingData.size(), trainingData[0].size(), trainingData[0][0].size(), trainingData[0][0][0].size());
	#endif

	// printf("sizes %lu & %lu\n", trainingData.size(),trueVals.size());
	int inputSize = __neuronSizes[0];

	for(int t = 0; t < trainingData.size(); t++)
	{
		// printf("data item %d\n",t);
		//if the trueVal does not yet have an index, this will resize the private class vectors and give it one.
		int trueIndex = getTrueValIndex(trueVals[t]);
		// printf("name => true index - %s => %d\n", trueVals[t].c_str(),trueIndex);

		__trainingData[trueIndex].push_back(new vector<double>(inputSize));
		// printf("input size is %d\n",inputSize);
		int dat = 0;
		for(int i=0; i < trainingData[t].size(); i++)
			for(int j=0; j < trainingData[t][i].size(); j++)
				for(int k=0; k < trainingData[t][i][j].size(); k++)
				{
					__trainingData[trueIndex].back()->at(dat++) = trainingData[t][i][j][k];
				}
	}

	#ifdef _DEBUG
	printf("done adding training data\n");
	#endif

	// __numClasses = __trueNames.size();
	return true;
}

bool Net::addTrainingData(const vector<Mat>& trainingData, const vector<string>& trueVals, bool rgb)
{
	printf("add train mat: ");
	if(trainingData.size() != trueVals.size())
		return false;
	printf("%lu images\n", trainingData.size());
	int inputSize = __neuronSizes[0];
	for(int t = 0; t < trainingData.size(); t++)
	{
		int trueIndex = getTrueValIndex(trueVals[t]);

		__trainingData[trueIndex].push_back(new vector<double>(inputSize));
		int dat = 0;
		for(int i = 0; i < trainingData[t].rows; i++)
			for(int j=0; j < trainingData[t].cols; j++)
			{
				const Vec3b& curPixel = trainingData[t].at<Vec3b>(i,j);
				if(rgb)
				{
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[2];
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[1];
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[0];
				}
				else //bgr
				{
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[0];
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[1];
					(__trainingData[trueIndex].back())->at(dat++) = curPixel[2];
				}
			}
	}
	return true;
}

void Net::clearTrainingData()
{
	for(int i = 0; i < __trainingData.size(); i++)
		for(int j = 0; j < __trainingData[i].size(); j++)
			delete __trainingData[i][j];
	__trainingData.resize(0);
	__trueNames.resize(0);
}

bool Net::setTrainingData(const vector<imVector>& trainingData, const vector<string>& trueVals)
{
	if(trainingData.size() != trueVals.size())
		return false;
	clearTrainingData();
	return addTrainingData(trainingData,trueVals);
}

bool Net::setTrainingData(const vector<Mat>& trainingData, const vector<string>& trueVals, bool rgb)
{
	if(trainingData.size() != trueVals.size())
		return false;
	clearTrainingData();
	return addTrainingData(trainingData, trueVals,rgb);
}

void Net::setTrueNameIndex(const string& name, int index)
{
	if(index >= __trueNames.size())
	{
		__trueNames.resize(index+1);
		__trainingData.resize(__trueNames.size());
		__trainRatioAmounts.resize(__trueNames.size(),0);

	}

	__trueNames[index] = name;
}

int Net::getIndexFromName(const string &name) const
{
	for(int i=0; i < __trueNames.size(); i++)
	{
		if(__trueNames[i] == name)
			return i;
	}
	return -1;
}

int Net::getTrueValIndex(const string& trueName, bool allowAppend)
{
	for(int i=0; i < __trueNames.size(); i++)
		if(__trueNames[i] == trueName)
			return i;

	if(!allowAppend)
	{
		printf("Trying to use a class name that doesn't exist.\n");
		exit(-1);
	}
	
	__trueNames.push_back(trueName);
	int newSize = __trueNames.size();
	__trainingData.resize(newSize);
	__trainRatioAmounts.resize(newSize,0);
	return newSize-1;
}

bool Net::addTestData(const vector<imVector>& testData, const vector<string>& trueNames)
{
	if(testData.size() != trueNames.size())
		return false;
	int oldSize = __testData.size();
	__testData.resize(oldSize + testData.size());
	__testTrueIndexes.resize(oldSize + testData.size());
	int curIndex;
	for(int t=0; t< testData.size(); t++)
	{
		curIndex = oldSize + t;
		__testData[curIndex].resize(__neuronSizes[0]);
		int dat = 0;
		__testTrueIndexes[curIndex] = getTrueValIndex(trueNames[t],false);
		for(int i=0; i < testData[t].size(); i++)
			for(int j=0; j < testData[t][i].size(); j++)
				for(int k=0; k < testData[t][i][j].size(); k++)
					__testData[curIndex][dat++] = testData[t][i][j][k];
	}
	return true;
}

bool Net::addTestData(const vector<Mat>& testData, const vector<string>& trueNames, bool rgb)
{
	printf("add test mat\n");
	if(testData.size() != trueNames.size())
		return false;
	int oldSize = __testData.size();
	__testData.resize(oldSize + testData.size());
	__testTrueIndexes.resize(oldSize + testData.size());
	int curIndex;
	for(int t = 0; t < testData.size(); t++)
	{
		curIndex = oldSize + t;
		__testData[curIndex].resize(__neuronSizes[0]);
		int dat = 0;
		__testTrueIndexes[curIndex] = getTrueValIndex(trueNames[t],false);
		for(int i = 0; i < testData[t].rows; i++)
			for(int j = 0; j < testData[t].cols; j++)
			{
				const Vec3b& curPixel = testData[t].at<Vec3b>(i,j);
				if(rgb)
				{
					__testData[curIndex][dat++] = curPixel[2];
					__testData[curIndex][dat++] = curPixel[1];
					__testData[curIndex][dat++] = curPixel[0];
				}
				else //bgr
				{
					__testData[curIndex][dat++] = curPixel[0];
					__testData[curIndex][dat++] = curPixel[1];
					__testData[curIndex][dat++] = curPixel[2];
				}
			}
	}
	return true;
}

void Net::clearTestData()
{
	__testData.resize(0);
	__testTrueIndexes.resize(0);
}

bool Net::setTestData(const vector<imVector>& testData, const vector<string>& trueNames)
{
	if(testData.size() != trueNames.size())
		return false;
	clearTestData();
	return addTestData(testData, trueNames);
}

bool Net::setTestData(const vector<Mat>& testData, const vector<string>& trueNames, bool rgb)
{
	if(testData.size() != trueNames.size())
		return false;
	clearTestData();
	return addTestData(testData, trueNames, rgb);
}

int Net::getNumClasses() const
{
	//return __numClasses;
	return __neuronSizes.back();
}

void Net::preprocessIndividually()
{
	// __preprocessIndividual = true;
	__preprocessType = __PREPROCESS_INDIVIDUAL;
}

void Net::preprocessCollectively()
{
	// __preprocessIndividual = false;
	__preprocessType = __PREPROCESS_COLLECTIVE;
}

void Net::preprocessDataIndividual() // thread this 
{
	//preprocess using (val - mean)/stdDeviation for all elements
	for(int i = 0; i < __data.size(); i++)
	{
		//get mean
		double mean = 0;
		for(int pix = 0; pix < __data[i].size(); pix++)
			mean += __data[i][pix];
		mean /= __data[i].size();

		//get stddev
		double stddev = 0; 
		double temp;
		for(int pix = 0; pix < __data[i].size(); pix++)
		{
			temp = __data[i][pix] - mean;
			stddev += temp * temp;
		}
		stddev = sqrt(stddev / __data[i].size());

		//adjust the values
		for(int pix=0; pix < __data[i].size(); pix++)
			__data[i][pix] = (__data[i][pix] - mean)/stddev;
	}

	__dataPreprocessed = true;
}

void Net::preprocessDataCollective()
{
	// printf("Preprocessing data collectively. Mean %lf Stddev %lf\n", __mean, __stddev);
	// getchar();
	for(int i = 0; i < __data.size(); i++)
	{
		for(int pix = 0; pix < __data[i].size(); pix++)
		{
			__data[i][pix] = (__data[i][pix] - __mean)/__stddev;
			// printf("preprocessed %lf\n", __data[i][pix]);
		}
	}
}

void Net::preprocessTestDataIndividual()
{
    printf("Preprocessing Test Data Individually\n");
	for(int i = 0; i < __testData.size(); i++)
	{
		//get mean
		double mean = 0;
		for(int pix = 0; pix < __testData[i].size(); pix++)
			mean += __testData[i][pix];
		mean /= __testData[i].size();

		//get stddev
		double stddev = 0; 
		double temp;
		for(int pix = 0; pix < __testData[i].size(); pix++)
		{
			temp = __testData[i][pix] - mean;
			stddev += temp * temp;
		}
		stddev = sqrt(stddev / __testData[i].size());

		//adjust the values
		for(int pix=0; pix < __testData[i].size(); pix++)
			__testData[i][pix] = (__testData[i][pix] - mean)/stddev;


	}
	__testDataPreprocessed = true;
}

void Net::preprocessTrainingDataIndividual()
{
    printf("Preprocessing Training Data Individually\n");
    unsigned long count = 0;
    for(int i = 0; i < __trainingData.size(); i++)
    {
        for(int im = 0; im < __trainingData[i].size(); im++)
        {
            //get mean
            double mean = 0;
            for(int pix = 0; pix < __trainingData[i][im]->size(); pix++)
            {
                mean += __trainingData[i][im]->at(pix);
            }
            mean /= __trainingData[i][im]->size();
            
            //get stddev
            double stddev = 0;
            double temp;
            for(int pix = 0; pix < __trainingData[i][im]->size(); pix++)
            {
                temp = __trainingData[i][im]->at(pix) - mean;
                stddev += temp * temp;
            }
            stddev = sqrt(stddev / __trainingData[i][im]->size());
            
            //adjust the values
            for(int pix=0; pix < __trainingData[i][im]->size(); pix++)
            {
                __trainingData[i][im]->at(pix) = (__trainingData[i][im]->at(pix) - mean)/stddev;
                //if(im==1)cout << __trainingData[i][im]->at(pix) << ", ";
            }
            count++;
	  //       if(im % 1000 == 0)
			// {
			// 	printf("Mean: %.3lf  StdDev: %.3lf\n", mean, stddev);
			// }
        }
    }
    __trainingDataPreprocessed = true;
}

void Net::preprocessTestDataCollective()
{
	if(__testData.size() == 0)
		return;

	printf("Preprocessing test data from collective mean and standard deviation\n");
	for(int i=0; i < __testData.size(); i++)
	{
		for(int pix = 0; pix < __testData[i].size(); pix++)
		{
			__testData[i][pix] = (__testData[i][pix] - __mean)/__stddev;
		}
	}

	__testDataPreprocessed = true;
}

void Net::preprocessTrainingDataCollective()
{
	printf("Preprocessing training data collectively\n");
	//getting mean and stddev on num pixels, storing and adjusting
	//based on num images to keep numbers smaller
	double mean = 0;
	double stddev = 0;
	unsigned long numPixels = 0;
	unsigned long numImages = 0;
	double temp;

	//calc mean
	for(int i = 0; i < __trainingData.size(); i++) // class
	{
		for(int im = 0; im < __trainingData[i].size(); im++) // image
		{
			for(int pix = 0; pix < __trainingData[i][im]->size(); pix++)
				mean += __trainingData[i][im]->at(pix);
			numPixels += __trainingData[i][im]->size();
		}
		numImages++;
	}
	mean /= numPixels;
    
    //calc std deviation
	for(int i = 0; i < __trainingData.size(); i++) // class
	{
		for(int im = 0; im < __trainingData[i].size(); im++) // image
		{
			for(int pix = 0; pix < __trainingData[i][im]->size(); pix++)
			{
				temp = __trainingData[i][im]->at(pix) - mean;
				stddev += temp * temp;
			}
		}
	}
	stddev = sqrt(stddev / numPixels);

	//if already trained network is adding more data, adjust mean and stddev
	if(__mean != 0 && numImages != 0) // if new training data
	{
		long totalSize = __trainingSize + numImages; 
		__mean = (__mean * __trainingSize)/totalSize + (mean * numImages)/totalSize;

		__stddev = __stddev * stddev/__trainingSize + stddev * stddev/numImages;
		__stddev = sqrt(stddev);
	}
	else if(numImages != 0)
	{
		__mean = mean;
		__stddev = stddev;
	}
	
	//adjust the values
	for(int i=0; i < __trainingData.size(); i++) // class
		for(int im = 0; im  < __trainingData[i].size(); im++) // image
			for(int pix=0; pix < __trainingData[i][im]->size(); pix++)
				__trainingData[i][im]->at(pix) = (__trainingData[i][im]->at(pix) - __mean)/__stddev;

	// printf("Mean: %.3lf  StdDev: %.3lf\n", __mean, __stddev);

	__trainingDataPreprocessed = true;
}

string Net::getNameForIndex(int index) const
{
	if(0 <= index && index < __trueNames.size())
		return __trueNames[index];
	return "";
}

void Net::printTrainingDistribution() const
{
    double numImages = 0;
    //get total num of images
    for(int i = 0; i < __trainingData.size(); i++)
    {
        numImages += __trainingData[i].size();
    }
    
    for(int i = 0; i < __trainingData.size(); i++)
    {
        printf("True val: %d %s   Amount %lu.   %.4lf%%\n", i, __trueNames[i].c_str(), __trainingData[i].size(), __trainingData[i].size()/numImages * 100.0);
    }
}

void Net::printTestDistribution() const 
{
	unordered_map<double, int> trueMap;

	for(int i = 0; i < __testTrueIndexes.size(); i++)
	{
		double val = __testTrueIndexes[i];
		unordered_map<double, int>::const_iterator got = trueMap.find(val);
		if(got == trueMap.end()) // not found
			trueMap[val] = 1;
		else // found
			trueMap[val]++;
	}

	double sum = 0;
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		sum += it->second;
	}
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		cout << "True val " << it->first << ": " << it->second << "   " << it->second/sum * 100 << "%\n";
	}
}

void Net::storeWeightsInHolder(WeightHolder& holder)
{
	holder.clearWeights();
	for(int l = 1; l < __layers.size(); l++)
	{
		if(__layers[l]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[l];

			holder.weights.push_back(new double[conv->numWeights]);
			for(int i = 0; i < conv->numWeights; i++)
				holder.weights.back()[i] = conv->weights[i];

			holder.biases.push_back(new double[conv->numBiases]);
			for(int i = 0; i < conv->numBiases; i++)
				holder.biases.back()[i] = conv->biases[i];
		}
	}
}

void Net::loadWeightsFromHolder(WeightHolder& holder)
{
	int curConvLayer = 0;
	for(int l = 1; l < __layers.size(); l++)
	{
		if(__layers[l]->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[l];

			for(int i = 0; i < conv->numWeights; i++)
				conv->weights[i] = holder.weights[curConvLayer][i];

			for(int i = 0; i < conv->numBiases; i++)
				conv->biases[i] = holder.biases[curConvLayer][i];

			curConvLayer++;
		}
	}
}

bool Net::stringToDims(string str, int* dims)
{
	//should look like 32x32x3 or 32 x 32 x 3
	str = convnet::tolower(str);
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	stringstream tokens(str);
	string item;
	int i = 0;
	while(getline(tokens, item, 'x'))
	{
		if(i == 3)
		{
			printf("You can only have 3 dimensions for a layer size.\n");
			return false;
		}
		if(!item.empty())
		{
			dims[i++] = stoi(item);
		}

	}
	return true;
}

/*****************************************
 * Load and Save
 *****************************************/

bool Net::load(const char* filename)
{
	if(__inited)
		return false;
	ifstream file;
	file.open(filename);
	string line;
	int loc;
	int lineNum = 0;

	if(!file.is_open())
	{
		cout << "File was unable to open" << endl;
		return false;
	}



	

	getline(file, line);
	if(line == "NET1.0")
	{
		setAutoActivLayer(false);
		int inputWidth, inputHeight, inputDepth, activationType;
		int netArgsFound = 0;
		getline(file, line); lineNum++;
		while(line != "END_NET")
		{
			if(line.find("activationType") != string::npos)
			{
				loc = line.find("=") + 1;
				activationType = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputWidth") != string::npos)
			{
				loc = line.find("=") + 1;
				inputWidth = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputHeight") != string::npos)
			{
				loc = line.find("=") + 1;
				inputHeight = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputDepth") != string::npos)
			{
				loc = line.find("=") + 1;
				inputDepth = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("RELU_CAP=") != string::npos)
			{
				loc = line.find("=") + 1;
				__RELU_CAP = stod(line.substr(loc));
			}
			else if(line.find("LEAKY_RELU_CONST") != string::npos)
			{
				loc = line.find("=") + 1;
				__LEAKY_RELU_CONST = stod(line.substr(loc));
			}
			else if(line.find("MEAN") != string::npos)
			{
				loc = line.find("=") + 1;
				__mean = stod(line.substr(loc));
				if(__mean != 0)
					// __preprocessIndividual = false;
					__preprocessType = __PREPROCESS_COLLECTIVE;
				else
					// __preprocessIndividual = true;
					__preprocessType = __PREPROCESS_INDIVIDUAL;
			}
			else if(line.find("STDDEV") != string::npos)
			{
				loc = line.find("=") + 1;
				__stddev = stod(line.substr(loc));
			}
			else if(line.find("TRAINING_SIZE") != string::npos)
			{
				loc = line.find("=") + 1;
				__trainingSize = stoul(line.substr(loc));
			}
			else if(line.find("CLASSES") != string::npos)
			{
				while(true)
				{
					getline(file,line); lineNum++; //this should get the trueVal
					if(line.find("END_CLASSES") != string::npos)
						break;
					int index = stoi(line);
					getline(file,line); lineNum++;
					string name = line;

					setTrueNameIndex(name,index);
				}
			}
			else
			{
				cout << "Improper file structure while getting Net args at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}

		// printf("End net\n");

		//check and make sure all 4 args were found
		if(netArgsFound < 4)
		{
			cout << "4 Net args needed. " << netArgsFound << " found. Exiting load.";
			file.close();
			return false;
		}
		//Lets init the Net.
		//cout << "Net params loaded" << endl;

		init(inputWidth,inputHeight,inputDepth);
		setActivType(activationType);


		//Now we get all the layers
		getline(file,line); lineNum++;
		while(line != "END_ALL")
		{
			// printf("%s\n", line.c_str());
			if(line == "ACTIV_LAYER")
			{
				int numActivArgs = 0, layer_activationType;
				getline(file,line); lineNum++;
				while(line != "END_ACTIV_LAYER")
				{
					if(line.find("activationType") != string::npos)
					{
						loc = line.find("=") + 1;
						layer_activationType = stoi(line.substr(loc));
						numActivArgs++;
					}
					else
					{
						cout << "Improper file structure while getting ActivLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}

				if(numActivArgs != 1)
				{
					cout << "1 ActivLayer arg needed. " << numActivArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}

				//make layer
				addActivLayer(layer_activationType);
			}
			else if(line == "MAX_POOL_LAYER")
			{
				int numPoolArgs = 0, pool_stride, pool_size;
				getline(file,line); lineNum++;
				while(line != "END_MAX_POOL_LAYER")
				{
					if(line.find("stride") != string::npos)
					{
						loc = line.find("=") + 1;
						pool_stride = stoi(line.substr(loc));
						numPoolArgs++;
					}
					else if(line.find("poolSize") != string::npos)
					{
						loc = line.find("=") + 1;
						pool_size = stoi(line.substr(loc));
						numPoolArgs++;
					}
					else
					{
						cout << "Improper file structure while getting MaxPoolLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}
				if(numPoolArgs != 2)
				{
					cout << "2 ActivLayer args needed. " << numPoolArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}

				addMaxPoolLayer(pool_size,pool_stride);
			}
			else if(line == "BATCH_NORM_LAYER")
			{
				getline(file,line); lineNum++;
				bool byFeatureMap = false;
				int gamma_size = -1, featFound = 0;
				string gamma="", beta="", e="", var="";
				while(line != "END_BATCH_NORM_LAYER")
				{
					if(line.find("byFeatureMap=1") != string::npos)
					{
						featFound++;
						byFeatureMap = true;
					}
					else if(line.find("byFeatureMap=0") != string::npos)
					{
						featFound++;
						byFeatureMap = false;
					}
					else if(line.find("gamma_size=") != string::npos)
						gamma_size = stoi(line.substr(line.find('=')+1));
					else if(line.find("gamma=") != string::npos)
						gamma = line.substr(line.find('=')+1);
					else if(line.find("beta=") != string::npos)
						beta = line.substr(line.find('=')+1);
					else if(line.find("e=") != string::npos)
						e = line.substr(line.find('=')+1);
					else if(line.find("var=") != string::npos)
						var = line.substr(line.find('=')+1);
					else
					{
						cout << "Improper file structure while getting BatchNormLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}

				if(featFound < 1)
				{
					printf("Layer ending on line %d: Missing byFeatureMap arg when getting BatchNormLayer args.\n",lineNum);
					file.close(); return false;
				}
				else if(featFound > 1)
				{
					printf("Layer ending on line %d: There should only be one byFeatureMap arg per BatchNormLayer.\n",lineNum);
					file.close(); return false;
				}
				else if(gamma_size < 1)
				{
					printf("Layer ending on line %d: BatchNormLayer - Unable to find gamma_size or gamma_size < 1.\n",lineNum);
					file.close(); return false;
				}
				else if(gamma == "")
				{
					printf("Layer ending on line %d: BatchNormLayer - Unable to find gamma.\n",lineNum);
					file.close(); return false;
				}
				else if(beta == "")
				{
					printf("Layer ending on line %d: BatchNormLayer - Unable to find beta.\n",lineNum);
					file.close(); return false;
				}
				else if(e == "")
				{
					printf("Layer ending on line %d: BatchNormLayer - Unable to find e.\n",lineNum);
					file.close(); return false;
				}
				else if(var == "")
				{
					printf("Layer ending on line %d: BatchNormLayer - Unable to find var.\n",lineNum);
					file.close(); return false;
				}

				bool success = addBatchNormLayer(byFeatureMap,gamma_size,gamma,beta,e,var);
				if(!success)
				{
					printf("Layer ending on line %d: BatchNormLayer - Failed to successfully add BatchNormLayer.\n",lineNum);
					file.close(); return false;
				}
			}
			else if(line == "CONV_LAYER")
			{
				int conv_stride, conv_pad, conv_numFilters, conv_filterSize, convArgs = 0;
				string conv_weights;
				getline(file,line); lineNum++;
				while(line != "END_CONV_LAYER")
				{
					// printf("%s\n", line.c_str());
					if(line.find("stride") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_stride = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("padding") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_pad = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("numFilters") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_numFilters = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("filterSize") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_filterSize = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("weights") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_weights = line.substr(loc);
						convArgs++;
					}
					else
					{
						cout << "Improper file structure while getting ConvLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}
				if(convArgs != 5)
				{
					cout << "5 ConvLayer args needed. " << convArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}
				addConvLayer(conv_numFilters,conv_stride,conv_filterSize,conv_pad,conv_weights);
			}
			else if(line == "SOFTMAX_LAYER")
			{ /*addSoftmaxLayer();*/}
			else
			{
				cout << "Improper file structure while getting layers at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}

		// printf("End load\n");
		file.close();
		return true;
	}
	else if(convnet::tolower(line) == "net_config")
	{
		int lineNum = 1;
		bool haveInput = false;
		char buf[500];
		//while(file >> line)
		while(getline(file, line))
		{
			// if(!file.good())
			// 	break;

			lineNum++;

			//see if it starts with pound. If so it is a comment.
			int j = 0;
			for(j = 0; j < line.length(); j++)
			{
				if(line[j] != ' ' && line[j] != '\t' && line[j] != '\n' && line[j] != '\r')
					break;
			}
			// printf("j: %d length: %lu\n", j, line.length());
			if(j >= line.length() || line[j] == '#')
				continue;

			//tokenize the line and put the results in the items vector
			vector<string> items;
			stringstream tokens(line);
			string item;
			int dims[] = {-1,-1,-1};
			while(getline(tokens, item, ' '))
			{
				if(!item.empty())
				{
					items.push_back(item);
				}
			}
			// cout << "line: " << line << endl;
			// cout << "items size: " << items.size() << endl;

			items[0] = convnet::tolower(items[0]);

			if(items[0] == "global_activ")
			{
				if(convnet::tolower(items[1]) == "leaky_relu")
					setActivType(LEAKY_RELU);
				else if(convnet::tolower(items[1]) == "relu")
					setActivType(RELU);
				else
				{
					printf("Line %d: Unimplemented activation type \"%s\".\n", lineNum, items[1].c_str());
					return false;
				}
			}
			else if(items[0] == "auto_activ")
			{
				if(convnet::tolower(items[1]) == "false")
					setAutoActivLayer(false);
				else if(convnet::tolower(items[1]) == "true")
					setAutoActivLayer(true);
				else
				{
					printf("Line %d: auto_activ must be set to either true or false.\n",lineNum);
					return false;
				}
			}
			else if(items[0] == "input")
			{
				if(haveInput)
				{
					printf("Line %d. Cannot have input twice.\n", lineNum);
					return false;
				}
				if(!stringToDims(items[1],dims))
					return false;
				init(dims[0],dims[1],dims[2]);
				haveInput = true;
			}
			else if(!haveInput)
			{
				printf("Line %d: You need to have the input layer before any other layers.\n", lineNum);
				return false;
			}
			else if(items[0] == "conv")
			{
				int filSize = -1, pad = -1, stride = -1, numFil = -1;
				int dimIndex;
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = convnet::tolower(items[i]);
					// cout << "lowered items[i]: " << items[i] << endl;
					if(items[i].find("numfil=") != string::npos)
						numFil = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("filsize=") != string::npos)
						filSize = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("pad=") != string::npos)
						pad = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("stride=") != string::npos)
						stride = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("x") != string::npos)
					{
						bool goodDims = stringToDims(items[i], dims);
						if(!goodDims) return false;
						dimIndex = i;
					}
					else
					{
						printf("Line %d: Unknown arg for Convolutional Layer \"%s\".\n",lineNum, items[i].c_str());
						return false;
					}
				}
				string errors = "";
				if(filSize <= 0)
				{
					sprintf(buf,"Line %d: filSize must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(pad < 0)
				{
					sprintf(buf,"Line %d: pad must exist and be non-negative\n",lineNum);
					errors += buf;
				}
				if(stride <= 0)
				{
					sprintf(buf,"Line %d: stride must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(numFil <= 0)
				{
					sprintf(buf,"Line %d: numFil must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(errors != "")
				{
					printf("%s\n", errors.c_str());
					return false;
				}
				bool success = addConvLayer(numFil,stride,filSize,pad);
				if(!success)
				{
					printf("Line %d: Conv Layer failed to load successfully. Make sure the stride fits previous layer size.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for conv layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else if(items[0] == "batchnorm")
			{
				if(items.size() == 1) //no other info
				{
					bool success = addBatchNormLayer();
					if(!success)
					{
						printf("Line %d: Error adding default batchNorm layer\n", lineNum);
						return false;
					}
				}
				else
				{
					if(items[1].find("byFeatureMap="))
					{
						string arg = convnet::tolower(items[1].substr(items[1].find('=')+1));
						bool success;
						if(arg == "true" || arg == "1")
							success = addBatchNormLayer(true);
						else if(arg == "false" || arg == "0")
							success = addBatchNormLayer(false);
						else
						{
							printf("Line %d: byFeatureMap arg should have value true or false or 1 or 0\n", lineNum);
							return false;
						}
						if(!success)
						{
							printf("Line %d: Error adding batchNorm layer\n", lineNum);
							return false;
						}

					}
					else
					{
						printf("Line %d: Unknown arg for batchNorm layer '%s'\n", lineNum, items[1].c_str());
						return false;
					}

				}
			}
			else if(items[0] == "activ")
			{
				string type = convnet::tolower(items[1]);
				int activType;
				if(type == "relu")
					activType = RELU;
				else if(type == "leaky_relu")
					activType = LEAKY_RELU;
				else
				{
					printf("Line %d: Unknown or unimplemented activation type \"%s\". Try \"relu\" or \"leaky_relu\".\n", lineNum,items[1].c_str());
					return false;
				}
				bool success = addActivLayer(activType);
				if(!success)
				{
					printf("Line %d: Error adding activation layer with type %s\n", lineNum, items[1].c_str());
					return false;
				}
			}
			else if(items[0] == "maxpool")
			{
				int stride = -1, pool = -1, dimIndex;
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = convnet::tolower(items[i]);
					if(items[i].find("stride=") != string::npos)
						stride = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("pool=") != string::npos)
						pool  = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find('x') != string::npos)
					{
						bool goodDims = stringToDims(items[i], dims);
						if(!goodDims) return false;
						dimIndex = i;						
					}
					else
					{
						printf("Line %d: Unknown arg for MaxPool Layer \"%s\".\n",lineNum, items[i].c_str());
						return false;
					}
				}

				string errors = "";
				if(stride <= 0)
				{
					sprintf(buf,"Line %d: stride must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(pool <= 0)
				{
					sprintf(buf,"Line %d: pool must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(errors != "")
				{
					printf("%s\n", errors.c_str());
					return false;
				}
				bool success = addMaxPoolLayer(pool, stride);
				if(!success)
				{
					printf("Line %d: MaxPool Layer failed to load correctly. Make sure stride fits previous layer size.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for maxpool layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else if(items[0] == "fc")
			{
				// printf("fully connected\n");
				int outputSize = -1, dimIndex;
				if(items.size() > 3)
				{
					printf("Line %d: Too many args for Fully Connected Layer\n", lineNum);
					return false;
				}
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = convnet::tolower(items[i]);
					if(items[i].find('x') != string::npos)
					{
						bool goodDims = stringToDims(items[i],dims);
						if(!goodDims) return false;
						dimIndex = i;
					}
					else
						outputSize = stoi(items[i]);
				}
				if(outputSize <= 0)
				{
					printf("Line %d: pool must exist and be positive\n",lineNum);
					return false;
				}
				bool success = addFullyConnectedLayer(outputSize);
				if(!success)
				{
					printf("Line %d: Error adding Fully Connected Layer.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for fc layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else
			{
				printf("Line %d: Unknown arg \"%s\"\n", lineNum, line.c_str());
				return false;
			}
		}
		file.close();
		return true;
	}
	file.close();
	cout << "Unknown file format" << endl;
	return false;
}

bool Net::save(const char* filename)
{
	//get file stuff
	ofstream file;
	file.open(filename, ofstream::trunc);

	if(!file.is_open())
		return false;

	char data[500];
	//need to save in such a way that it can load dynamically
	string out = "NET1.0\n";

	//put in Net hyperparameters only. NOT TRAINING OR REAL DATA
	sprintf(data,"%d", __defaultActivType);
	out += "activationType="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][0]);
	out += "inputWidth="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][1]);
	out += "inputHeight="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][2]); 
	out += "inputDepth="; out += data; out += '\n';

	sprintf(data,"%lf",__RELU_CAP);
	out += "RELU_CAP="; out += data; out += '\n';

	sprintf(data,"%lf",__LEAKY_RELU_CONST);
	out += "LEAKY_RELU_CONST="; out += data; out += '\n';

	sprintf(data,"%lf",__mean);
	out += "MEAN="; out += data; out += '\n';

	sprintf(data, "%lf", __stddev);
	out += "STDDEV="; out += data; out += '\n';

	sprintf(data, "%lu", __trainingSize);
	out += "TRAINING_SIZE="; out += data; out += '\n';

	out  += "CLASSES\n";
	for(int c = 0; c < __trueNames.size(); c++)
	{
		sprintf(data, "%d\n", c); //index
		out  += data;
		sprintf(data, "%s\n", __trueNames[c].c_str()); //name
		out  += data;
	}
	out += "END_CLASSES\n";

	out += "END_NET\n";

	for(int l=1; l < __layers.size(); l++)
	{
		int type = __layers[l]->layerType;
		if(type == MAX_POOL_LAYER)
		{
			MaxPoolLayer *pool = (MaxPoolLayer*)__layers[l];
			out += "MAX_POOL_LAYER\n";

			sprintf(data,"%d",pool->stride);
			out += "stride="; out += data; out += '\n';

			sprintf(data,"%d",pool->poolSize);
			out += "poolSize="; out += data; out += '\n';

			out += "END_MAX_POOL_LAYER\n";
		}
		else if(type == ACTIV_LAYER)
		{
			ActivLayer *act = (ActivLayer*)__layers[l];
			out += "ACTIV_LAYER\n";

			sprintf(data,"%d",act->activationType);
			out += "activationType="; out += data; out += '\n';

			out += "END_ACTIV_LAYER\n";

		}
		else if(type == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[l];
			out += "CONV_LAYER\n";

			sprintf(data,"%d",conv->stride);
			out += "stride="; out += data; out += '\n';

			sprintf(data,"%d",conv->padding);
			out += "padding="; out += data; out += '\n';

			sprintf(data,"%d",conv->numBiases); // same as numFilters
			out += "numFilters="; out += data; out += '\n';

			sprintf(data,"%d",conv->filterSize);
			out += "filterSize="; out += data; out += '\n';

			out += "weights=";
			for(int f=0; f < conv->numWeights; f++)
			{
				sprintf(data,"%lf,",conv->weights[f]);
				out += data;
			}
			out += "_"; // biases
			for(int b=0; b < conv->numBiases; b++)
			{
				sprintf(data,"%lf,",conv->biases[b]);
				out += data;
			}
			out += '\n';

			out += "END_CONV_LAYER\n";
		}
		else if(type == BATCH_NORM_LAYER)
		{
			BatchNormLayer* batch = (BatchNormLayer*)__layers[l];
			out += "BATCH_NORM_LAYER\n";
			if(batch->byFeatureMap)
				out += "byFeatureMap=1\n";
			else
				out += "byFeatureMap=0\n";

			sprintf(data,"%lu\n",batch->gamma.size());
			out += "gamma_size="; out += data;
			out += "gamma=";
			for(int g = 0; g < batch->gamma.size(); g++)
			{
				sprintf(data,"%lf,",batch->gamma[g]);
				out += data;
			}
			out += '\n';

			out += "beta=";
			for(int g = 0; g < batch->beta.size(); g++)
			{
				sprintf(data,"%lf,",batch->beta[g]);
				out += data;
			}
			out += '\n';
			out += "e=";
			for(int g = 0; g < batch->e.size(); g++)
			{
				sprintf(data,"%lf,",batch->e[g]);
				out += data;
			}
			out += '\n';
			out += "var=";
			for(int g = 0; g < batch->var.size(); g++)
			{
				sprintf(data,"%lf,",batch->var[g]);
				out += data;
			}
			out += '\n';


			out += "END_BATCH_NORM_LAYER\n";
		}
	}
	out += "SOFTMAX_LAYER\n";
	out += "END_ALL";
	file << out;
	file.close();

	return true;
}

/*****************************************
 * OpenCL Functions
 *****************************************/

int Net::getDevice() const
{
	return __device;
}

bool Net::setDevice(unsigned int device)
{
	if(device >= __deviceIdCount)
		return false;
	cl_uint canDouble;
	CheckError(clGetDeviceInfo(__deviceIds[device], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &canDouble, nullptr));
	if(canDouble == 0)
		return false;
	__device = device;
    __isFinalized = false;
	return true;
}

void Net::setDevice(cl_device_id device, cl_platform_id platform)
{
	bool found = false;
	for(int i = 0; i < __deviceIds.size(); i++)
		if(device == __deviceIds[i])
			__device = i;

	if(!found)
	{
		__deviceIds[0] = device;
		__device = 0;
	}

		//context
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platform),
		0,0
	};
	cl_int error;
	clReleaseContext(__context);
	__context = clCreateContext(contextProperties, 1, &device,
		nullptr, nullptr, &error);
	CheckError(error);
}

void Net::setGPU(bool useGPU)
{
	__useGPU = useGPU;
}

void Net::setConstantMem(bool useConstantMem)
{
	__constantMem = useConstantMem;
}

// void Net::CheckError (cl_int error)
// {
// 	if (error != CL_SUCCESS) {
// 		error_mtx.lock();
// 		std::cerr << "OpenCL call failed with error " << error << std::endl;
// 		error_mtx.unlock();
// 		std::exit (1);
// 	}
// }

std::string Net::LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

int Net::getMaxNeuronSize() const
{
	int max = 0;
	for(int i = 0; i < __neuronSizes.size(); i++)
		if(__neuronSizes[i] > max)
			max = __neuronSizes[i];
	return max;
}

cl_program Net::CreateProgram (std::string source, cl_context& context, int programNum)
{
	char buf[200];
	int location;
	if(programNum == TRAINING_PROGRAM)
	{
		//change defines so they match what is in the variables here
		
		//l2Lambda
		location = source.find("#define l2Lambda"); //should be #define l2Lambda 0.05
		source.erase(location + 17,4);
		sprintf(buf,"%lf",__l2Lambda);
		source.insert(location + 17,buf);

		//MOMENT_CONST
		location = source.find("#define MOMENT_CONST"); //should be #define MOMENT_CONST .9
		source.erase(location + 21,2);
		sprintf(buf,"%lf",__MOMENT_CONST);
		source.insert(location + 21,buf);

		//MAX_NORM_CAP
		location = source.find("#define MAX_NORM_CAP"); //should be #define MAX_NORM_CAP 6.0
		source.erase(location + 21,3);
		sprintf(buf,"%lf",__MAX_NORM_CAP);
		source.insert(location + 21,buf);

		int max_neuron_size = getMaxNeuronSize();
		location = source.find("#define MAX_NEURON_SIZE"); //should be #define MAX_NORM_CAP 6.0
		source.erase(location + 24,5);
		sprintf(buf,"%d",max_neuron_size);
		source.insert(location + 24,buf);
	}

	if(programNum == TRAINING_PROGRAM || programNum == RUNNING_PROGRAM)
	{
		//RELU_CAP
		location = source.find("#define RELU_CAP"); //should be #define RELU_CAP 5000.0
		source.erase(location + 17,6);
		sprintf(buf,"%lf",__RELU_CAP);
		source.insert(location + 17,buf);

		//LEAKY_RELU_CONST
		location = source.find("#define LEAKY_RELU_CONST"); //should be #define LEAKY_RELU_CONST .01
		source.erase(location + 25,3);
		sprintf(buf,"%lf",__LEAKY_RELU_CONST);
		source.insert(location + 25,buf);

	}

	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}


Net::WeightHolder::~WeightHolder()
{
	clearWeights();
}

void Net::WeightHolder::clearWeights()
{
	for(int i = 0; i < weights.size(); i++)
	{
		delete weights[i];
		delete biases[i];
	}
	weights.resize(0);
	biases.resize(0);
}


Net::ConvLayer& Net::ConvLayer::operator=(const Net::ConvLayer& other)
{
	if(this != &other)
	{
		// printf("Copying convLayer\n");
		//see if we need to clean up our weights and biases
		if(weights != nullptr)
		{
			printf("Deleting old weights\n");
			delete weights;
		}
		if(biases != nullptr)
		{
			printf("Deleting old biases\n");
			delete biases;
		}

		//copy over everything
		numWeights = other.numWeights;
		numBiases = other.numBiases;
		numNeurons = other.numNeurons;
		padding = other.padding;
		stride = other.stride;
		filterSize = other.filterSize;
		paddedNeuronWidth = other.paddedNeuronWidth;
		paddedNeuronHeight = other.paddedNeuronHeight;
		paddedNeuronSize = other.paddedNeuronSize;
		maxSizeNeeded = other.maxSizeNeeded;

		weights = new double[numWeights];
		biases = new double[numBiases];
		for(int i = 0; i < numWeights; i++)
			weights[i] = other.weights[i];
		for(int i = 0; i < numBiases; i++)
			biases[i] = other.biases[i];
	}
	return *this;
}

bool Net::ConvLayer::equals(const Net::ConvLayer& other)
{
	bool ret = true;
	ret = ret && (numWeights == other.numWeights);
	ret = ret && (numBiases == other.numBiases);
	ret = ret && (numNeurons == other.numNeurons);
	ret = ret && (padding == other.padding);
	ret = ret && (stride == other.stride);
	ret = ret && (filterSize == other.filterSize);
	ret = ret && (paddedNeuronWidth == other.paddedNeuronWidth);
	ret = ret && (paddedNeuronHeight == other.paddedNeuronHeight);
	ret = ret && (paddedNeuronSize == other.paddedNeuronSize);
	ret = ret && (maxSizeNeeded == other.maxSizeNeeded);
	if(!ret)
		return !ret;
	for(int i = 0; i < numWeights; i++)
		ret = ret && (weights[i] == other.weights[i]);
	for(int i = 0; i < numBiases; i++)
		ret = ret && (biases[i] == other.biases[i]);

	return ret;
}

int Net::check_counter(int count)
{
	if(count >= 1)
		return count;
	return 1;
}

Net::BNRunMems::BNRunMems() {}

Net::BNRunMems::BNRunMems(int numThreads, Net* parent)
{
	load(numThreads,parent);
}

void Net::BNRunMems::load(int numThreads, Net* parent)
{
	if(parent == nullptr || numThreads < 1)
	{
		printf("Error creating BNRunMems! Exiting\n");
		exit(1);
	}
	if(numThreads == this->numThreads && parent == this->parent)
		return;
	this->numThreads = numThreads;
	p.resize(numThreads);
	n.resize(numThreads);
	prevNeurons.resize(numThreads);
	neurons.resize(numThreads);
	kernels.resize(numThreads);
	queues.resize(numThreads);
	denoms.resize(numThreads);

	cl_int error;
	for(int t = 0; t < numThreads; t++)
	{
		p[t] = clCreateBuffer(parent->__context, CL_MEM_READ_WRITE, sizeof(double) * parent->__maxNeuronSize, nullptr, &error); CheckError(error);
		n[t] = clCreateBuffer(parent->__context, CL_MEM_READ_WRITE, sizeof(double) * parent->__maxNeuronSize, nullptr, &error); CheckError(error);
		prevNeurons[t] = &(p[t]);
		neurons[t] = &(n[t]);

		//1D inits
		parent->buildKernels(kernels[t],parent->__device); // if called multiple times will only build once
		queues[t] = clCreateCommandQueue(parent->__context, parent->__deviceIds[parent->__device], 0, &error); CheckError(error);
		denoms[t] = clCreateBuffer(parent->__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error); CheckError(error);
	}
}

Net::BNRunMems::~BNRunMems()
{
	destroy();
}

void Net::BNRunMems::destroy()
{
	prevNeurons.clear();
	neurons.clear(); // so no dangling pointers

	for(int i = 0; i < numThreads; i++)
	{
		clReleaseMemObject(p[i]);
		clReleaseMemObject(n[i]);
		clReleaseCommandQueue(queues[i]);
		clReleaseMemObject(denoms[i]);
	}

	//kernels releases in its own destructor
}

int Net::BNRunMems::getNumThreads() const
{
	return numThreads;
}

//assumes rgb image
void Net::convertGreyscale(vector<double>& image)
{
	assert(image.size() % 3 == 0);
	vector<double> greyImage(image.size()/3);
	for(int i = 0; i < greyImage.size(); i++)
		greyImage[i] = .21 * image[i] + .72 * image[i+1] + .07 * image[i+2];
	image = greyImage;
}

void Net::makeTrainingGreyscale()
{
	for(int i = 0; i < __trainingData.size(); i++)
		for(int j = 0; j < __trainingData[i].size(); j++)
			convertGreyscale(*__trainingData[i][j]);
}
