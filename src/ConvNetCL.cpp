
#include "ConvNetCL.h"
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <random>
#include <fstream>
#include <random>
#include <unordered_map>

// 	includes brought in from ConvNetCL.h
//
// #include <vector>	
// #include <string> 
// #ifdef __APPLE__
//  	#include "OpenCL/opencl.h"
// #else
//  	#include "CL/cl.h"
// #endif
//typedef std::vector<std::vector<std::vector<double> > > imVector; // typedef pulled from the .h file

using namespace std;

#define GETMAX(x,y) (x > y) ? x: y

// void printArray(double* array, int size)
// {
// 	for(int i=0; i< size; i++)
// 	{
// 		cout << array[i] << ", ";
// 	}
// 	cout << endl << endl;
// }


/*****************************************
 * Constructors and Destructors and inits
 *****************************************/

Net::Net(const char* filename)
{
	load(filename);
}

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	init(inputWidth, inputHeight, inputDepth);
}

Net::~Net()
{
	Layer *point;
	for(int i=0; i< __layers.size(); i++)
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

	for(int t=0; t < __trainingData.size(); t++) // class
		for(int i = 0; i < __trainingData[t].size(); i++) // image in class
			delete __trainingData[t][i];

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
		//running
		clReleaseKernel(convKernelF);
		clReleaseKernel(zeroPadKernelF);
		clReleaseKernel(maxPoolKernelF);
		clReleaseKernel(reluKernelF);
		clReleaseKernel(leakyReluKernelF);
		clReleaseKernel(softmaxKernelF);
		clReleaseProgram(CNForward);

		//training
		clReleaseKernel(convKernel);
		clReleaseKernel(zeroPadKernel);
		clReleaseKernel(maxPoolKernel);
		clReleaseKernel(reluKernel);
		clReleaseKernel(leakyReluKernel);
		clReleaseKernel(softmaxKernel);
		clReleaseKernel(convBackNeuronsKernel);
		clReleaseKernel(convBackBiasesKernel);
		clReleaseKernel(convBackWeightsKernel);
		clReleaseKernel(zeroPadBackKernel);
		clReleaseKernel(maxPoolBackKernel);
		clReleaseKernel(reluBackKernel);
		clReleaseKernel(leakyReluBackKernel);
		clReleaseKernel(softmaxBackKernel);
		clReleaseProgram(CNTraining);
	}

	clReleaseContext(__context);
}

void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	pushBackLayerSize(inputWidth,inputHeight,inputDepth);
    Layer* abstract = new Layer;
	__layers.resize(1);
    __layers.back() = abstract;
	initOpenCL();
}

void Net::initOpenCL()
{
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

bool Net::addActivLayer()
{
	return addActivLayer(__defaultActivType);
}

bool Net::addActivLayer(int activType)
{
	if(activType >= MAX_ACTIV || activType < 0)
		return false;

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

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad)
{
	return addConvLayer(numFilters, stride, filterSize, pad, string("random"));
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
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
		initRandomWeights(conv);
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

bool Net::addFullyConnectedLayer(int outputSize)
{
	return addConvLayer(outputSize, 1, __neuronDims.back()[0], 0);
}

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


void Net::pushBackLayerSize(int width, int height, int depth)
{
	__neuronSizes.push_back(width * height * depth);
	__neuronDims.resize(__neuronDims.size() + 1);
	__neuronDims.back().resize(3);
	__neuronDims.back()[0] = width;
	__neuronDims.back()[1] = height;
	__neuronDims.back()[2] = depth;
}

void Net::initRandomWeights(ConvLayer* conv)
{
    //cout << "making random weights" << endl;
	default_random_engine gen(time(0));

	//use the number of inputs to get the random start weights
	double numInputs = conv->filterSize * conv->filterSize + 1;
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

void Net::initWeights(ConvLayer* conv, const string& weights)
{
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
}

bool Net::setActivType(int activationType)
{
	if(activationType >= MAX_ACTIV || activationType < 0)
		return false;
	__defaultActivType = activationType;
	return true;
}

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
}

void Net::setAutoActivLayer(bool isAuto)
{
	__autoActivLayer = isAuto;
}

string Net::getErrorLog() const
{
	return __errorLog;
}

bool Net::finalize()
{
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
	}

	printf("Finalizing CNN using device %d\n",q);

	cl_int error;

	if(!__stuffBuilt)
	{
		//build the program
		//running
		CNForward = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), __context);
		const cl_device_id* deviceToBuild = &(__deviceIds[q]);
		CheckError(clBuildProgram(CNForward, 1, deviceToBuild, nullptr, nullptr, nullptr));
		//training
		CNTraining = CreateProgram(LoadKernel("../kernels/ConvNetTraining_kernel.cl"), __context);
		CheckError(clBuildProgram(CNTraining, 1, deviceToBuild, nullptr, nullptr, nullptr));
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
		else
		{
			cout << "Unknown layer type. Aborting finalize." << endl;
			cout << "Type: " << type << endl;
			return false;
		}
	}
	n = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	neurons = &n;
	p = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	prevNeurons = &p;

	denom = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);


	__isFinalized = true;
	return true;
}

bool Net::set_learningRate(double rate)
{
	if(rate < 0)
		return false;
	__learningRate = rate;
	return true;
}

bool Net::set_RELU_CAP(double cap)
{
	if(cap <= 0)
		return false;
	__RELU_CAP = cap;
	__isFinalized = false;
	return true;
}

bool set_LEAKY_RELU_CONST(double lconst)
{
	if(lconst < 0 || 1 < lconst)
		return false;
	__LEAKY_RELU_CONST = lconst;
	__isFinalized = false;
	return true;
}

bool set_l2Lambda(double lambda)
{
	if(lambda < 0)
		return false;
	__l2Lambda  = lambda;
	__isFinalized = false;
	return true;
}

bool set_MOMENT_CONST(double mconst)
{
	if(mconst < 0 || 1 < mconst)
		return false;
	__MOMENT_CONST = mconst;
	__isFinalized = false;
	return true;
}

bool set_MAX_NORM_CAP(double cap)
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

void Net::run(bool useGPU)
{
 	// if(useGPU != __useGPU)
 	// {
 	// 	__useGPU = useGPU;
 	// 	__isFinalized = false;
 	// }

 	if(!__isFinalized)
 		finalize();

 	if(!__dataPreprocessed)
 		preprocessData();

 	if(!__isTraining)
 		__dataPointer = &__data;
 	else
 	{
 		__confidences.resize(__dataPointer->size());
 	}

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	for(int i=0; i < __confidences.size(); i++)
 		__confidences[i].resize(__neuronSizes.back());

 	//vector<double> test(__maxNeuronSize);

	cl_mem *temp;

 	for(int r = 0; r < __dataPointer->size(); r++)
 	{
 		//put in the next image
 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
				sizeof(double) * __neuronSizes[0],
				(*__dataPointer)[r].data(), 0, nullptr, nullptr));
		clFinish(queue);

		int curConvLayer = 0;
		size_t globalWorkSize[] = {0,0,0};
		//go through the layers
		for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
		{
			//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
			if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)__layers[i];
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
					//cout << "using constant" << endl;
					clSetKernelArg(convKernelFC, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernelFC, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernelFC, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
					clSetKernelArg(convKernelFC, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
					clSetKernelArg(convKernelFC, 4, sizeof(int), &(conv->numBiases)); //numFilters
					clSetKernelArg(convKernelFC, 5, sizeof(int), &(conv->filterSize));
					clSetKernelArg(convKernelFC, 6, sizeof(int), &(conv->stride));
					clSetKernelArg(convKernelFC, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
					clSetKernelArg(convKernelFC, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

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
		// 	CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 		test.data(), 0, nullptr, nullptr));
		// 	for(int j=0; j< __neuronSizes[i]; j++)
		// 	{
		// 		cout << test[j] << ", ";
		// 	}
		// 	cout << endl << endl;
		// 	getchar();

			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;
		}

		//cout << "Softmax" << endl;
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

		CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	__confidences[r].data(), 0, nullptr, nullptr));

	// for(int c=0; c < __confidences[r].size(); c++)
	// 	cout << __confidences[r][c] << " ";
	// cout << endl;
	// getchar();


 	}
}

void Net::getCalculatedClasses(vector<int>& dest) const
{
 	dest.resize(__confidences.size());
 	for(int i=0; i < dest.size(); i++)
 		dest[i] = getMaxElementIndex(__confidences[i]);
}

void Net::getConfidences(vector<vector<double> >& confidences) const
{
 	confidences = __confidences;
}

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

void Net::train(int epochs)
{
 	if(!__isFinalized || __constantMem)
 	{
 		__constantMem = false;
 		if(!finalize())
 		{
 			printf("Net was unable to finalize. Aborting training.\n");
 			return;
 		}
 	}
    
    size_t valueSize;
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, 0, NULL, &valueSize));
	char* name = new char[valueSize];
	printf("Training with device %d: ",__device);
	CheckError(clGetDeviceInfo(__deviceIds[__device], CL_DEVICE_NAME, valueSize, name, nullptr));
	printf("%s\n",name);

   	if(__trainingType == TRAIN_AS_IS)
   		printf("Training using AS IS\n");
   	else if(__trainingType == TRAIN_EQUAL_PROP)
   		printf("Training using EQUAL PROPORTIONS\n");

 	__isTraining = true;

 	//preprocess the data, training and test
 	if(!__trainingDataPreprocessed)
 		//preprocessTrainingDataCollective();
        preprocessTrainingDataIndividual();
    if(__testData.size() != 0 && !__testDataPreprocessed)
    	preprocessTestDataIndividual();

 	//set up stuff so we can exit based on error on test data
 	vector<vector<double> > confidences;
 	double prevError = DBL_MAX;
 	double curError;
 	if(__testData.size() != 0)
 	{
 		__dataPointer = &__testData;
 	}
 	//set up all the layerNeeds for training.
 	vector<cl_mem> layerNeeds(0);
 	setupLayerNeeds(layerNeeds);

 	//set up velocity stuff
	vector<cl_mem> velocities(0); // still needs to exist even if not using momentum so it compiles.
	if(__useMomentum)
		initVelocities(velocities);

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));

 	vector<double> test(__maxNeuronSize);

 	if(epochs == -1)
 	{
 		if(__testData.size() == 0)
 			epochs = 1;
 		else
 			epochs = 9999; // if you need more than this set it yourself
 	}
    
	cl_mem *temp;
	string ep = to_string(epochs);
	int epSize = ep.size();

	vector<double> soft(__neuronSizes.back());

	vector<vector<double>* > trainingData(0);
	vector<double> trueVals(0);

	////////////////////////////
	// start of training
	// start of epochs
	////////////////////////////
	setbuf(stdout,NULL);
	for(int e = 1; e <= epochs; e++)
	{   
		//adjust learning rate
		if(e % 5 == 0 && e != 0)
		{
			__learningRate *= .5;
			printf("\tChanged learning rate from %.3e to %.3e before starting epoch %d\n",__learningRate*2,__learningRate,e);
		}
		cout << "Epoch: ";
	 	cout << setw(epSize) << e;
	 	cout << ". ";
	 	// printf("Epoch: %d",e);

		getTrainingData(trainingData, trueVals); // this gets the training data for this epoch
		
		int numCorrect = 0;

	 	for(int r = 0; r < trainingData.size(); r++)
	 	{
	 		//put in the next image
	 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
					sizeof(double) * __neuronSizes[0],
					trainingData[r]->data(), 0, nullptr, nullptr));
			clFinish(queue);

			int curConvLayer = 0;
			size_t globalWorkSize[] = {(size_t)__neuronSizes[0],0,0}; // initialized for copyArrayKernel

			////////////////////////////
			// start forwardprop
			////////////////////////////
			for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
			{
				//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
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

			//cout << "Softmax" << endl;
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

			//get the output and see if it was right
			CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
			 	soft.data(), 0, nullptr, nullptr));
            clFinish(queue);
			if(getMaxElementIndex(soft) == trueVals[r])
				numCorrect++;
            
        //print soft
        // cout << "Softmax forward" << endl;
        // for(int s = 0; s < soft.size(); s++)
        //     cout << "| " << soft[s] << " ";
        // cout << "|\n";
        // cout << "max element: " << getMaxElementIndex(soft) << ". TrueVal: " << trueVals[r] << endl;


			////////////////////////////
			// start backprop
			////////////////////////////
        	int curTrueVal = trueVals[r];
			clSetKernelArg(softmaxBackKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxBackKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxBackKernel, 2, sizeof(int), &curTrueVal);
			//globalWorkSize[0] = (size_t)__neuronSizes.back(); // this is still true
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

			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;

			curConvLayer--;
			//prevNeurons has become prevdNeurons
			//neurons has become dneurons
			for(int i = __layers.size() -1; i > 0; i--)
			{
				if(__layers[i]->layerType == ACTIV_LAYER)
				{
					int type = ((ActivLayer*)__layers[i])->activationType;
					globalWorkSize[0] = (size_t)__neuronSizes[i-1];
					if(type == RELU)
					{
						//cout << "running reluBackKernel " << endl;
						clSetKernelArg(reluBackKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(reluBackKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, reluBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
					else if(type == LEAKY_RELU)
					{
						//cout << "running leakyReluBackKernel " << endl;
						clSetKernelArg(leakyReluBackKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(leakyReluBackKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo
						CheckError(clEnqueueNDRangeKernel(queue, leakyReluBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
				}
				else if(__layers[i]->layerType == MAX_POOL_LAYER)
				{
					//cout << "running maxPoolBackKernel " << endl;
					int numIndexes = __neuronSizes[i];
					clSetKernelArg(maxPoolBackKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(maxPoolBackKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i]));
					clSetKernelArg(maxPoolBackKernel, 3, sizeof(int), &numIndexes);
					globalWorkSize[0] = (size_t)__neuronSizes[i-1];
					CheckError(clEnqueueNDRangeKernel(queue, maxPoolBackKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else if(__layers[i]->layerType == CONV_LAYER)
				{
					//backprop neurons					
					//cout << "running convBackNeuronsKernel" << endl;
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
					//cout << "running convBackBiasesKernel " << curConvLayer << ": " << sizeOfNeurons << "/" << hyper[4] << endl;
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
					//cout << "running convBackWeightsKernel" << endl;
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
						//cout << "Running zeroPadBackKernel" << endl;
						globalWorkSize[0] = (size_t)conv->paddedNeuronSize;
						CheckError(clEnqueueNDRangeKernel(queue,zeroPadBackKernel, 1,
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

				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}// end for loop for backprop
	 	}// end for loop for training data (meaning the epoch has finished)
	 	
	 	//beginning of this line is at the top of the epoch loop
	 	cout << "Accuracy on training data: " << numCorrect << " out of " << trueVals.size() << ". " << numCorrect/(double)trueVals.size()*100 << "%" << endl;
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
		 		if(getMaxElementIndex(__confidences[c]) == __testTrueVals[c])
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
	pullCLWeights();
	//clean up anything we may have messed with in order to use run.
	__confidences.resize(__data.size());
	__isTraining = false;
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
				trueVals[oldSize + i] = __trueVals[t];
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
				trueVals[index] = __trueVals[t];
				index++;
			}
		}
		//shuffle trainingData to mix up the classes
		shuffleTrainingData(trainingData,trueVals,2);
	}
}

void Net::shuffleData(vector<vector<double>* >& data, int times)
{
	if(times < 1)
		return;
	default_random_engine gen(time(0));
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
	if(times < 1)
		return;
	default_random_engine gen(time(0));
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
}

bool Net::setTrainingType(int type)
{
	if(type < 0 || 1 < type)
		return false;
	__trainingType = type;
	return true;
}

void Net::setHorizontalReflections(bool useHReflect)
{
	__useHorizontalReflections = useHReflect;
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
	for(int d = 0; d < data.size(); d++)
	{
		__data.resize(__data.size() + 1);
		__data.back().resize(__neuronSizes[0]);
		int dat = 0;
		for(int i=0; i < data[d].size(); i++)
			for(int j=0; j < data[d][i].size(); j++)
				for(int k=0; k < data[d][i][j].size(); k++)
					__data.back()[dat++] = data[d][i][j][k];
	}
	__confidences.resize(__data.size());
	__dataPreprocessed = false;
}

void Net::clearData()
{
	__data.resize(0);
}

void Net::setData(const vector<imVector>& data)
{
	clearData();
	addData(data);
}

//training
bool Net::addTrainingData(const vector<imVector>& trainingData, const vector<double>& trueVals)
{
	if(trainingData.size() != trueVals.size())
		return false;

	int width = __neuronDims[0][0];
	int height = __neuronDims[0][1];
	int depth = __neuronDims[0][2];
	double temp; // only used if __useHorizontalReflections == true

	int inputSize = __neuronSizes[0];

	for(int t = 0; t < trainingData.size(); t++)
	{
		//if the trueVal does not yet have an index, this will resize the private class vectors and give it one.
		int trueIndex = getTrueValIndex(trueVals[t]);

		//__trainingData[trueIndex].resize(__trainingData[trueIndex].size() + 1);
		//__trainingData[trueIndex].back() = new vector<double>(__neuronSizes[0]);
		__trainingData[trueIndex].push_back(new vector<double>(inputSize));
		int dat = 0;
		for(int i=0; i < trainingData[t].size(); i++)
			for(int j=0; j < trainingData[t][i].size(); j++)
				for(int k=0; k < trainingData[t][i][j].size(); k++)
				{
					(__trainingData[trueIndex].back())->at(dat++) = trainingData[t][i][j][k];
					if(__useHorizontalReflections)
					{
						cout << "horizontally reflecting" << endl;
						__trainingData[trueIndex].push_back(new vector<double>(__neuronSizes[0]));
						//copy in original from prev vector
						for(int c = 0; c < __neuronSizes[0]; c++)
							(*(__trainingData[trueIndex].back()))[c] = (*(__trainingData[trueIndex][__trainingData.size() - 2]))[c];
						//horizontally reflect new vector
						for(int ii = 0; ii < height; ii++)
						{
							int rowstart = ii * width * depth;
							for(int jj = 0; jj < width/2; jj++)
							{
								int colstart = rowstart + jj * depth;
								int revcolstart = rowstart + (width - jj) * depth;
								for(int kk = 0; kk < depth; kk++)
								{
									temp = (*(__trainingData[trueIndex].back()))[colstart + k];
									(*(__trainingData[trueIndex].back()))[colstart + k] = (*(__trainingData[trueIndex].back()))[revcolstart + k];
									(*(__trainingData[trueIndex].back()))[revcolstart + k] = temp;
								}
							}
						}
					}
				}
	}

	__numClasses = __trueVals.size();
	return true;
}

void Net::clearTrainingData()
{
	__trainingData.resize(0);
	__trueVals.resize(0);
}

bool Net::setTrainingData(const vector<imVector>& trainingData, const vector<double>& trueVals)
{
	if(trainingData.size() != trueVals.size())
		return false;
	clearTrainingData();
	return addTrainingData(trainingData,trueVals);
}

int Net::getTrueValIndex(double trueVal)
{
	for(int i=0; i < __trueVals.size(); i++)
		if(__trueVals[i] == trueVal)
			return i;

	
	__trueVals.push_back(trueVal);
	int newSize = __trueVals.size();
	__trainingData.resize(newSize);
	return newSize-1;
}

bool Net::addTestData(const vector<imVector>& testData, const vector<double>& trueVals)
{
	if(testData.size() != trueVals.size())
		return false;
	int oldSize = __testData.size();
	__testData.resize(oldSize + testData.size());
	__testTrueVals.resize(oldSize + testData.size());
	int curIndex;
	for(int t=0; t< testData.size(); t++)
	{
		curIndex = oldSize + t;
		__testData[curIndex].resize(__neuronSizes[0]);
		int dat = 0;
		__testTrueVals[curIndex] = trueVals[t];
		for(int i=0; i < testData[t].size(); i++)
			for(int j=0; j < testData[t][i].size(); j++)
				for(int k=0; k < testData[t][i][j].size(); k++)
					__testData[curIndex][dat++] = testData[t][i][j][k];
	}
	return true;
}

void Net::clearTestData()
{
	__testData.resize(0);
	__testTrueVals.resize(0);
}

bool Net::setTestData(const vector<imVector>& testData, const vector<double>& trueVals)
{
	if(testData.size() != trueVals.size())
		return false;
	clearTestData();
	return addTestData(testData, trueVals);
}

int Net::getNumClasses() const
{
	return __numClasses;
}

void Net::preprocessData() // thread this 
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

void Net::preprocessTestDataIndividual()
{
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
    cout << "Preprocessing Individually" << endl;
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
         //    if(im==1)
         //    {
	        //     cout << endl;
	        //     exit(0);
	        // }
        }
    }
    __trainingDataPreprocessed = true;
}

void Net::preprocessTrainingDataCollective()
{
	//getting mean and stddev on num pixels, storing and adjusting
	//based on num images to keep numbers smaller
	double mean = 0;
	double stddev = 0;
	unsigned long numPixels = 0;
	unsigned long numImages = 0;
	double temp;
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
	if(false) // if new training data
	{
		long totalSize = __trainingSize + numImages; 
		__mean = (__mean * __trainingSize)/totalSize + (mean * numImages)/totalSize;

		__stddev = __stddev * stddev/__trainingSize + stddev * stddev/numImages;
		__stddev = sqrt(stddev);
	}
	else
	{
		__mean = mean;
		__stddev = stddev;
	}
	
	//adjust the values
	for(int i=0; i < __trainingData.size(); i++)
		for(int im = 0; im  < __trainingData[i].size(); im++)
			for(int pix=0; pix < __trainingData[i][im]->size(); pix++)
				__trainingData[i][im]->at(pix) = (__trainingData[i][im]->at(pix) - __mean)/__stddev;

	__trainingDataPreprocessed = true;
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
        printf("True val: %.0lf. Amount %lu.   %.4lf%%\n", __trueVals[i], __trainingData[i].size(), __trainingData[i].size()/numImages * 100.0);
    }
}

void Net::printTestDistribution() const 
{
	unordered_map<double, int> trueMap;

	for(int i = 0; i < __testTrueVals.size(); i++)
	{
		double val = __testTrueVals[i];
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

/*****************************************
 * Load and Save
 *****************************************/

bool Net::load(const char* filename)
{
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

	setAutoActivLayer(false);

	getline(file, line);
	if(line == "NET1.0")
	{
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
			else
			{
				cout << "Improper file structure while getting Net args at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}

		//check and make sure all 4 args were found
		if(netArgsFound != 4)
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
			else if(line == "CONV_LAYER")
			{
				int conv_stride, conv_pad, conv_numFilters, conv_filterSize, convArgs = 0;
				string conv_weights;
				getline(file,line); lineNum++;
				while(line != "END_CONV_LAYER")
				{
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
	file.open(filename);

	if(!file.is_open())
		return false;

	char data[50];
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

void Net::setGPU(bool useGPU)
{
	__useGPU = useGPU;
}

void Net::setConstantMem(bool useConstantMem)
{
	__constantMem = useConstantMem;
}

void Net::CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string Net::LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

cl_program Net::CreateProgram (const std::string& source, cl_context& context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}
