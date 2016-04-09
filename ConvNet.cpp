/*************************************************************************************************
 *
 *  ConvNet.cpp
 *  
 *
 *  Created by Connor Bowley on 3/15/16.
 *
 *	Classes
 *		Net
 *		Layer - Abstract
 *			ConvLayer
 *			MaxPoolLayer
 *			ActivLayer - does the activation function. Defaults to RELU.
 *			InputLayer
 *			FullyConnectedLayer??? - not currently implemented.
 *
 *	Each Layer needs forwardprop and backprop
 *		Can we build in local dneurons in forwardprop? No?
 *		Multiply(and add) by forward derivatives in backprop
 * 		
 *		each layer finishes the derivatives for the layer before it
 *		layers never need to know what layer comes after??? even in backprop??? I think so.
 *
 *
 *	If you want to add a new layer, in Net you need to update:
 *		add a new layer type in the .h file. static const int NEW_LAYER = 1+whatever the last one is
 *		search for that layer in the save and load functions. Prob make a getHyperparameters function
 *		make a new add function. bool addNewLayer(Layer&, hyperparameters needed)
 *
 *
 *	Todo: For CPU maxPool forward and backprop, have forwardprop create a vector the size of m_neurons
 *		  and keep track of which indexes feed into each neuron. in backprop set all to 0 and += on the
 *		  saved indexes whatever m_dneurons is for that element
 *
 *	Todo: Fix batch gradient descent?
 *
 *	Todo: init random weights on Gaussian distr. w = np.random.randn(n) * sqrt(2.0/n)
 *		  where n is number of inputs to neuron
 *
 *	Todo: loss based on size of weights
 *
 *	Todo: fix miniBatchTrain?
 *
 *	Todo: implement a gradient check using numerical gradients.
 *
 * 	Todo: make a special forwardprop and backprop for ConvLayer for when padding = 0.
 *
 * 	Todo: make a semi-shallow copy of net (and layers?) that has it's own neurons but all point to the 
 *		  same weights?
 *	
 *	Todo: Threads! and GPUs!
 *
 *	Todo: Check build status for each device and run accordingly
 *
 *	Todo: zeroPad kernel
 *
 *************************************************************************************************/

#include "ConvNet.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <limits>
#include <string>
#include <fstream>
#include <random>
#include <cassert>
#include <thread>


#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

//#include "ConvNet_kernel.cl.h"

#define GETMAX(x,y) (x > y) ? x: y

using namespace std;

/***********************************
 * Class Implementations
 ***********************************/

/**********************
 * Net
 **********************/

float Net::stepSize = 1e-3;

int Net::n_activationType = 0;

const float Net::GRADCHECK_H = .01;

bool Net::gradCheck = false;


void Net::debug()
{
	
}

Net::Net(const char* filename)
{
	load(filename);
}

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	init(inputWidth,inputHeight,inputDepth);
}

void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	n_hasConvLayer = false;
	n_hasMaxPoolLayer =false;
	n_hasRELULayer = false;
	n_hasLeakyRELULayer = false;
	n_hasSoftmax  = false;
	//Normal Net init stuff
	resize3DVector(n_blankVector,inputWidth,inputHeight,inputDepth);
	n_blankInput.setImage(&n_blankVector, &n_blankVector);
	n_layers.push_back(&n_blankInput);

	//Init OpenCL stuff
}

Net::~Net()
{
	//do I need to run delete on the vectors in the layers????????
	Layer *point;
	
	for(int i=0; i< n_layers.size(); i++)
	{
		point = n_layers.back();
		n_layers.pop_back();
		delete point;
	}
	/*
	for(int i=0; i< n_trainingData.size(); i++)
	{
		delete n_trainingData[i];
	}
	*/
	
}

void Net::forwardprop()
{
	for(int i=1; i< n_layers.size(); i++)
	{
		n_layers[i]->forwardprop(*n_layers[i-1]);
	}
}

void Net::backprop()
{
	for(int i= n_layers.size()-1; i > 0; i--)
	{
		n_layers[i]->backprop(*n_layers[i-1]);
	}
}

void Net::runTrainingData()
{
	int numCorrect = 0;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	//set the next training image as the InputLayer for the net
	for(int t=0; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		//run forward pass
		forwardprop();

		//set trueVal
		soft->setTrueVal(n_trainingDataTrueVals[t]);

		int predictedClass = soft->getPredictedClass();
		if(predictedClass == n_trainingDataTrueVals[t])
			numCorrect++;
	}
	cout << "Run on Training data: " <<  "Accuracy: " << (float)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

}

void Net::splitTrain(int epochs)
{

	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}
	//set 1 in the d_neurons in the last layer
			//or do we need to set it to the error? -> I don't think so.
	string ep = to_string(epochs);

	int startValidationIndex = n_trainingData.size() * 0.9;

	gradCheck = false;
	//vector<vector<vector<float> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< startValidationIndex; t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal
			soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

			//run backward pass beside the weight no
			backprop();

			//

		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (float)numCorrect/(startValidationIndex)*100 << "%, " << numCorrect << " out of " << startValidationIndex << endl;

		shuffleTrainingData();
	}

	cout << "Running Validation set" << endl;
	numCorrect = 0;
	for(int t=startValidationIndex; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		//run forward pass
		forwardprop();

		//set trueVal
		soft->setTrueVal(n_trainingDataTrueVals[t]);

		int predictedClass = soft->getPredictedClass();
		if(predictedClass == n_trainingDataTrueVals[t])
			numCorrect++;
	}
	cout << "Validation run on Training data: " <<  "Accuracy: " << (float)numCorrect/(n_trainingData.size()-startValidationIndex)*100 << "%, " << numCorrect << " out of " << (n_trainingData.size()-startValidationIndex) << endl;

}

void Net::train(int epochs)
{
	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}

	//set 1 in the d_neurons in the last layer
			//or do we need to set it to the error? -> I don't think so.
	string ep = to_string(epochs);

	gradCheck = false;
	//vector<vector<vector<float> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal
			soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

			//run backward pass beside the weight no
			backprop();

			//

		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (float)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

	}

}

void Net::miniBatchTrain(int epochs, int batchSize)
{
	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}

	string ep = to_string(epochs);
	int origBatchSize = batchSize;

	gradCheck = false;
	//vector<vector<vector<float> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	vector<float> errors(2);
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	int numFullBatches = n_trainingData.size()/batchSize;
	int remain = n_trainingData.size() % batchSize;
	int curBatch = 0;
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		batchSize = origBatchSize;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t+=batchSize)
		{
			if(curBatch >= numFullBatches)
				batchSize = remain;
			for(int b=0; b<batchSize; b++)
			{
				n_layers[0] = n_trainingData[t+b];

				//run forward pass
				forwardprop();

				//set trueVal
				soft->setTrueVal(n_trainingDataTrueVals[t]);

				//get error
				vector<float> curError = soft->getError();
				for(int i=0; i< curError.size(); i++)
				{
					errors[i] += curError[i];
				}

				int predictedClass = soft->getPredictedClass();
				//cout << "Pred: "<<predictedClass << " True: "<< n_trainingDataTrueVals[t] << "\n";
				if(predictedClass == n_trainingDataTrueVals[t])
					numCorrect++;
			}

			for(int i=0; i< errors.size(); i++)
			{
				errors[i] /= n_trainingData.size();
			}
			soft->setError(errors);
			backprop();
			for(int i=0; i< errors.size(); i++)
				errors[i] = 0;

		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (float)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

		

	}
}

void Net::gradientCheck()
{
	gradCheck = true;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int t=0; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		forwardprop();

		soft->setTrueVal(n_trainingDataTrueVals[t]);

		backprop();
		cout << "Image "<<t << endl;
		soft->gradientCheck(*n_layers[n_layers.size()-2]);
	}
	

	//for layers other than softmax do
	//vector<...> prevNeurons = prevLayer.getNeurons
	//without putting an & before prevNeurons. then you can change it.
}

void Net::run(bool useGPU) // run only goes forward and will be on the GPU if possible.
{
	cl_int error = CL_SUCCESS;

	vector<int> calculatedClasses(n_trainingData.size());
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);

	cl_uint gpudeviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_GPU, 0, nullptr, &gpudeviceIdCount);

	//if we have gpus, use them. else run forward like normal.
	if(gpudeviceIdCount > 0 && useGPU)
	{
		cout << "GPU(s) found. Running program on GPU" << endl;
		vector<cl_device_id> gpudeviceIds(gpudeviceIdCount);
		clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, gpudeviceIdCount, gpudeviceIds.data(), nullptr);

		//use all the gpus?
		//make the context
		const cl_context_properties contextProperties[] = 
		{
			CL_CONTEXT_PLATFORM,
			reinterpret_cast<cl_context_properties>(platformIds[0]),
			0,0
		};
		cl_context context = clCreateContext(contextProperties, gpudeviceIdCount, gpudeviceIds.data(), 
		nullptr, nullptr, &error);
		CheckError(error);

		//build the program
		cl_program CNForward = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), context);
		const char* options = "-cl-single-precision-constant";
		//cout << "Build Program " << clBuildProgram(CNForward, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr) << endl;
		CheckError(clBuildProgram(CNForward, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr));
		cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel;

		//Figure out which kernels we need.
		if(n_hasRELULayer)
		{
			reluKernel = clCreateKernel(CNForward, "relu", &error);
			CheckError(error);
		}
		if(n_hasLeakyRELULayer)
		{
			leakyReluKernel = clCreateKernel(CNForward, "leakyRelu", &error);
			CheckError(error);
		}
		if(n_hasConvLayer)
		{
			convKernel = clCreateKernel(CNForward, "convolve", &error);
			CheckError(error);
			zeroPadKernel = clCreateKernel(CNForward, "zeroPad", &error);
			CheckError(error);
		}
		if(n_hasMaxPoolLayer)
		{
			maxPoolKernel = clCreateKernel(CNForward, "maxPool", &error);
			CheckError(error);
		}
		if(n_hasSoftmax)
		{
			softmaxKernel = clCreateKernel(CNForward, "softmax", &error);
			CheckError(error);
		}

		//cout << "Kernels created" << endl;
			
		//make the command queue
		int q = 1;
		cl_command_queue queue = clCreateCommandQueue(context, gpudeviceIds[q], 0, &error);
			CheckError(error);

		//make a vector of pointers to kernels that is parallel to n_layers. Use this to make memory
		//on gpu for the weights and biases. get maxSize of every set of neurons and make two of
		//those on gpu
		int numConvLayers = 0;

		int maxNeuronSize = n_layers[0]->getNumNeurons();

		vector<cl_kernel*> layers(n_layers.size());
		layers[0] = nullptr; //because input layer
		vector<cl_mem> weights;
		vector<int> weightSizes;
		vector<cl_mem> biases;
		vector<int> biasSizes;
		for(int i=1; i< n_layers.size(); i++)
		{
			int type = n_layers[i]->getType();
			if(type == Net::CONV_LAYER)
			{
				//cout << "conv " << numConvLayers << "i " << i << endl;
				//cout << "Conv" << endl;
				layers[i] = &convKernel;
				numConvLayers++;
				ConvLayer *conv = (ConvLayer*) n_layers[i];
				int numWeights = conv->getNumWeights();
				//cout << "get some weights" << endl;
				float* w = conv->getWeights();
				//cout << "weights got" << endl;
				weights.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(float) * numWeights, w, &error));
				weightSizes.push_back(numWeights);
				CheckError(error);

				//get biases too
				int numBiases = conv->getNumBiases();
				float *b = conv->getBiases();
				biases.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(float) * numBiases, b, &error));
				biasSizes.push_back(numBiases);
				CheckError(error);

				int maxSizeNeeded = conv->getMaxSizeNeeded();
				if(maxSizeNeeded > maxNeuronSize)
				{
					maxNeuronSize = maxSizeNeeded;
				}

				delete w;
				delete b;
				//cout << "End Conv" << endl;
			}
			else if (type == Net::MAX_POOL_LAYER)
				layers[i] = &maxPoolKernel;
			else if (type == Net::SOFTMAX_LAYER)
				layers[i] = &softmaxKernel;
			else if (type == Net::ACTIV_LAYER)
			{
				ActivLayer* act = (ActivLayer*)n_layers[i];
				if(act->getActivationType() == ActivLayer::RELU)
					layers[i] = &reluKernel;
				else if (act->getActivationType() == ActivLayer::LEAKY_RELU)
					layers[i] = &leakyReluKernel;
			}
			else
			{
				cout << "Unknown layer type for the GPU implemenation. Defaulting to CPU." << endl;
				gradCheck = false;
				for(int r=0; r < n_trainingData.size(); r++)
				{
					n_layers[0] = n_trainingData[r];

					//run forward pass
					forwardprop();

					//get the results and save them into n_results
				}
				return;
			}
			if(n_layers[i]->getNumNeurons() > maxNeuronSize)
			{
				maxNeuronSize = n_layers[i]->getNumNeurons();
			}
		}

		//cout << "GPU Layers initialized" << endl;
		
		/*
		cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeuronSize,
			nullptr, &error);
		CheckError(error);
		cl_mem pn = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeuronSize,
			nullptr, &error);
		CheckError(error);
		*/
		cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeuronSize,
				nullptr, &error);
		CheckError(error);
		cl_mem *neurons = &n;
		cl_mem p = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeuronSize,
				nullptr, &error);
		CheckError(error);
		cl_mem *prevNeurons = &p;

		float* testHold = new float[maxNeuronSize];

		//cout << "Max neuron size: " << maxNeuronSize << endl;
		
		int imageSize = ((InputLayer*)n_layers[0])->getImageSize();
		float* image = new float[imageSize];

		/**************************************
		 *
		 *	Start going through the images
		 *
		 **************************************/
		int softSize = n_layers.back()->getNumNeurons();
		vector<float> neur(softSize);
		//cout << "Starting run on GPU(s)" << endl;
		for(int r=0; r < n_trainingData.size(); r++)
		{
			//set the new image(s)
			int startForThisRound = r;
			if(r < n_trainingData.size())
				n_trainingData[r]->getImage(image, imageSize);
			

			//printArray(image, imageSize);

			//cout << "copy image(s) into GPU" << endl;
			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
					sizeof(float) * imageSize,
					image, 0, nullptr, nullptr));
			clFinish(queue);






			/*
			cout  << "trying convKernel" << endl;

			float tweights[] = {1,0,1,-1,0,0,-1,1,0,0,1,0,1,0,0,1,-1,-1,-1,0,0,1,1,-1,-1,1,0,0,1,-1,-1,-1,-1,-1,0,-1,1,1,-1,0,0,-1,0,1,-1,1,-1,1,1,1,-1,-1,0,1};
			float tbiases[] = {1,0};
			cl_mem tw = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 54,
				tweights, &error);
			cl_mem tb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2,
				tbiases, &error);
			CheckError(error);


			float testArray[] = {
				0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
				2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
				1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
				1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
				2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};
			cl_mem ta = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 5*5*3,
				testArray, &error);
			CheckError(error);
			int stride = 2;
			int filterSize = 3;
			int pwid = 5;
			int pdep = 3;

			cout << "setting args" << endl;


			clSetKernelArg(convKernel, 0, sizeof(cl_mem), &ta);
			clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convKernel, 2, sizeof(cl_mem), &tw);
			clSetKernelArg(convKernel, 3, sizeof(cl_mem), &tb);
			int numFilters = 2;
			clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
			clSetKernelArg(convKernel, 5, sizeof(int), &filterSize);
			clSetKernelArg(convKernel, 6, sizeof(int), &stride);
			clSetKernelArg(convKernel, 7, sizeof(int), &pwid);
			clSetKernelArg(convKernel, 8, sizeof(int), &pdep);
			cout << "args set" << endl;
			const size_t globalWorkSiz[] = {8,0,0};
			CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
				nullptr,
				globalWorkSiz,
				nullptr, 0, nullptr, nullptr));

			cout << "convKernel done" << endl;

			clEnqueueReadBuffer(queue, *neurons, CL_TRUE, 0, sizeof(float) * maxNeuronSize,
						testHold, 0, nullptr, nullptr);
					//cout << testHold[0] << endl;
			printArray(testHold, 18);
			
			return;
			*/





			//run kernels from index = 1 to index = end - 1. special stuff needs to be done on 
			//data before softmax is run
			int curConvLayer = 0;
			size_t globalWorkSize[] = {0,0,0};
			cl_mem *temp;
			
			for(int i=1; i< layers.size() - 1; i++)
			{	
				//cout << "start for" << endl;
				if(layers[i] == &convKernel)
				{	/*
					vector<int> hyper(7);
					hyper[0] = c_weights[0].size(); // filterSize
					hyper[1] = c_stride;
					hyper[2] = c_prevNeuronWidth;
					hyper[3] = c_prevNeuronDepth;
					hyper[4] = c_biases.size();  // num filters
					hyper[5] = c_padding;
					hyper[6] = c_prevNeuronHeight;
					*/

					//cout << "running convKernel " << curConvLayer << endl;
					vector<int> hyper = ((ConvLayer*)n_layers[i])->getKernelHyperParameters();

					if(hyper[5] != 0) //if padding != 0
					{
						//run the zeroPad Kernel
						clSetKernelArg(zeroPadKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(zeroPadKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(zeroPadKernel, 2, sizeof(int), &(hyper[5])); // padding
						clSetKernelArg(zeroPadKernel, 3, sizeof(int), &(hyper[2])); // prevWidth
						clSetKernelArg(zeroPadKernel, 4, sizeof(int), &(hyper[6])); // prevHeight
						clSetKernelArg(zeroPadKernel, 5, sizeof(int), &(hyper[3])); // depth (before and after zero pad)

						// run it for the size of the new array
						//							  (prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
						globalWorkSize[0] = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
						CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernel, 1,
							nullptr,
							globalWorkSize,
							nullptr, 0, nullptr, nullptr));
						clFinish(queue);

						//swap the buffers so prevNeurons holds the zero padded data
						temp = neurons;
						neurons = prevNeurons;
						prevNeurons = temp;

					}
					
					clSetKernelArg(convKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernel, 2, sizeof(cl_mem), &weights[curConvLayer]);
					clSetKernelArg(convKernel, 3, sizeof(cl_mem), &biases[curConvLayer]);
					
					int numFilters = hyper[4];
					clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
					clSetKernelArg(convKernel, 5, sizeof(int), &(hyper[0])); // filterSize
					clSetKernelArg(convKernel, 6, sizeof(int), &(hyper[1])); // stride
					clSetKernelArg(convKernel, 7, sizeof(int), &(hyper[2])); // prevWidth
					clSetKernelArg(convKernel, 8, sizeof(int), &(hyper[3])); // prevDepth

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
					curConvLayer++;
				}
				else if(layers[i] == &maxPoolKernel)
				{
					//cout << "running maxPoolKernel " << endl;
					vector<int> hyper = ((MaxPoolLayer*)n_layers[i])->getKernelHyperParameters();

					clSetKernelArg(maxPoolKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(maxPoolKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(maxPoolKernel, 2, sizeof(int), &(hyper[0]));
					clSetKernelArg(maxPoolKernel, 3, sizeof(int), &(hyper[1]));
					clSetKernelArg(maxPoolKernel, 4, sizeof(int), &(hyper[2]));
					clSetKernelArg(maxPoolKernel, 5, sizeof(int), &(hyper[3]));
					
					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &reluKernel)
				{
					//cout << "running reluKernel " << endl;
					clSetKernelArg(reluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluKernel, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, reluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &leakyReluKernel)
				{
					//cout << "running leakyReluKernel " << endl;
					clSetKernelArg(leakyReluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluKernel, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				clFinish(queue);

				/*
				if(i == 3)
				{
					clEnqueueReadBuffer(queue, *neurons, CL_TRUE, 0, sizeof(float) * maxNeuronSize,
						testHold, 0, nullptr, nullptr);
					cout << testHold[0] << endl;

					printArray(testHold, globalWorkSize[0]);
					cout << "From layer " << i << endl;
					return;
				}
				*/
				
				

				//swap prev and cur neuron pointers
				
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}

			//cout << "starting softmaxKernel" <<  endl;
			//end of forward through normal layers. now softmax is all that is left
			// the current neurons are in prevNeurons due to the pointer swap
			
			CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(float) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			//clFinish(queue[q]);
			maxSubtraction(neur);
			float denom = vectorESum(neur);
			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(float) * softSize,
				neur.data(), 0, nullptr, nullptr));

			clFinish(queue);

			//set kernel args
			//cout << "running softmaxKernel " << r <<  endl;
			clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxKernel, 2, sizeof(float), &denom);
			globalWorkSize[0] = (size_t)softSize;
			CheckError(clEnqueueNDRangeKernel(queue, softmaxKernel, 1,
				nullptr,
				globalWorkSize,
				nullptr, 0, nullptr, nullptr));


			clFinish(queue);


			CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(float) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			calculatedClasses[startForThisRound++] = getMaxElementIndex(neur);
			//cout << getMaxElementIndex(neur) << " | " << neur[getMaxElementIndex(neur)] << endl;;
		}

		delete image;

		if(calculatedClasses.size() == n_trainingDataTrueVals.size())
		{
			int numCorrect = 0;
			for(int i=0; i< calculatedClasses.size(); i++)
			{
				if(calculatedClasses[i] == n_trainingDataTrueVals[i])
					numCorrect++;
			}
			cout << "Accuracy on training data run: " << numCorrect << " out of " << n_trainingDataTrueVals.size() << ". " << numCorrect/(float)calculatedClasses.size()*100 << "%" << endl;
		}
		else
		{
			for(int i=0; i < calculatedClasses.size(); i++)
			{
				cout << calculatedClasses[i] << endl;
			}
		}

		clReleaseCommandQueue(queue);
		clReleaseMemObject(*neurons);
		clReleaseMemObject(*prevNeurons);
		for(int w = 0; w < weights.size(); w++)
			clReleaseMemObject(weights[w]);
		for(int b = 0; b <biases.size(); b++)
			clReleaseMemObject(biases[b]);

		//clReleaseKernel(convKernel);
		//clReleaseKernel(softmaxKernel);
		//clReleaseProgram(program);

		//clReleaseContext(context);


	}
	else // if no gpus
	{
		//CPU implemenation
		gradCheck = false;
		//vector<vector<vector<float> > >& lastLayerGradients = n_layers.back()->getdNeurons();
		//setAll3DVector(lastLayerGradients,1);
		int numCorrect;
		SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal only needed for backprop
			//soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

		}
		cout << "Accuracy: " << (float)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

	}
}

int Net::numLayers() 
{
	return n_layers.size()-1; // subtract one because the input layer doesn't count as a real layer.
}

bool Net::addActivLayer()
{
	try
	{
		ActivLayer *activ = new ActivLayer(*n_layers.back(),n_activationType);
		n_layers.push_back(activ);
		if(n_activationType == ActivLayer::RELU)
		{
			n_hasRELULayer = true;
		}
		else if(n_activationType == ActivLayer::LEAKY_RELU)
		{
			n_hasLeakyRELULayer = true;
		}
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addActivLayer(int activationType)
{
	try
	{
		ActivLayer *activ = new ActivLayer(*n_layers.back(),activationType);
		n_layers.push_back(activ);
		if(activationType == ActivLayer::RELU)
		{
			n_hasRELULayer = true;
		}
		else if(activationType == ActivLayer::LEAKY_RELU)
		{
			n_hasLeakyRELULayer = true;
		}
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad)
{
	try
	{
		ConvLayer *conv = new ConvLayer(*(n_layers.back()),numFilters,stride,filterSize,pad);
		n_layers.push_back(conv);
		n_hasConvLayer = true;
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
{
	try
	{
		ConvLayer *conv = new ConvLayer(*(n_layers.back()),numFilters,stride,filterSize,pad, weightsAndBiases);
		n_layers.push_back(conv);
		n_hasConvLayer = true;
		return true;
	}
	catch(...)
	{
		cout << "in catch" << endl;
		return false;
	}
}

bool Net::addMaxPoolLayer(int poolSize, int stride)
{
	try
	{
		MaxPoolLayer *pool = new MaxPoolLayer(*(n_layers.back()),poolSize, stride);
		n_layers.push_back(pool);
		n_hasMaxPoolLayer = true;
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addSoftmaxLayer()
{
	try
	{
		SoftmaxLayer *soft = new SoftmaxLayer(*(n_layers.back()));
		n_layers.push_back(soft);
		n_hasSoftmax = true;
		return true;
	}
	catch(...)
	{
		return false;
	}
}

//The true value will be the index of the correct class
void Net::addTrainingData(const vector<vector<vector<vector<float> > > >& trainingData, vector<float>& trueVals)
{
	const vector<vector<vector<vector<float> > > >& t = trainingData;	
	for(int n=0; n< t.size(); n++)
	{
		InputLayer *in = new InputLayer(t[n],&n_blankVector);
		n_trainingData.push_back(in);
		n_trainingDataTrueVals.push_back(trueVals[n]);
	}
}

void Net::addRealData(const vector<vector<vector<vector<float> > > >& realData)
{
	const vector<vector<vector<vector<float> > > >& r = realData;
	n_results.resize(r.size());	
	for(int n=0; n< r.size(); n++)
	{
		InputLayer *in = new InputLayer(r[n],&n_blankVector);
		n_trainingData.push_back(in);
	}
}

void Net::clear()
{
	n_layers.clear();
}

vector<vector<vector<float> > >* Net::getBlankVectorPointer()
{
	return &n_blankVector;
}

bool Net::setActivType(int activationType)
{
	if(0 <= activationType && activationType < ActivLayer::NUM_ACTIV_TYPES)
	{
		n_activationType = activationType;
		return true;
	}
	return false;
}

float Net::calcLoss(int indexOfTrueVal)
{
	// call softmax on the last layer of input data. The last layer should go down to a vector that is
	// 1x1xn or 1xnx1 or nx1x1
	vector<float> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions); // gets out predictions
	return -log(normedPredictions[indexOfTrueVal]);

}

int Net::getPredictedClass()
{
	vector<float> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions);
	int maxLoc = 0;
	for(int i=1; i<normedPredictions.size();i++)
	{
		if(normedPredictions[i] > normedPredictions[maxLoc])
		{
			maxLoc = i;
		}
	}
	//cout << maxLoc << endl;
	return maxLoc;
}

void Net::shuffleTrainingData(int times)
{
	if(times < 1)
		return;
	default_random_engine gen(time(0));
	uniform_int_distribution<int> distr(0,n_trainingData.size()-1);
	if(times == 1)
		cout << "Shuffling training images 1 time... ";
	else
		cout << "Shuffling training images " << times << " times... ";
	for(int t=0; t< times; t++)
	{
		for(int i=0; i< n_trainingData.size(); i++)
		{
			int swapIndex = distr(gen);
			InputLayer* temp  = n_trainingData[i];
			int tempTrue = n_trainingDataTrueVals[i];

			n_trainingData[i] 		  = n_trainingData[swapIndex];
			n_trainingDataTrueVals[i] = n_trainingDataTrueVals[swapIndex];

			n_trainingData[swapIndex] 		  = temp;
			n_trainingDataTrueVals[swapIndex] = tempTrue;
		}
	}
	cout << "Done" << endl;
}

bool Net::load(const char* filename)
{
	ifstream file;
	file.open(filename);
	string line;
	int loc;
	
	int lineNum = 0;

	if(!file.is_open())
		return false;

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
				//cout << "ActivLayer added" << endl;
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
				//cout << "MaxPoolLayer added" << endl;
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
				//cout << "At addConvLayer" << endl;
				addConvLayer(conv_numFilters,conv_stride,conv_filterSize,conv_pad,conv_weights);
				//cout << "ConvLayer added" << endl;
			}
			else if(line == "SOFTMAX_LAYER")
			{
				addSoftmaxLayer();
				//cout << "SoftmaxLayer added" << endl;
			}
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

	//put in Net hyperparameters. NOT TRAINING OR REAL DATA
	sprintf(data,"%d", n_activationType);
	out += "activationType=";
	out += data;
	out += '\n';

	vector<vector<vector<float> > > blankVector = n_blankInput.getNeurons();

	sprintf(data,"%lu",blankVector.size());
	out += "inputWidth=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",blankVector[0].size());
	out += "inputHeight=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",blankVector[0][0].size());
	out += "inputDepth=";
	out += data;
	out += '\n';

	out += "END_NET\n";

	for(int l=1; l < n_layers.size(); l++)
	{
		int type = n_layers[l]->getType();
		if(type == Net::MAX_POOL_LAYER)
		{
			MaxPoolLayer *pool = (MaxPoolLayer*)n_layers[l];
			out += "MAX_POOL_LAYER\n";

			out += pool->getHyperParameters();

			out += "END_MAX_POOL_LAYER\n";
		}
		else if(type == Net::ACTIV_LAYER)
		{
			ActivLayer *act = (ActivLayer*)n_layers[l];
			out += "ACTIV_LAYER\n";

			out += act->getHyperParameters();

			out += "END_ACTIV_LAYER\n";

		}
		else if(type == Net::CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)n_layers[l];
			out += "CONV_LAYER\n";

			out += conv->getHyperParameters();

			out += "END_CONV_LAYER\n";
		}
		else if(type == Net::SOFTMAX_LAYER)
		{
			out += "SOFTMAX_LAYER\n";
		}
	}
	out += "END_ALL";
	file << out;
	file.close();

	return true;
}

/**********************
 * InputLayer
 **********************/

const int InputLayer::i_type = Net::INPUT_LAYER;

InputLayer::InputLayer()
{
	i_numNeurons = 0;
	i_resizeable = true;
}

InputLayer::~InputLayer(){}

InputLayer::InputLayer(const vector<vector<vector<float> > >& trainingImage, vector<vector<vector<float> > >* blankdNeurons)
{
	i_neurons = &trainingImage;
	//resize3DVector(i_dneurons,trainingImage.size(),trainingImage[0].size(),trainingImage[0][0].size());
	i_dneurons = blankdNeurons;
	i_resizeable = false;

	i_numNeurons = trainingImage.size() * trainingImage[0].size() * trainingImage[0][0].size();
}

int InputLayer::getType() const
{
	return i_type;
}

float* InputLayer::getImage() const
{
	float *image = new float[i_numNeurons];
	int imIndex = 0;
	for(int i=0; i< i_neurons->size(); i++)
	{
		for(int j=0; j< (*i_neurons)[i].size(); j++)
		{
			for(int k=0; k< (*i_neurons)[i][j].size(); k++)
			{
				image[imIndex++] = (*i_neurons)[i][j][k];
			}
		}
	}
	return image;
}

void InputLayer::getImage(float* image, int size) const
{
	//cout << "here" << endl;
	int start = 0;
	if(size == i_numNeurons)
	{
		for(int i=0; i< i_neurons->size(); i++)
		{
			for(int j=0; j< (*i_neurons)[i].size(); j++)
			{
				for(int k=0; k< (*i_neurons)[i][j].size(); k++)
				{
					image[start++] = (*i_neurons)[i][j][k];
					//cout << (*i_neurons)[i][j][k] << endl;
				}
			}
		}
	}
	else
	{
		cout << size << " != " << i_numNeurons << endl;
	}
}

int InputLayer::getImageSize() const
{
	return i_numNeurons;
}

int InputLayer::getNumNeurons() const
{
	return i_numNeurons;
}

void InputLayer::forwardprop(const Layer& prevLayer){}
void InputLayer::backprop(Layer& prevLayer){}

const vector<vector<vector<float> > >& InputLayer::getNeurons() const
{
	//cout << "getNeurons: " << i_neurons->size() << endl;
	//cout << "neuron mem: " << i_neurons << endl;
	return *i_neurons;
}

vector<vector<vector<float> > >& InputLayer::getdNeurons()
{
	return *i_dneurons;
}

bool InputLayer::setImage(const vector<vector<vector<float> > >* trainingImage, vector<vector<vector<float> > >* blankdNeurons)
{
	if(i_resizeable)
	{
		//cout << "In setImage.\ntrainingImage.size() = " <<trainingImage->size() << endl;
		//const vector<vector<vector<float> > >& t = trainingImage;
		i_neurons = trainingImage;
		i_dneurons = blankdNeurons;
		//cout << "trainingImage mem " << trainingImage  << endl;
		//vectorClone(trainingImage,*i_neurons);
		//cout << "i_neurons.size() = " << i_neurons->size() << endl;
		i_resizeable = false;
		i_numNeurons = trainingImage->size() * (*trainingImage)[0].size() * (*trainingImage)[0][0].size();
		return true;
	}
	return false;
}

/**********************
 * ConvLayer
 **********************/

const int ConvLayer::c_type = Net::CONV_LAYER;

ConvLayer::ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
{
	init(prevLayer,numFilters,stride,filterSize,pad);

	//need some way to initialize, save, and load weights and biases
	initRandomWeights();
}

ConvLayer::ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
{
	//cout << "Init ConvLayer with loaded weights" << endl;
	init(prevLayer,numFilters,stride,filterSize,pad);

	//need some way to initialize, save, and load weights and biases
	initWeights(weightsAndBiases);
}

void ConvLayer::init(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
{
	// need to set size of neurons, dneurons, weights, dweights. Also make sure that the sizes and such match
	// up with the amount of padding and stride. If padding is greater than 0 will need to make new 3d vector 
	// to hold padded array (or use if statements?) DO IN FORWARD PROP AND MAKE LOCAL VARIABLE.
	
	//cout << "prevLayer.getNeurons().size() = " << prevLayer.getNeurons().size() << endl;

	const vector<vector<vector<float> > > prevNeurons = prevLayer.getNeurons();
	int prevWidth = prevNeurons.size();
	int prevHeight = prevNeurons[0].size();
	int prevDepth = prevNeurons[0][0].size();

	c_prevNeuronWidth = prevWidth;
	c_prevNeuronHeight = prevHeight;
	c_prevNeuronDepth = prevDepth;

	//size of new layer (prevWidth - filterSize + 2*pad)/stride+1 x (prevHeight - filterSize + 2pad)/stride+1 
	// x numFilters
	float fnewWidth = (prevWidth - filterSize + 2*pad)/((float)stride) + 1;
	int newWidth = (prevWidth - filterSize + 2*pad)/stride + 1;
	float fnewHeight = (prevHeight - filterSize + 2*pad)/((float)stride) + 1;
	int newHeight = (prevHeight - filterSize + 2*pad)/stride + 1;
	int newDepth = numFilters;

	if(fabs(fnewHeight - newHeight) > .01 || fabs(fnewWidth - newWidth) > .01 )
	{
		cout << "Hyperparameters lead to bad size. Must have this equation be an int.\n\t(prevWidth - filterSize + 2*pad)/stride + 1;";
		throw "Incorrect hyperparameters";
	}

	c_stride = stride;
	c_padding = pad;

	resize3DVector(c_neurons,newWidth,newHeight,newDepth);
	resize3DVector(c_dneurons,newWidth,newHeight,newDepth);
	c_numNeurons = newWidth * newHeight * newDepth;

	int padVecSize = (prevWidth+2*pad)*(prevHeight+2*pad)*prevDepth;

	c_maxSizeNeeded = GETMAX(c_numNeurons, padVecSize);
	
	//Size of c_weights (and c_dweights) will be numFilters x filterSize x filterSize x numFilters
	c_weights.resize(numFilters);
	c_dweights.resize(numFilters);
	for(int i=0; i< numFilters; i++)
	{
		resize3DVector(c_weights[i], filterSize, filterSize, prevDepth);
		resize3DVector(c_dweights[i], filterSize, filterSize, prevDepth);
	}
	//cout << numFilters << "x" << filterSize << "x" << filterSize << "x" << prevDepth << endl;
	c_numWeights = numFilters * filterSize * filterSize * prevDepth;
	//cout << c_numWeights << endl;

	c_biases.resize(numFilters);
	c_dbiases.resize(numFilters);

	c_numBiases = numFilters;
}

ConvLayer::~ConvLayer(){}

int ConvLayer::getType() const
{
	return c_type;
}

int ConvLayer::getNumNeurons() const
{
	return c_numNeurons;
}

int ConvLayer::getNumWeights() const
{
	return c_numWeights;
}

int ConvLayer::getMaxSizeNeeded() const
{
	return c_maxSizeNeeded;
}

void ConvLayer::initRandomWeights()
{
	default_random_engine gen(time(0));
	uniform_real_distribution<float> distr(-.005,.005);
	for(int f = 0;f<c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					//float rnum = distr(gen);
					c_weights[f][i][j][k] = distr(gen);//rnum;
					//cout << rnum << endl;
				}
			}
		}
	}

	for(int b=0; b< c_biases.size(); b++)
	{
		c_biases[b] = distr(gen);
	}


}

void ConvLayer::initWeights(string weights)
{
	float weight;
	int startIndex = 0, endIndex;
	for(int f = 0;f<c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					endIndex = weights.find(',',startIndex);
					weight = stod(weights.substr(startIndex,endIndex));
					c_weights[f][i][j][k] = weight;
					startIndex = endIndex + 1;
				}
			}
		}
	}
	// now do the biases
	startIndex = weights.find('_') + 1;
	for(int b=0; b < c_biases.size(); b++)
	{
		endIndex = weights.find(',',startIndex);
		c_biases[b] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;
	}
}

vector<int> ConvLayer::getKernelHyperParameters() const
{
	vector<int> hyper(7);
	hyper[0] = c_weights[0].size(); // filterSize
	hyper[1] = c_stride;
	hyper[2] = c_prevNeuronWidth; // width of prevneurons
	hyper[3] = c_prevNeuronDepth;
	hyper[4] = c_biases.size();  // num filters
	hyper[5] = c_padding;
	hyper[6] = c_prevNeuronHeight;

	return hyper;
}

void ConvLayer::_putWeights(float* weights, int vectIndex) const
{
	int start = c_weights[0].size() * c_weights[0][0].size() * c_weights[0][0][0].size() * vectIndex;
	//printf("%d %d\n", start, vectIndex);
	for(int i=0; i< c_weights[vectIndex].size(); i++)
	{
		for(int j=0; j< c_weights[vectIndex][i].size(); j++)
		{
			for(int k=0; k< c_weights[vectIndex][i][j].size(); k++)
			{
				weights[start++] = c_weights[vectIndex][i][j][k];
				//weights[start++] = c_weights.at(vectIndex).at(i).at(j).at(k);

			}
		}
	}
}

float* ConvLayer::getWeights() const 
{
	//cout << "allocing " << c_numWeights << " weights" << endl;
	//cout << c_weights.size() << "x" << c_weights[0].size() << "x" << c_weights[0][0].size() << "x" << c_weights[0][0][0].size() << endl;
	//int numWeights = c_weights.size() * c_weights[0].size() * c_weights[0][0].size();
	float *weights = new float[c_numWeights];

	//cout << "allocing threads" << endl;
	thread *t = new thread[c_weights.size()];

	for(int f=0; f< c_weights.size(); f++)
	{
		t[f] = thread(&ConvLayer::_putWeights,this,weights,f);
	}

	//cout << "joining threads" << endl;
	for(int f = 0; f < c_weights.size(); f++)
	{
		t[f].join();
	}

	//delete t;

	//cout << "end conv" << endl;
	return weights;

}

int ConvLayer::getNumBiases() const
{
	return c_numBiases;
}

float* ConvLayer::getBiases() const
{
	float *biases = new float[c_numBiases];

	for(int b = 0; b < c_numBiases; b++)
	{
		biases[b] = c_biases[b];
	}

	return biases;
}

void ConvLayer::forwardprop(const Layer& prevLayer)
{
	vector<vector<vector<float> > > padSource;
	const vector<vector<vector<float> > > *source;
	if(c_padding != 0)
	{
		padZeros(prevLayer.getNeurons(),c_padding,padSource);
		source = &padSource;
	}
	else
		source = &prevLayer.getNeurons();
	float sum;
	int oX, oY;
	int subsetWidth = c_weights[0].size(); // this is the same as filterSize
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i <= source->size()-subsetWidth; i+= c_stride) // row in source   		i < source.size()-1
		{
			oY = 0;
			for(int j=0; j <= (*source)[i].size()-subsetHeight;j+=c_stride) // col in source	j < source[i].size()-1
			{
				//now we go into the stride subset
				sum = 0;
				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < (*source)[i][j].size(); k++) // depth in source
						{
							sum += (*source)[i+s][j+r][k] * c_weights[f][s][r][k];
							//sum += source.at(i+s).at(j+r).at(k) * weights.at(f).at(s).at(r).at(k);

							//can I set some of the prevLayer.getdNeuron local values at this point???
						}
					}
				}
				// add bias
				sum += c_biases[f];
				// add into c_neurons[i%stride, j%stride, f]
				c_neurons[oX][oY++][f] = sum;
			}
			oX++;
		}
	}
	/*
	printVector(c_neurons);
	cout << "conv" << endl;
	exit(0);
	*/

	if(Net::walkthrough)
	{
		cout << "In ConvLayer forwardprop" << endl;
		printVector(c_neurons);
		getchar();
	}
}



void ConvLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In ConvLayer backprop" << endl;
		printVector(c_dneurons);
		getchar();
	}

	vector<vector<vector<float> > >& p_dNeurons = prevLayer.getdNeurons();
	vector<vector<vector<float> > > padded_dNeurons;
	resize3DVector(padded_dNeurons,p_dNeurons.size() + 2*c_padding,p_dNeurons[0].size() + 2*c_padding, p_dNeurons[0][0].size());
	setAll3DVector(p_dNeurons, 0);
	//setAll1DVector(c_dbiases, 0);
	for(int b=0; b< c_dbiases.size(); b++)
	{
		c_dbiases[b] = 0;
	}
	setAll4DVector(c_dweights, 0);

	vector<vector<vector<float> > > padSource;
	const vector<vector<vector<float> > > *source;
	if(c_padding != 0)
	{
		padZeros(prevLayer.getNeurons(),c_padding,padSource);
		source = &padSource;
	}
	else
		source = &prevLayer.getNeurons();

	//vector<vector<vector<float> > > source;
	//padZeros(prevLayer.getNeurons(),c_padding,source);

	int oX, oY; // outX & outY
	int subsetWidth = c_weights[0].size();
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i <= source->size()-subsetWidth; i+= c_stride) // row in source
		{
			oY = 0;
			for(int j=0; j <= (*source)[i].size()-subsetHeight;j+=c_stride) // col in source
			{
				//now we go into the stride subset				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < (*source)[i][j].size(); k++) // depth in source
						{
							padded_dNeurons[i+s][j+r][k] += c_dneurons[oX][oY][f] * c_weights[f][s][r][k];
							c_dweights[f][s][r][k]  	 += c_dneurons[oX][oY][f] * (*source)[i+s][j+r][k];

							//padded_dNeurons.at(i+s).at(j+r).at(k) += c_dneurons.at(oX)[oY][f] * c_weights.at(f).at(s).at(r).at(k);
							//c_dweights.at(f).at(s).at(r).at(k) += c_dneurons.at(oX)[oY][f] * source.at(i+s).at(j+r).at(k);
						}
					}
				}
				// add bias
				c_dbiases[f] += c_dneurons[oX][oY++][f];
				//c_dbiases.at(f) += c_dneurons.at(oX).at(oY++).at(f);
			}
			oX++;
		}
	}


	//put the padded_dNeurons into the real p_dNeurons
	for(int i=c_padding; i < padded_dNeurons.size() - c_padding; i++)
	{
		for(int j=c_padding; j < padded_dNeurons[i].size()-c_padding; j++)
		{
			for(int k=0; k < padded_dNeurons[i][j].size(); k++)
			{
				p_dNeurons[i-c_padding][j-c_padding][k] = padded_dNeurons[i][j][k];
			}
		}
	}

	//end here if doing gradient check cause we don't want to update the weights

	//we need to substract the gradient * the stepSize from the weights and biases
	if(!Net::gradCheck)
	{
		//update the weights
		for(int f=0;f<c_weights.size();f++)
		{
			for(int i=0; i< c_weights[0].size(); i++)
			{
				for(int j=0; j< c_weights[0][0].size(); j++)
				{
					for(int k=0; k< c_weights[0][0][0].size(); k++)
					{
						//if(c_dweights[f][i][j][k] != 0)
							//cout << c_dweights[f][i][j][k] << endl;
						c_weights[f][i][j][k] -= Net::stepSize * c_dweights[f][i][j][k];
					}
				}
			}
		}

		//update the biases
		for(int i=0; i< c_biases.size(); i++)
		{
			c_biases[i] -= Net::stepSize * c_dbiases[i];
		}	
	}
}

const vector<vector<vector<float> > >& ConvLayer::getNeurons() const
{
	return c_neurons;
}

vector<vector<vector<float> > >& ConvLayer::getdNeurons()
{
	return c_dneurons;
}

string ConvLayer::getHyperParameters() const
{
	char data[50];
	string out = "";

	sprintf(data,"%d",c_stride);
	out += "stride=";
	out += data;
	out += '\n';

	sprintf(data,"%d",c_padding);
	out += "padding=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",c_weights.size());
	out += "numFilters=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",c_weights[0].size());
	out += "filterSize=";
	out += data;
	out += '\n';

	out += "weights=";
	for(int f=0; f < c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					sprintf(data,"%lf,",c_weights[f][i][j][k]);
					out += data;
				}
			}
		}
	}
	//out += '\n';

	//out += "biases=";
	out += "_";
	for(int b=0; b< c_biases.size(); b++)
	{
		sprintf(data,"%lf,",c_biases[b]);
		out += data;
	}
	out += '\n';

	return out;
}

/**********************
 * MaxPoolLayer
 **********************/

const int MaxPoolLayer::m_type = Net::MAX_POOL_LAYER;

MaxPoolLayer::MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride)
{
 	// need to set size of neurons and dneurons and make sure it goes evenly across new neurons
 	const vector<vector<vector<float> > > prevNeurons = prevLayer.getNeurons();
 	//cout << "prevLayer.getNeurons().size() = " << prevNeurons.size() << endl;
 	int pWidth = prevNeurons.size();
 	int pHeight = prevNeurons[0].size();
 	int pDepth = prevNeurons[0][0].size();

 	// new volume is (W1 - poolSize)/stride + 1 x (H1 - poolSize)/stride + 1 x D1
 	float fnewWidth = (pWidth - poolSize)/((float)stride) + 1;
 	int newWidth = (pWidth - poolSize)/stride + 1;
 	float fnewHeight = (pHeight - poolSize)/((float)stride) + 1;
 	int newHeight = (pHeight - poolSize)/stride + 1;
 	int newDepth = pDepth;

 	if(fabs(fnewHeight - newHeight) > .01 || fabs(fnewWidth - newWidth) > .01 )
	{
		cout << "Hyperparameters lead to bad size. Must have this equation be an int.\n\t(prevWidth - poolSize)/stride + 1;";
		throw "Incorrect hyperparameters";
	}

	m_stride = stride;
	m_poolSize = poolSize;
	m_prevWidth = pWidth;
	m_prevDepth = pDepth;

	resize3DVector(m_neurons,newWidth,newHeight,newDepth);
	resize3DVector(m_dneurons,newWidth,newHeight,newDepth);
	m_numNeurons = newWidth * newHeight * newDepth;
}

MaxPoolLayer::~MaxPoolLayer(){}

int MaxPoolLayer::getType() const
{
	return m_type;
}

int MaxPoolLayer::getNumNeurons() const
{
	return m_numNeurons;
}

vector<int> MaxPoolLayer::getKernelHyperParameters() const
{
	vector<int> hyper(4);
	hyper[0] = m_prevWidth;
	hyper[1] = m_prevDepth;
	hyper[2] = m_poolSize;
	hyper[3] = m_stride;

	return hyper;
}

void MaxPoolLayer::forwardprop(const Layer& prevLayer)
{
	const vector<vector<vector<float> > >& source = prevLayer.getNeurons();
	int oX=0, oY=0;//, di, dj; //, dk;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size()-1;i+=m_stride) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size()-1;j+=m_stride) // cols
			{
				float maxVal = source[i][j][k];
				for(int s=0;s<m_poolSize;s++)
				{
					for(int r=0; r< m_poolSize; r++)
					{
						if(source[i+s][j+r][k] >= maxVal)
						{
							maxVal = source[i+s][j+r][k];
						}
					}
				}
				m_neurons[oX][oY++][k] = maxVal;
			}
			oX++;
		}
	}

	/*
	cout << m_neurons[0][0][0] << endl;
	printVector(m_neurons);
	cout << "max layer. size " << getNumNeurons() << endl;
	exit(0);*/

	if(Net::walkthrough)
	{
		cout << "In MaxPoolLayer forwardprop" << endl;
		printVector(m_neurons);
		getchar();
	}
}

void MaxPoolLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In MaxPoolLayer backprop" << endl;
		printVector(m_dneurons);
		getchar();
	}

	//m_dneurons should have been set at this point by the next layer's backprop
	//set all prev dNeurons to 0
	vector<vector<vector<float> > >& p_dNeurons = prevLayer.getdNeurons();
 	setAll3DVector(p_dNeurons, 0);
 	const vector<vector<vector<float> > >& source = prevLayer.getNeurons();

 	//for each set in maxPool, add in the m_dneurons val to the max. All others will stay 0.
	int oX=0, oY=0, di, dj; //, dk;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size()-1;i+=m_stride) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size()-1;j+=m_stride) // cols
			{
				float maxVal = source[i][j][k];
				for(int s=0;s<m_poolSize;s++)
				{
					for(int r=0; r< m_poolSize; r++)
					{
						if(source[i+s][j+r][k] >= maxVal)
						{
							maxVal = source[i+s][j+r][k];
							di = i+s;
							dj = j+r;
							//dk = k;
						}

					}
				}
				p_dNeurons[di][dj][k] += m_dneurons[oX][oY++][k];				
			}
			oX++;
		}
	}
}


const vector<vector<vector<float> > >& MaxPoolLayer::getNeurons() const
{
	return m_neurons;
}

vector<vector<vector<float> > >& MaxPoolLayer::getdNeurons()
{
	return m_dneurons;
}

string MaxPoolLayer::getHyperParameters() const
{
	string out = "";
	char data[50];

	sprintf(data,"%d",m_stride);
	out += "stride=";
	out += data;
	out += '\n';

	sprintf(data,"%d",m_poolSize);
	out += "poolSize=";
	out += data;
	out += '\n';

	return out;
}

/**********************
 * ActivLayer
 **********************/

const int ActivLayer::a_type = Net::ACTIV_LAYER;

const float ActivLayer::LEAKY_RELU_CONST = .01;

const float ActivLayer::RELU_CAP = 5000;

ActivLayer::ActivLayer(const Layer& prevLayer, const int activationType)
{
	int notFound = true;
	//check if valid activation Type
	if(0 <= activationType && activationType < ActivLayer::NUM_ACTIV_TYPES)
	{ 
		notFound = false;
	}
	
	if(notFound)
	{
		cout << "The activationType was not valid. Please use a valid activationType. Exp: ActivLayer::RELU";
		throw "Invalid activationType";
	}

	//cout << "prevLayer.getNeurons().size() = " << prevLayer.getNeurons().size() << endl;

	a_activationType = activationType;
	vector<vector<vector<float> > > prevNeurons = prevLayer.getNeurons();
	int w = prevNeurons.size();
	int h = prevNeurons[0].size();
	int d = prevNeurons[0][0].size();
	resize3DVector(a_neurons,w,h,d);
	resize3DVector(a_dneurons,w,h,d);
	a_numNeurons = w * h * d;
}

ActivLayer::~ActivLayer(){}

int ActivLayer::getType() const
{
	return a_type;
}

int ActivLayer::getNumNeurons() const
{
	return a_numNeurons;
}

int ActivLayer::getActivationType() const
{
	return a_activationType;
}

void ActivLayer::forwardprop(const Layer& prevLayer)
{
	const vector<vector<vector<float> > >& prevNeurons = prevLayer.getNeurons();

	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					a_neurons[i][j][k] = GETMAX(0,prevNeurons[i][j][k]);
					if(a_neurons[i][j][k] > ActivLayer::RELU_CAP)
						a_neurons[i][j][k] = ActivLayer::RELU_CAP;
				}
			}
		}
	}
	else if (a_activationType == ActivLayer::LEAKY_RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] < 0)
					{
						a_neurons[i][j][k] = LEAKY_RELU_CONST * prevNeurons[i][j][k];
						if(a_neurons[i][j][k] < -ActivLayer::RELU_CAP)
							a_neurons[i][j][k] = -ActivLayer::RELU_CAP;
					}
					else if(prevNeurons[i][j][k] > ActivLayer::RELU_CAP)
					{
						a_neurons[i][j][k] = ActivLayer::RELU_CAP;
					}
					else
						a_neurons[i][j][k] = prevNeurons[i][j][k];
				}
			}
		}
	}

	/*
	cout << a_neurons[0][0][0]<<endl;
	printVector(a_neurons);
	cout << "activLayer" << endl;
	exit(0);
	*/

	if(Net::walkthrough)
	{
		cout << "In ActivLayer forwardprop" << endl;
		printVector(a_neurons);
		getchar();
	}
}

void ActivLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In ActivLayer backprop" << endl;
		printVector(a_dneurons);
		getchar();
	}

	const vector<vector<vector<float> > >& prevNeurons = prevLayer.getNeurons();
	vector<vector<vector<float> > >& p_dNeurons = prevLayer.getdNeurons();
	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] >= 0 && prevNeurons[i][j][k] <= ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = a_dneurons[i][j][k];
					}
					else
					{
						p_dNeurons[i][j][k] = 0;
					}
				}
			}
		}
	}
	else if(a_activationType == ActivLayer::LEAKY_RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] >= 0 && prevNeurons[i][j][k] <= ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = a_dneurons[i][j][k];
					}
					else if(prevNeurons[i][j][k] >= -ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = LEAKY_RELU_CONST * a_dneurons[i][j][k];
					}
					else
					{
						p_dNeurons[i][j][k] = 0;
					}
				}
			}
		}
	}
}

const vector<vector<vector<float> > >& ActivLayer::getNeurons() const
{
	return a_neurons;
}

vector<vector<vector<float> > >& ActivLayer::getdNeurons()
{
	return a_dneurons;
}

string ActivLayer::getHyperParameters() const
{
	string out = "";
	char data[50];

	sprintf(data,"%d",a_activationType);
	out += "activationType=";
	out += data;
	out += '\n';

	return out;
}

/**********************
 * SoftmaxLayer
 **********************/

 const int SoftmaxLayer::s_type = Net::SOFTMAX_LAYER;

 SoftmaxLayer::SoftmaxLayer(const Layer& prevLayer)
 {
 	const vector<vector<vector<float> > >& prevNeurons = prevLayer.getNeurons();
 	int numDirs = 0;
	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {numDirs++; dir = &i; s_neurons.resize(width);}
	if(height > 1) {numDirs++; dir = &j; s_neurons.resize(height);}
	if(depth  > 1) {numDirs++; dir = &k; s_neurons.resize(depth);}
	if(numDirs != 1 || width < 1 || height < 1 || depth < 1)
	{
		throw "Incorrect dimensions";
	}
	s_dneurons.resize(s_neurons.size());

	s_3neurons.resize(1);
	s_3neurons[0].resize(1);
	s_3neurons[0][0] = s_neurons;

	s_3dneurons.resize(1);
	s_3dneurons[0].resize(1);
	s_3dneurons[0][0] = s_dneurons;
 }

 SoftmaxLayer::~SoftmaxLayer(){}

 int SoftmaxLayer::getNumNeurons() const
 {
 	return s_neurons.size();
 }

 void SoftmaxLayer::forwardprop(const Layer& prevLayer)
 {
 	const vector<vector<vector<float> > >& prevNeurons = prevLayer.getNeurons();
	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {dir = &i;}
	else if(height > 1) {dir = &j;}
	else {dir = &k;}

	for(i=0;i<width;i++)
	{
		for(j=0;j<height;j++)
		{
			for(k=0;k<depth;k++)
			{
				s_neurons[*dir] = prevNeurons[i][j][k];
			}
		}
	}

	maxSubtraction(s_neurons);
	float denom = vectorESum(s_neurons);
	for(int n=0; n < s_neurons.size(); n++)
	{
		s_neurons[n] = exp(s_neurons[n])/denom;
	}

	if(Net::walkthrough)
	{
		cout << "In SoftmaxLayer forwardprop" << endl;
		printVector(s_neurons);
		getchar();
	}

 }

 void SoftmaxLayer::setTrueVal(int trueVal)
 {
 	if(trueVal < 0 || s_neurons.size() <= trueVal)
 		throw "Invalid trueVal";
 	s_trueVal = trueVal;
 }

 void SoftmaxLayer::gradientCheck(Layer& prevLayer)
 {
 	vector<vector<vector<float> > >& prevdNeurons = prevLayer.getdNeurons();
 	const vector<vector<vector<float> > >& prevNeurons = prevLayer.getNeurons();
 	cout << "\n\n";
 	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {dir = &i;}
	else if(height > 1) {dir = &j;}
	else {dir = &k;}
 	for(i = 0; i< width; i++)
 	{
 		for(j=0; j< height; j++)
 		{
 			for(k=0; k < depth; k++)
 			{
 				cout << "Softmax prevNeurons["<<i<<"]["<<j<<"]["<<k<<"]:\n";
 				cout << "Analytical Gradient:\t" << prevdNeurons[i][j][k]<< endl;
 				cout << "old s_neurons: ";
 				printVector(s_neurons);

 				vector<float> oldError = getError();
 				s_neurons[*dir] += Net::GRADCHECK_H;

 				//from forwardprop
 				maxSubtraction(s_neurons);
				float denom = vectorESum(s_neurons);
				for(int n=0; n < s_neurons.size(); n++)
				{
					s_neurons[n] = exp(s_neurons[n])/denom;
				}

				if(Net::walkthrough)
				{
					cout << "In Gradient SoftmaxLayer forwardprop" << endl;
					printVector(s_neurons);
					getchar();
				}
				//end from forwardprop

				cout << "new s_neurons: ";
				printVector(s_neurons);

				vector<float> newError = getError();
				float numericalGradient = 0;
				for(int e=0; e<newError.size(); e++)
				{
					cout << (newError[e] - oldError[e])/Net::GRADCHECK_H << endl;
					numericalGradient += (newError[e] - oldError[e])/Net::GRADCHECK_H;
				}
				cout << "Numberical Gradient:\t" << numericalGradient << endl;

 				s_neurons[*dir] -= Net::GRADCHECK_H;

 				cout <<"\n\n";
 			}
 		}
 	}
 }

 void SoftmaxLayer::backprop(Layer& prevLayer)
 {
 	for(int i=0; i< s_neurons.size(); i++)
 	{
 		if(i == s_trueVal)
 		{
 			//s_dneurons[i] = 1 - s_neurons[i];
 			s_dneurons[i] = s_neurons[i] - 1;
 		}
 		else
 		{
 			//s_dneurons[i] = -s_neurons[i];
 			s_dneurons[i] = s_neurons[i];
 		}
 	}

 	if(Net::walkthrough || Net::showErrors)
	{
		cout << "In SoftmaxLayer backprop" << endl;
		//printVector(s_neurons);
		printVector(s_dneurons);
		if(Net::walkthrough)
			getchar();
	}

 	vector<vector<vector<float> > >& prevdNeurons = prevLayer.getdNeurons();
 	int n = 0;
 	for(int i=0; i< prevdNeurons.size(); i++)
 	{
 		for(int j=0; j< prevdNeurons[i].size(); j++)
 		{
 			for(int k=0; k< prevdNeurons[i][j].size(); k++)
 			{
 				prevdNeurons[i][j][k] = s_dneurons[n++];
 			}
 		}
 	}
 }

 int SoftmaxLayer::getPredictedClass()
 {
 	int maxLoc = 0;
	for(int i=1; i<s_neurons.size();i++)
	{
		if(s_neurons[i] > s_neurons[maxLoc])
		{
			maxLoc = i;
		}
	}
	//cout << maxLoc << endl;
	return maxLoc;
 }

 vector<float> SoftmaxLayer::getError() 
 {
 	for(int i=0; i< s_neurons.size(); i++)
 	{
 		if(i == s_trueVal)
 		{
 			s_dneurons[i] = 1 - s_neurons[i];
 		}
 		else
 		{
 			s_dneurons[i] = -s_neurons[i];
 		}
 	}
 	if(Net::showErrors)
 		printVector(s_dneurons);
 	return s_dneurons;
 }

 void SoftmaxLayer::setError(vector<float> error)
 {
 	if(error.size() != s_dneurons.size())
 	{
 		throw "Incorrect error vector size";
 	}
 	for(int i=0;i<error.size(); i++)
 	{
 		s_dneurons[i] = error[i];
 	}
 }

 int SoftmaxLayer::getType() const 
 {
 	return s_type;
 }

 const vector<vector<vector<float> > >& SoftmaxLayer::getNeurons() const
 {
 	return s_3neurons;
 } 

 vector<vector<vector<float> > >& SoftmaxLayer::getdNeurons()
 {
 	return s_3dneurons;
 }

/***********************************
 * Functions
 ***********************************/
void softmax(const vector<vector<vector<float> > >& vect, vector<float>& out)
{
	//cout << "orig vector";
	//printVector(vect);
	int numDirs = 0;
	int i,j,k, *dir;
	int width = vect.size();
	int height = vect[0].size();
	int depth = vect[0][0].size();
	if(width  > 1) {numDirs++; dir = &i; out.resize(width);}
	if(height > 1) {numDirs++; dir = &j; out.resize(height);}
	if(depth  > 1) {numDirs++; dir = &k; out.resize(depth);}
	if(numDirs != 1 || width < 1 || height < 1 || depth < 1)
	{
		throw "Incorrect dimensions";
	}
	//if(dir == &i) cout << "i" << endl;
	//if(dir == &j) cout << "j" << endl;
	//if(dir == &k) cout << "k" << endl;
	for(i=0;i<width;i++)
	{
		for(j=0;j<height;j++)
		{
			for(k=0;k<depth;k++)
			{
				out[*dir] = vect[i][j][k];
			}
		}
	}

	//now its in a 1D array

	//subtract the max for numerical stability
	maxSubtraction(out);
	//cout << "Pre exp" << endl;
	//printVector(out);
	float denom = vectorESum(out);
	//cout << "denom " << denom << endl;
	for(int n=0; n< out.size(); n++)
	{
		out[n] = exp(out[n])/denom;
	}
	//cout << "Final array ";
	//printVector(out);
}

void resize3DVector(vector<vector<vector<float> > > &vect, int width, int height, int depth)
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

void setAll4DVector(vector<vector<vector<vector<float> > > > &vect, float val)
{
	for(int n=0; n< vect.size(); n++)
	{
		setAll3DVector(vect[n],val);
	}
}

void setAll3DVector(vector<vector<vector<float> > > &vect, float val)
{
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = val;
			}
		}
	}
}

void printVector(const vector<vector<vector<vector<float> > > > &vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		printVector(vect[n]);
		cout << endl;
	}
}

void printVector(const vector<vector<vector<float> > > &vect)
{
	for(int i=0;i < vect.size();i++) // rows
	{
		cout << "|";
		for(int j=0;j < vect[i].size();j++)
		{
			for(int k=0; k < vect[i][j].size(); k++)
			{
				cout << setw(4) << vect[i][j][k];
				if(k != vect[i][j].size()-1) cout << ",";
			}
			cout << "|";
		}
		cout << endl;
	}
}

void printVector(const vector<float>& vect)
{
	cout << "|";
	for(int i=0; i< vect.size(); i++)
	{
		cout << setw(4) << vect[i];
		if(i != vect.size()-1) cout << ",";
	}
	cout << "|" << endl;
}

/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
void padZeros(const vector<vector<vector<float> > > &source, int numZeros, vector<vector<vector<float> > > &dest)
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
				dest[i][j][k] = source[i-numZeros][j-numZeros][k];
			}
		}
	}
}


float vectorESum(const vector<float> &source)
{
	float sum = 0;
	for(int i=0; i < source.size(); i++)
	{
		sum += exp(source[i]);
	}
	return sum;
}

void vectorClone(const vector<vector<vector<float> > > &source, vector<vector<vector<float> > > &dest)
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

void compressImage(vector<vector<vector<vector<float> > > >& vect, float newMin, float newMax)
{
	for(int n=0; n< vect.size(); n++)
	{
		compressImage(vect[n],newMin, newMax);
	}
}

void compressImage(vector<vector<vector<float> > >& vect, float newMin, float newMax)
{
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = vect[i][j][k]/255 * (newMax-newMin) + newMin;
			}
		}
	}
}

void maxSubtraction(vector<float>& vect)
{
	float max = vect[0];
	for(int i=1;i<vect.size();i++)
	{
		if(vect[i] > max)
			max = vect[i];
	}
	for(int i=0; i< vect.size(); i++)
	{
		vect[i] -= max;
	}
}

void preprocess(vector<vector<vector<vector<float> > > >& vect)
{
	for(int v=0; v< vect.size(); v++)
	{
		preprocess(vect[v]); //thread this
	}
}

void preprocess(vector<vector<vector<float> > > & vect)
{
	//preprocess using (val - mean)/stdDeviation for all elements
	float m = mean(vect);
	float stddv = stddev(vect,m);
	for(int i=0; i< vect.size();i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = (vect[i][j][k] - m)/stddv;
			}
		}
	}
}

float mean(const vector<vector<vector<float> > > & vect)
{
	float mean = 0;
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				mean += vect[i][j][k];
			}
		}
	}
	mean /= vect.size() * vect[0].size() * vect[0][0].size();
	return mean;
}

float stddev(const vector<vector<vector<float> > > & vect) 
{
	float m = mean(vect);
	return stddev(vect,m);
}

float stddev(const vector<vector<vector<float> > > & vect, float mean) 
{
	float sqdiffs = 0;
	float temp;
	for(int i=0; i< vect.size();i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				temp = (vect[i][j][k] - mean);
				sqdiffs += temp * temp;
			}
		}
	}
	float sqdiffMean = sqdiffs / (vect.size() * vect[0].size() * vect[0][0].size());
	return sqrt(sqdiffMean);
}

void meanSubtraction(vector<vector<vector<vector<float> > > >& vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		meanSubtraction(vect[n]);
	}
}

void meanSubtraction(vector<vector<vector<float> > >& vect)
{
	float mean = 0;
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				mean += vect[i][j][k];
			}
		}
	}
	mean /= vect.size() * vect[0].size() * vect[0][0].size();

	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				vect[i][j][k] -= mean;;
			}
		}
	}
}

void meanSubtraction(vector<float>& vect)
{
	float mean = 0;
	for(int i=0;i<vect.size();i++)
	{
		mean += vect[i];
	}
	mean /= vect.size();
	for(int i=0; i< vect.size(); i++)
	{
		vect[i] -= mean;
	}
}

void printArray(float* array, int size)
{
	for(int i=0; i< size; i++)
	{
		cout << array[i] << ", ";
	}
	cout << endl;
}

int getMaxElementIndex(const vector<float>& vect)
{
	if(vect.size() < 1)
		return -1;
	float max = vect[0];
	float maxIndex = 0;
	for(int i=1; i< vect.size(); i++)
	{
		if(vect[i] > max)
		{
			max = vect[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

cl_program CreateProgram (const std::string& source, cl_context& context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}
