//
//  ConvNetCL.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNetCL__
#define ____ConvNetCL__

#include <string>
#include <vector>

#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

//defines for layers
#define ABSTRACT_LAYER -1
#define CONV_LAYER 0
#define MAX_POOL_LAYER 1
#define ACTIV_LAYER 2

//defines for ActivTypes
#define RELU 0
#define LEAKY_RELU 1
#define MAX_ACTIV 2

//defines for training types
#define TRAIN_AS_IS 0
#define TRAIN_EQUAL_PROP 1

//defines for programs
#define TRAINING_PROGRAM 0
#define RUNNING_PROGRAM 1

typedef std::vector<std::vector<std::vector<double> > > imVector;

class Net{
private: 	// structs
	struct Layer{
		int layerType = ABSTRACT_LAYER;
	};

	struct ConvLayer : Layer{
		double* weights;
		double* biases;
		int numWeights;
		int numBiases;
		int numNeurons;
		int padding;
		int stride;
		int filterSize;
		int paddedNeuronWidth;
		int paddedNeuronHeight;
		int paddedNeuronSize;
		int maxSizeNeeded;
	};

	struct MaxPoolLayer : Layer{
		int stride;
		int poolSize;
	};

	struct ActivLayer : Layer{
		int activationType;
	};

private: 	// members
	//hyperparameters
	double __learningRate = 1e-4;
	double __RELU_CAP = 5000.0;
	double __LEAKY_RELU_CONST = 0.01;
	double __l2Lambda = 0.05;
	double __MOMENT_CONST = 0.9;
	double __MAX_NORM_CAP = 6.0;
	//members dealing with layers
	std::vector<Layer*> __layers;  //[0] is input layer
	bool __autoActivLayer = true;
	std::vector<int> __neuronSizes; //[0] is input layer
	int __maxNeuronSize;
	std::vector<std::vector<int> > __neuronDims;  //[0] is input layer
	int __defaultActivType = 0;
	int __maxWeightSize = 0;

	bool __isFinalized = false;
	std::string __errorLog;

	//data and related members
	int __numClasses = 0;
		//training
		//should this be a map?
		bool __trainingDataPreprocessed = false;
		bool __testDataPreprocessed = false;
		double __mean = 0;
		double __stddev = 0;
		double __trainingSize = 0;
		bool __isTraining = false;
		std::vector<std::vector<std::vector<double>* > > __trainingData; // class<list of<pointers-to-flattenedImages> >
		std::vector<double> __trueVals; // parallel vector of true values for __trainingData
		std::vector<std::vector<double> > __testData;
		std::vector<double> __testTrueVals;
		bool __useMomentum = true;
		int __trainingType = TRAIN_AS_IS;
		int __smallestClassSize;
		bool __useHorizontalReflections = false;
		//running
		bool __dataPreprocessed = false;
		std::vector<std::vector<double> > __data; // list of<flattened images>
		std::vector<std::vector<double> > *__dataPointer;
		std::vector<std::vector<double> > __confidences; // image<list of confidences for each class<confidence> > 

	//OpenCL related members
	cl_uint __platformIdCount;
	cl_uint __deviceIdCount;
	std::vector<cl_platform_id> __platformIds;
	std::vector<cl_device_id> __deviceIds;
	cl_context __context;
	cl_uint __device = -1;
	bool __useGPU = true;
	bool __constantMem = false;
	bool __stuffBuilt = false;
	cl_program CNForward, CNTraining;
	//running kernels
	cl_kernel reluKernelF, leakyReluKernelF, convKernelF, convKernelFC, maxPoolKernelF, softmaxKernelF, zeroPadKernelF;
	//training kernels
	cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel, reluBackKernel,
		zeroPadBackKernel, softmaxBackKernel, maxPoolBackKernel, leakyReluBackKernel, convBackNeuronsKernel, 
		convBackBiasesKernel, convBackWeightsKernel, copyArrayKernel, convBackWeightsMomentKernel,
		maxSubtractionKernel, vectorESumKernel;
	cl_command_queue queue;
	std::vector<cl_mem> clWeights;
	std::vector<cl_mem> clBiases;
	cl_mem n, p, *neurons, *prevNeurons, denom;

public: 	// functions
	//Constructors and Destructors
	Net(const char* filename);
	Net(int inputWidth, int inputHeight, int inputDepth);
	~Net();
	
	//functions dealing with layers
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addFullyConnectedLayer(int outputSize);
	bool setActivType(int activationType);
	void setAutoActivLayer(bool isAuto);
	void printLayerDims() const;

	bool finalize();
	std::string getErrorLog() const;

	//functions dealing with data
		//training
		bool addTrainingData(const std::vector<imVector>& trainingData, const std::vector<double>& trueVals);
		bool setTrainingData(const std::vector<imVector>& trainingData, const std::vector<double>& trueVals);
		void clearTrainingData();
		bool addTestData(const std::vector<imVector>& testData, const std::vector<double>& trueVals);
		bool setTestData(const std::vector<imVector>& testData, const std::vector<double>& trueVals);
		void clearTestData();
		bool setTrainingType(int type);
		void setHorizontalReflections(bool use);
        void printTrainingDistribution() const;
        void printTestDistribution() const;

		//running
		void addData(const std::vector<imVector>& data);
		void setData(const std::vector<imVector>& data);
		void clearData();

	int getNumClasses() const;

	//sets for hyperparameters
	bool set_learningRate(double rate);
	bool set_RELU_CAP(double cap);
	bool set_LEAKY_RELU_CONST(double lconst);
	bool set_l2Lambda(double lambda);
	bool set_MOMENT_CONST(double mconst);
	bool set_MAX_NORM_CAP(double cap);

	//running
	void run(bool useGPU=true);
	void getCalculatedClasses(std::vector<int>& dest) const;
	void getConfidences(std::vector<std::vector<double> >& confidences) const;

	//training
	void train(int epochs=-1);
	void setMomentum(bool useMomentum);

	//OpenCL functions
	int getDevice() const;
	bool setDevice(unsigned int device);
	void setGPU(bool useGPU);
	void setConstantMem(bool useConstantMem);
    
    //save
    bool save(const char* filename);

private:	// functions
	//inits
	void init(int inputWidth, int inputHeight, int inputDepth);
	void initOpenCL();

	//functions dealing with layers
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, std::string weightsAndBias);
	void pushBackLayerSize(int width, int height, int depth);

	//weights and biases
	void initRandomWeights(ConvLayer* conv);
	void initWeights(ConvLayer* conv, const std::string& weights);

	//functions dealing with data
	int getTrueValIndex(double trueVal);
	int getMaxElementIndex(const std::vector<double>& vect) const;
	void preprocessData();
	void preprocessTestDataIndividual();
    void preprocessTrainingDataIndividual();
	void preprocessTrainingDataCollective();

	//training
	void setupLayerNeeds(std::vector<cl_mem>& layerNeeds);
	void getTrainingData(std::vector<std::vector<double>* >& trainingData, std::vector<double>& trueVals);
	void initVelocities(std::vector<cl_mem>& velocities);
	void pullCLWeights();
	void shuffleTrainingData(std::vector<std::vector<double>* >& trainingData, std::vector<double>& trueVals, int times = 1);
	void shuffleData(std::vector<std::vector<double>* >& trainingData, int times = 1);

	//load
	bool load(const char* filename);

	//OpenCL functions
	void CheckError(cl_int error);
	std::string LoadKernel(const char* name);
	cl_program CreateProgram(std::string source, cl_context& context, int programNum = -1);
};

#endif /* defined(____ConvNetCL__) */