//
//  ConvNetCL.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNetCL__
#define ____ConvNetCL__

#include "ConvNetCommon.h"

#include <cfloat>
#include <string>
#include <vector>
#include <thread>
#include <cassert>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//OpenCL
#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

// #define NET_SHOW_ERRORS 0

//defines for preprocessing
#define __PREPROCESS_INDIVIDUAL 0
#define __PREPROCESS_COLLECTIVE 1
#define __PREPROCESS_BATCH_NORM 2

//defines for layers
#define ABSTRACT_LAYER -1
#define CONV_LAYER 0
#define MAX_POOL_LAYER 1
#define ACTIV_LAYER 2
#define BATCH_NORM_LAYER 3
#define AVG_POOL_LAYER 4

//defines for ActivTypes
#define RELU 0
#define LEAKY_RELU 1
#define MAX_ACTIV 2

//defines for training types
#define TRAIN_AS_IS 0
#define TRAIN_EQUAL_PROP 1
#define TRAIN_RATIO 2 // params needed 

//defines for programs
#define TRAINING_PROGRAM 0
#define RUNNING_PROGRAM 1

//defines for DE
#define DE_RAND 0
#define DE_BEST 1
#define DE_CURRENT_TO_BEST 2
#define DE_QUIN_AND_SUGANTHAN 3

#define DE_BINOMIAL_CROSSOVER 0
#define DE_EXPONENTIAL_CROSSOVER 1

//defines for ant
//init types
#define ANT_INIT_FANT 0
#define ANT_INIT_MMAS 1

//pheromone update types
#define ANT_UPDATE_SIMPLE 0
#define ANT_UPDATE_BEST 1
#define ANT_UPDATE_FANT 2
#define ANT_UPDATE_ACS 3

//pheromone leak types
#define ANT_LEAK_NONE 0
#define ANT_LEAK_LINEAR_DECREASE 1
#define ANT_LEAK_EXPONENTIAL_DECREASE 2

//macro
#define CheckError(error) {if((error) != CL_SUCCESS){printf("OpenCL call failed with error %d\n",(error)); assert(0);}}

class Net{
public:     // structs
	// struct ClassInfo{
	// 	std::string name = "";
	// 	int trueVal = -1;
	// };
private: 	// structs
	struct Layer{
		int layerType = ABSTRACT_LAYER;
	};

	struct ConvLayer : Layer{
		// int layerType = CONV_LAYER;
		double* weights = nullptr;
		double* biases = nullptr;
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

		ConvLayer& operator=(const ConvLayer& other);
		bool equals(const ConvLayer& other);
	};

	struct BatchNormLayer : Layer{
		std::vector<double> gamma;
		std::vector<double> beta;
		std::vector<double> e;
		std::vector<double> var;
		bool byFeatureMap = true; //if false, by activation
	};

	struct MaxPoolLayer : Layer{
		int stride;
		int poolSize;
	};

	struct AvgPoolLayer : Layer{
		int stride;
		int poolSize;
	};

	struct ActivLayer : Layer{
		int activationType;
	};

	struct WeightHolder{
		double trainAccuracy = 0;
		double testAccuracy = 0;
		double trainError = DBL_MAX;
		std::vector<double*> weights; //weights[convLayer][weight]
		std::vector<double*> biases;

		~WeightHolder();
		void clearWeights();
	};

	struct Kernels{
		~Kernels();
		bool built = false;
		cl_program CNForward, CNTraining;
		cl_kernel reluKernelF;
		cl_kernel leakyReluKernelF;
		cl_kernel convKernelF;
		cl_kernel convKernelFC;
		cl_kernel zeroPadKernelF;
		cl_kernel maxPoolKernelF;
		cl_kernel softmaxKernelF;
		cl_kernel reluKernel;
		cl_kernel reluBackKernel;
		cl_kernel leakyReluKernel;
		cl_kernel leakyReluBackKernel;
		cl_kernel convKernel;
		cl_kernel convBackNeuronsKernel;
		cl_kernel convBackBiasesKernel;
		cl_kernel convBackWeightsKernel;
		cl_kernel convBackWeightsMomentKernel;
		cl_kernel zeroPadKernel;
		cl_kernel zeroPadBackKernel;
		cl_kernel maxPoolKernel;
		cl_kernel maxPoolBackKernel;
		cl_kernel softmaxKernel;
		cl_kernel softmaxBackKernel;
		cl_kernel copyArrayKernel;
		cl_kernel maxSubtractionKernel;
		cl_kernel vectorESumKernel;
		cl_kernel plusEqualsKernel;
		cl_kernel divideEqualsKernel;
		cl_kernel zeroMemKernel, convBackWeightsNoUpdateAccumKernel, convBackBiasesNoUpdateAccumKernel, 
			updateWeightsKernel, updateWeightsMomentKernel, updateBiasesKernel, batchNormKernel, batchNormBackKernel, 
			updateGammaAndBetaKernel, batchNormRunKernel;
	};

	class BNRunMems{
	public:
		BNRunMems();
		BNRunMems(int numThreads, Net* parent);
		~BNRunMems();
		int getNumThreads() const;
		void load(int numThreads, Net* parent);
		std::vector<cl_mem> p, n;
		std::vector<cl_mem*> prevNeurons, neurons;
		std::vector<Kernels> kernels;
		std::vector<cl_command_queue> queues;
		std::vector<cl_mem> denoms;
	private:
		int numThreads = -1;
		Net* parent = nullptr;
		void destroy();
	};

	static int check_counter(int count);

	//from https://www.daniweb.com/programming/software-development/threads/498822/c-11-thread-equivalent-of-pthread-barrier
	class spinlock_barrier
	{
	public:
	  spinlock_barrier(const spinlock_barrier&) = delete;
	  spinlock_barrier& operator=(const spinlock_barrier&) = delete;

	  explicit spinlock_barrier(unsigned int count) :
	    m_count(Net::check_counter(count)), m_generation(0), 
	    m_count_reset_value(count)
	  {
	  }

	  void count_down_and_wait()
	  {
	  	mut.lock();
	    unsigned int gen = m_generation.load();
	    if (--m_count == 0)
	    {
	      if (m_generation.compare_exchange_weak(gen, gen + 1))
	      {
	        m_count = m_count_reset_value;
	      }
	      mut.unlock();
	      return;
	    }
	    mut.unlock();

	    bool cont = true;
	    while (cont)
	    {
	    	mut.lock();
	    	if((gen == m_generation) && (m_count != 0))
	     		std::this_thread::yield();
	     	else
	     		cont = false;
	     	mut.unlock();
	     }

	    // while ((gen == m_generation) && (m_count != 0))
	    //   std::this_thread::yield();
	  }

	private:
	  std::atomic<unsigned int> m_count;
	  std::atomic<unsigned int> m_generation;
	  std::mutex mut;
	  unsigned int m_count_reset_value;
	};


private: 	// members
	bool __inited = false;
	//hyperparameters
	double __learningRate = 1e-3;
	double __RELU_CAP = 5000.0;
	double __LEAKY_RELU_CONST = 0.01;
	double __l2Lambda = 0.01;
	double __MOMENT_CONST = 0.9;
	double __MAX_NORM_CAP = 50.0;

	bool __programs_already_created = false;
	
	//members dealing with layers
	std::vector<Layer*> __layers;  //[0] is input layer
	std::vector<int> __neuronSizes; //[0] is input layer
	std::vector<std::vector<int> > __neuronDims;  //[0] is input layer
	bool __autoActivLayer = true;
	int __maxNeuronSize;
	int __defaultActivType = RELU;
	int __maxWeightSize = 0;

	bool usesSoftmax = true;
	bool isApproximator = false;

	bool __isFinalized = false;
	std::string __errorLog;

	//data and related members
	int __numClasses = 0;
	// std::vector<ClassInfo> __classes;
		//training
		bool __trainingDataPreprocessed = false;
		bool __testDataPreprocessed = false;
		// bool __preprocessIndividual = false;
		int __preprocessType = __PREPROCESS_COLLECTIVE;
		double __mean = 0;
		double __stddev = 0;
		unsigned long __trainingSize = 0;
		bool __isTraining = false;
		std::vector<std::vector<std::vector<double>* > > __trainingData; // class<list of<pointers_to_flattenedImages> >
		std::vector<std::string> __trueNames; // parallel vector of class names for __trainingData
		std::vector<std::vector<double> > __testData;
		std::vector<double> __testTrueIndexes; // parallel vector to __testData that has the trueVal indexes for the data
		bool __useMomentum = true;
		int __trainingType = TRAIN_AS_IS;
		unsigned int __smallestClassSize = -1;
		unsigned int __smallestClassIndex = -1;
		std::string __saveName;
		bool __saveNet = false;
		//running
		bool __dataPreprocessed = false;
		std::vector<std::vector<double> > __data; // list of<flattened images>
		std::vector<std::vector<double> > *__dataPointer;
		std::vector<std::vector<double> > __confidences; // image<list of confidences for each class<confidence> > 

		std::vector<int> __trainRatioAmounts;
		std::vector<int> __trainActualAmounts;

		//batch norm
		// std::atomic<int> mu_reset_done = false;
		std::mutex mtx, gw_mtx, gb_mtx; //mutex, gradient_weights_mutex, gradient_biases_mutex;
		std::vector<std::mutex> mtx_a;
		// int thread_count = 0;
		// bool mu_reset_done = false;

		std::vector<std::vector<std::vector<std::vector<double> > > > bn_x, bn_xhat;

		std::vector<std::vector<double> > mu, delta_mu;
		std::vector<std::vector<double> > sigma_squared, delta_sigma2; //is delta_sigma_squared
		std::vector<cl_mem> mu_cl, delta_mu_cl; // one cl_mem per BNLayer
		std::vector<cl_mem> sigma_squared_cl, delta_sigma2_cl;


		std::vector<cl_mem> gamma; // size of numBatchNormLayers. [numBNLayer] size of cl_mem differs depending on layer
		std::vector<cl_mem> beta;  // size of numBatchNormLayers. [numBNLayer] size of cl_mem differs depending on layer
		std::vector<std::vector<double> > delta_gamma;
		std::vector<std::vector<double> > delta_beta;
		std::vector<cl_mem> delta_gamma_cl; // size of numBatchNormLayers. [numBNLayer] size of cl_mem differs depending on layer
		std::vector<cl_mem> delta_beta_cl;  // size of numBatchNormLayers. [numBNLayer] size of cl_mem differs depending on layer
		std::mutex bnNumCorrect_mtx;
		std::vector<int> bnClassCorrect, bnClassTotal;
		int bnNumCorrect, bnNumZeros = 0;
		double moveAlpha = 0.1;
		bool setupBatchNormCLMems_running_done = false;
		bool setupBatchNormCLMems_done = false;
		double bnTotalError;
		BNRunMems bnrunmems;

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
	std::string CNForwardPath, CNTrainingPath;
	//running kernels
	cl_kernel reluKernelF, leakyReluKernelF, convKernelF, convKernelFC, maxPoolKernelF, softmaxKernelF, zeroPadKernelF;
	//training kernels
	cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel, reluBackKernel,
		zeroPadBackKernel, softmaxBackKernel, maxPoolBackKernel, leakyReluBackKernel, convBackNeuronsKernel, 
		convBackBiasesKernel, convBackWeightsKernel, copyArrayKernel, convBackWeightsMomentKernel,
		maxSubtractionKernel, vectorESumKernel, plusEqualsKernel, divideEqualsKernel,
		zeroMemKernel, convBackWeightsNoUpdateAccumKernel, convBackBiasesNoUpdateAccumKernel, updateWeightsKernel, updateWeightsMomentKernel, updateBiasesKernel,
		batchNormRunKernel, batchNormKernel, batchNormBackKernel, updateGammaAndBetaKernel;

	cl_command_queue queue;
	std::vector<cl_mem> clWeights;
	std::vector<cl_mem> clBiases;
	cl_mem n, p, *neurons, *prevNeurons, denom;


	//DE
	int __targetSelectionMethod = DE_BEST;

	//delete r1v,r2v
	// std::vector<int> r1v, r2v;


	std::mutex error_mtx, __program_creation_mutex;

	#ifdef NET_SHOW_ERRORS
	std::mutex show_error_mtx;
	#endif

public: 	// functions
	//Constructors and Destructors
	Net();
	Net(const Net& other);
	Net(const char* filename);
	Net(int inputWidth, int inputHeight, int inputDepth);
	void init(int inputWidth, int inputHeight, int inputDepth);
	~Net();

	//Equals
	Net& operator=(const Net& other);
	
	//functions dealing with layers and sizes
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addBatchNormLayer(bool byFeatureMap = true);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addAvgPoolLayer(int poolSize, int stride);
	bool addFullyConnectedLayer(int outputSize);
	bool setActivType(int activationType);
	void setAutoActivLayer(bool isAuto);
	void setSaveName(const char *saveName);
	void setSaveName(std::string saveName);
	void printLayerDims() const;
	int getInputWidth() const;
	int getInputHeight() const;
	unsigned int getTotalWeights() const;
	unsigned int getTotalBiases() const;

	bool finalize();
	std::string getErrorLog() const;

	//functions dealing with data
		//training
		bool addTrainingData(const std::vector<convnet::imVector>& trainingData, const std::vector<std::string>& trueNames);
		bool addTrainingData(const std::vector<cv::Mat>& trainingData, const std::vector<std::string>& trueNames, bool rgb=false);
        //bool setTrainingDataShallow(double** images, double* trueVals, unsigned long numImages);
		bool setTrainingData(const std::vector<convnet::imVector>& trainingData, const std::vector<std::string>& trueNames);
		bool setTrainingData(const std::vector<cv::Mat>& trainingData, const std::vector<std::string>& trueNames, bool rgb=false);
		void clearTrainingData();
		bool addTestData(const std::vector<convnet::imVector>& testData, const std::vector<std::string>& trueNames);
		bool addTestData(const std::vector<cv::Mat>& testData, const std::vector<std::string>& trueNames, bool rgb=false);
        //bool setTestDataShallow(double** images, double* trueVals,
		bool setTestData(const std::vector<convnet::imVector>& testData, const std::vector<std::string>& trueNames);
		bool setTestData(const std::vector<cv::Mat>& testData, const std::vector<std::string>& trueNames, bool rgb=false);
		void clearTestData();
		bool setTrainingType(int type, const std::vector<std::string>& params = std::vector<std::string>());
        void printTrainingDistribution() const;
        void printTestDistribution() const;

		//running
		void addData(const std::vector<convnet::imVector>& data);
		void addData(const std::vector<cv::Mat>& data, bool rgb=false);
		void setData(const std::vector<convnet::imVector>& data);
		void setData(const std::vector<cv::Mat>& data, bool rgb=false);
		void clearData();

	int getNumClasses() const;
	// void setClassNames(std::vector<std::string> names, std::vector<int> trueVals);
	void getClassNames(std::vector<std::string>& names) const;
	std::string getNameForIndex(int index) const;

	//sets for hyperparameters
	bool set_learningRate(double rate);
	bool set_RELU_CAP(double cap);
	bool set_LEAKY_RELU_CONST(double lconst);
	bool set_l2Lambda(double lambda);
	bool set_MOMENT_CONST(double mconst);
	bool set_MAX_NORM_CAP(double cap);
	void preprocessIndividually();
	void preprocessCollectively();

	void setTrueNameIndex(const std::string& name, int index);


	int getIndexFromName(const std::string& name) const;

	//running
	void run();
	void run_parallel();
	void getCalculatedClasses(std::vector<int>& dest) const;
	void getConfidences(std::vector<std::vector<double> >& confidences) const;

	//training
	void train(int epochs=-1);
	void miniBatchTrain(int batchSize, int epochs=-1);
	void batchNormTrain(int batchSize, int epochs=-1);
	void DETrain(int generations, int population = 25, double mutationScale = 0.5, int crossMethod = DE_EXPONENTIAL_CROSSOVER, double crossProb = 0.1, bool BP = true);
	void DETrain_sameSize(int mutationType, int generations, int dataBatchSize, int population = 15, double mutationScale = 0.5, int crossMethod = DE_BINOMIAL_CROSSOVER, double crossProb = 0.8, bool BP = true);
	bool setDETargetSelectionMethod(int method);
	void setMomentum(bool useMomentum);
	void antTrain(unsigned int maxIterations, unsigned int population, int dataBatchSize);

	//OpenCL functions
	int getDevice() const;
	bool setDevice(unsigned int device);
	void setDevice(cl_device_id device, cl_platform_id platform);
	void setGPU(bool useGPU);
	void setConstantMem(bool useConstantMem);
    
    //save and load
    bool save(const char* filename);
	bool load(const char* filename);

private:	// functions
	//functions for operator=
	void copyLayers(const Net& other);
	void destroy();

	//inits
	void initOpenCL();

	//functions dealing with layers
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, const std::string& weightsAndBias);
	bool addBatchNormLayer(bool byFeatureMap, int gamma_size, const std::string& gamma, const std::string& beta, const std::string& e, const std::string& var);
	void pushBackLayerSize(int width, int height, int depth);

	//weights and biases
	void initRandomWeights(ConvLayer* conv, int prevDepth);
	void initWeights(ConvLayer* conv, const std::string& weights);

	//functions dealing with data
	int getTrueValIndex(const std::string& trueVal, bool allowAppends = true);
	int getMaxElementIndex(const std::vector<double>& vect) const;
	int getMaxElementIndex(const std::vector<int>& vect) const;
	void preprocessDataIndividual();
	void preprocessDataCollective();
	void preprocessTestDataIndividual();
    void preprocessTrainingDataIndividual();
    void preprocessTestDataCollective();
	void preprocessTrainingDataCollective();

	//training
	void setupLayerNeeds(std::vector<cl_mem>& layerNeeds);
	void destroyVectorCLMems(std::vector<cl_mem>& vect);
	void getTrainingData(std::vector<std::vector<double>* >& trainingData, std::vector<double>& trueVals);
	void initVelocities(std::vector<cl_mem>& velocities);
	void pullCLWeights();
	void pullCLWeights(Net* net, const std::vector<cl_mem>& clWeights, const cl_command_queue& queue);
	void pushCLWeights(std::vector<Layer*>& layers, const std::vector<cl_mem>& clWeights, const std::vector<cl_mem>& clBiases, const cl_command_queue& queue, cl_bool block);
	void shuffleTrainingData(std::vector<std::vector<double>* >& trainingData, std::vector<double>& trueVals, int times = 1);
	void shuffleData(std::vector<std::vector<double>* >& trainingData, int times = 1);
	void trainSetup();
	void trainSetup(std::vector<cl_mem>& layerNeeds, std::vector<cl_mem>& velocities);
	void feedForward(std::vector<cl_mem>& layerNeeds);
	void feedForward(cl_mem** prevNeurons, cl_mem** neurons, std::vector<std::vector<int> >& __neuronDims, std::vector<Layer*>& __layers,
 		const std::vector<cl_mem>& layerNeeds, const std::vector<cl_mem>& clWeights, const std::vector<cl_mem>& clBiases, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k);
	void feedForward_running(cl_mem** prevNeurons, cl_mem** neurons, std::vector<std::vector<int> >& __neuronDims, std::vector<Layer*>& __layers,
		const std::vector<cl_mem>& clWeights, const std::vector<cl_mem>& clBiases, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k);
	void softmaxForward();
	void softmaxForward(cl_mem* prevNeurons, cl_mem* neurons, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k);
	void softmaxBackprop(int curTrueVal);
	void softmaxBackprop(int curTrueVal, cl_mem** prevNeurons, cl_mem** neurons, const cl_command_queue& queue, const Kernels& k, Net* net);
	void backprop(std::vector<cl_mem>& layerNeeds, std::vector<cl_mem>& velocities);
	void backprop(int curTrueVal, cl_mem** prevNeurons, cl_mem** neurons, Net* net, const std::vector<cl_mem>& layerNeeds, const std::vector<cl_mem>& velocities, 
		const std::vector<cl_mem>& clWeights, const std::vector<cl_mem>& clBiases, const cl_command_queue queue, const Kernels& k);
	void storeWeightsInHolder(WeightHolder& holder);
	void loadWeightsFromHolder(WeightHolder& holder);
	bool stringToDims(std::string str, int* dims);


	//de train
	void setupRandomNets(std::vector<Net*>& nets);
	void setupEquivalentNets(std::vector<Net*>& nets);
	void releaseCLMem();
	double getFitness(std::vector<double>& prediction, double trueVal, Net* net);
	int getTargetVector(int method, const std::vector<double>& fits, int curNet);
	void getHelperVectors(const std::vector<Net*>& nets, int target, int curNet, std::vector<Net*>& helpers);
	Net* makeDonor(const std::vector<Net*>& helpers, double scaleFactor, bool shallow = false);
	Net* makeDonor(int mutType, const std::vector<Net*>& nets, const std::vector<double>& netfit, int curIndex, int n, double scaleFactor);
	inline int POSITION(int filter, int x, int y, int z, int filsize, int prevdepth);
	Net* crossover(Net* parent, Net* donor, int method, double prob);
	int mapConvLayer(Net* orig, int layerNum, Net* dest);
	void mapPosIndexes(ConvLayer* origConv, int* origpos, ConvLayer* destConv, int* destpos);
	// void DE_mutation_crossover_selection(int netNum, );
	void buildKernels(Kernels& k, int device);
	// void releaseKernels(Kernels&k);
	void trial_thread(int netIndex, std::vector<Net*>* nets, double netfit, Net* trial, double* trainDataPtr, int curTrueVal, cl_mem** prevNeurons, 
		cl_mem** neurons, const std::vector<cl_mem>& layerNeeds, const std::vector<cl_mem>& clWeights, const std::vector<cl_mem>& clBiases, 
		const std::vector<cl_mem>& velocities, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k, bool BP);

	//OpenCL functions
	// void CheckError(cl_int error);
	std::string LoadKernel(const char* name);
	cl_program CreateProgram(std::string source, cl_context& context, int programNum = -1);

	//minibatch training
	void zeroMem(std::vector<cl_mem>& mem, const std::vector<size_t>& sizes);
	void backprop_noUpdate(std::vector<cl_mem>& layerNeeds, std::vector<cl_mem>& gradients_weights, std::vector<cl_mem>& gradients_biases);
	void updateWeights(std::vector<cl_mem>& gradients_weights, std::vector<cl_mem>& gradients_biases, std::vector<cl_mem>& velocities);

	//batchnorm training
	void feedForward_BN(const int num_threads, const int minibatch_size, const int thread_num, const std::vector<std::vector<double>* >& trainingData, const std::vector<double>& trueVals, int start, int end, std::vector<cl_mem*>* prevNeurons, std::vector<cl_mem*>* neurons,//cl_mem** prevNeurons, cl_mem** neurons,
		const std::vector<std::vector<cl_mem> >& layerNeeds, const cl_command_queue& queue, const cl_mem& denom, const Kernels& k, spinlock_barrier* barrier);
	void backprop_noUpdate_BN(const int num_threads, const int minibatch_size, const int thread_num, const int start, const int amount, const std::vector<double>& trueVals, std::vector<cl_mem*> *prevNeurons, std::vector<cl_mem*> *neurons,
		const std::vector<std::vector<cl_mem> > &layerNeeds, const cl_command_queue& queue, const Kernels &k, spinlock_barrier* barrier,
		const std::vector<cl_mem>& gradients_weights, const std::vector<cl_mem>& gradients_biases, const std::vector<cl_mem>& bn_x_cl);
	void setupBatchNormCLMems(int num_threads, const std::vector<int>& thread_sizes, std::vector<std::vector<cl_mem> > &bn_x_cl);
	void setupBatchNormCLMems_running(int num_threads, const std::vector<int>& thread_sizes);
	void pullGammaAndBeta();
	void updateGammaAndBeta();
	int getNumBatchNormLayers();
	void batchNormRun();
	void feedForward_BN_running(const int num_threads, const int minibatch_size, const int thread_num, int start, int end, std::vector<std::vector<double> >* __dataPointer, cl_mem** prevNeurons, cl_mem** neurons, 
	 const cl_command_queue& queue, const cl_mem& denom, const Kernels& k);
	void destroyBatchNormCLMems();


	int getMaxNeuronSize() const;
	void convertGreyscale(std::vector<double>& image);
	void makeTrainingGreyscale();

};

#endif /* defined(____ConvNetCL__) */
