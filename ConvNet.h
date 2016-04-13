//
//  ConvNet.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNet__
#define ____ConvNet__

//#include <stdio.h>
#include <vector>

#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif


// Classes

class Layer{
public:
	virtual ~Layer(){};
	virtual int getType() const = 0;
	virtual int getNumNeurons() const = 0;
	virtual unsigned long getMem() const = 0;
	virtual void forwardprop(const Layer& prevLayer) = 0;
	virtual void backprop(Layer& prevLayer) = 0;
	virtual const std::vector<std::vector<std::vector<float> > >& getNeurons() const = 0;
	virtual std::vector<std::vector<std::vector<float> > >& getdNeurons() = 0;
};

class InputLayer : public Layer{
public:
	InputLayer();
	~InputLayer();
	int getType() const;
	int getNumNeurons() const;
	unsigned long getMem() const;
	float* getImage() const;
	void getImage(float* dest, int size) const;
	int getImageSize() const;
	InputLayer(const std::vector<std::vector<std::vector<float> > >& trainingImage, std::vector<std::vector<std::vector<float> > >* blankdNeurons);
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<float> > >& getNeurons() const;
	std::vector<std::vector<std::vector<float> > >& getdNeurons();
	bool setImage(const std::vector<std::vector<std::vector<float> > >* trainingImage,std::vector<std::vector<std::vector<float> > >* blankdNeurons);
private:
	static const int i_type;
	bool i_resizeable;
	int i_numNeurons;
	const std::vector<std::vector<std::vector<float> > >  *i_neurons;
	std::vector<std::vector<std::vector<float> > > *i_dneurons;
};

class ConvLayer : public Layer{
public:
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	~ConvLayer();
	int getType() const;
	int getNumNeurons() const;
	unsigned long getMemWeightsAndBiases() const;
	unsigned long getMem() const;
	int getMaxSizeNeeded() const;
	float* getWeights() const;
	int getNumWeights() const;
	float* getBiases() const;
	int getNumBiases() const;
	std::vector<int> getKernelHyperParameters() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<float> > >& getNeurons() const;
	std::vector<std::vector<std::vector<float> > >& getdNeurons();
private:
	void _putWeights(float* weights, int vectIndex) const;
	void init(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	void initRandomWeights();
	void initWeights(std::string weights);
	static const int c_type;
	std::vector<std::vector<std::vector<float> > > c_neurons;
	std::vector<std::vector<std::vector<float> > > c_dneurons;
	std::vector<std::vector<std::vector<std::vector<float> > > > c_weights;
	std::vector<std::vector<std::vector<std::vector<float> > > > c_dweights;
	int c_numWeights;
	int c_numBiases;
	int c_numNeurons;
	std::vector<float> c_biases;
	std::vector<float> c_dbiases;
	int c_padding;
	int c_stride;
	int c_prevNeuronWidth;
	int c_prevNeuronHeight;
	int c_prevNeuronDepth;
	int c_maxSizeNeeded;
};

class MaxPoolLayer : public Layer{
public:
	MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride);
	~MaxPoolLayer();
	int getType() const;
	int getNumNeurons() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	unsigned long getMem() const;
	std::vector<int> getKernelHyperParameters() const;
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<float> > >& getNeurons() const;
	std::vector<std::vector<std::vector<float> > >& getdNeurons();
private:
	static const int m_type;
	std::vector<std::vector<std::vector<float> > > m_neurons;
	std::vector<std::vector<std::vector<float> > > m_dneurons;
	int m_stride;
	int m_poolSize;
	int m_numNeurons;
	int m_prevWidth;
	int m_prevDepth;
};

class ActivLayer : public Layer{
public:
	ActivLayer(const Layer& prevLayer, const int activationType);
	~ActivLayer();
	//ACTIV_TYPES (exp. RELU) must start at 0 and go up by one. The check to see if a type is valid is
	// if(0 <= type && type < ACTIV_NUM_TYPES) then Valid
	static const int RELU = 0;
	static const int LEAKY_RELU = 1;
	static const int NUM_ACTIV_TYPES = 2;
	int getType() const;
	unsigned long getMem() const;
	int getNumNeurons() const;
	int getActivationType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<float> > >& getNeurons() const;
	std::vector<std::vector<std::vector<float> > >& getdNeurons();
private:
	static const int a_type;
	static const float RELU_CAP;
	static const float LEAKY_RELU_CONST;
	int a_activationType;
	std::vector<std::vector<std::vector<float> > > a_neurons;
	std::vector<std::vector<std::vector<float> > > a_dneurons;
	int a_numNeurons;
};

class SoftmaxLayer : public Layer{
public:
	SoftmaxLayer(const Layer& prevLayer);
	~SoftmaxLayer();
	int getPredictedClass();
	std::vector<float> getError();
	void setError(std::vector<float> error);
	int getType() const;
	unsigned long getMem() const;
	int getNumNeurons() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	void setTrueVal(int trueVal);
	void gradientCheck(Layer& prevLayer);
	const std::vector<std::vector<std::vector<float> > >& getNeurons() const;
	std::vector<std::vector<std::vector<float> > >& getdNeurons();
private:
	static const int s_type;
	std::vector<float> s_neurons;
	std::vector<std::vector<std::vector<float> > > s_3neurons;
	std::vector<float> s_dneurons;
	std::vector<std::vector<std::vector<float> > > s_3dneurons;
	int s_trueVal;
};

class Net{
public:
	Net(const char* filename);
	Net(int inputWidth, int inputHeight, int inputDepth);
	
	~Net();
	void forwardprop();
	void backprop();
	static float stepSize;
	static const int CONV_LAYER = 0;
	static const int MAX_POOL_LAYER = 1;
	static const int ACTIV_LAYER = 2;
	static const int INPUT_LAYER = 3;
	static const int SOFTMAX_LAYER = 4;
	static const bool walkthrough = false;
	static const bool showErrors = false;
	static bool gradCheck;
	static const float GRADCHECK_H;

	void debug();

	int numLayers();
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addSoftmaxLayer();
	void addTrainingData(const std::vector<std::vector<std::vector<std::vector<float> > > >& trainingData, std::vector<float>& trueVals);
	void clear();
	void newRun(bool useGPU = true);
	bool setActivType(int activationType);
	void train(int epochs);
	void splitTrain(int epochs, bool useGPU = true);
	void runTrainingData();
	void miniBatchTrain(int epochs, int batchSize);
	void addRealData(const std::vector<std::vector<std::vector<std::vector<float> > > >& realData);
	void run(bool useGPU=true);
	std::vector<std::vector<std::vector<float> > >* getBlankVectorPointer();
	int getPredictedClass();
	void gradientCheck();
	float calcLoss(int indexOfTrueVal);
	void shuffleTrainingData(int times=1);
	unsigned long getMem() const;
	unsigned long getMemForward() const;

	bool save(const char* filename);
private:
	static int n_activationType;
	std::vector<InputLayer*> n_trainingData;
	std::vector<float> n_results;
	std::vector<float> n_trainingDataTrueVals;
	std::vector<std::vector<std::vector<float> > > n_blankVector;
	InputLayer n_blankInput;
	std::vector<Layer*> n_layers;
	bool n_training;
	bool n_hasConvLayer, n_hasMaxPoolLayer, n_hasRELULayer, n_hasLeakyRELULayer, n_hasSoftmax;

	bool load(const char* filename);
	void init(int, int, int);
	unsigned long getMaxNeuronSize() const;

};



//OpenCL helper functions
std::string LoadKernel (const char* name);

void CheckError (cl_int error);

cl_program CreateProgram (const std::string& source, cl_context& context);

// Other Functions
void padZeros(const std::vector<std::vector<std::vector<float> > > &source, int numZeros, std::vector<std::vector<std::vector<float> > > &dest);

void printVector(const std::vector<std::vector<std::vector<std::vector<float> > > > &vect);

void printVector(const std::vector<std::vector<std::vector<float> > > &vect);

void printVector(const std::vector<float>& vect);

void printArray(float* array, int size);

void resize3DVector(std::vector<std::vector<std::vector<float> > > &vect, int width, int height, int depth);

void setAll3DVector(std::vector<std::vector<std::vector<float> > > &vect, float val);

void setAll4DVector(std::vector<std::vector<std::vector<std::vector<float> > > > &vect, float val);

void softmax(const std::vector<std::vector<std::vector<float> > > &vect, std::vector<float>& normedPredictionsContainer);

void maxSubtraction(std::vector<float>& vect);

void meanSubtraction(std::vector<float>& vect);

float mean(const std::vector<std::vector<std::vector<float> > >& vect);

float stddev(const std::vector<std::vector<std::vector<float> > >& vect);

float stddev(const std::vector<std::vector<std::vector<float> > >& vect, float mean);

void preprocess(std::vector<std::vector<std::vector<float> > >& vect);

void preprocess(std::vector<std::vector<std::vector<std::vector<float> > > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<float> > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<std::vector<float> > > >& vect);

void compressImage(std::vector<std::vector<std::vector<std::vector<float> > > >& vect, float newMin, float newMax);

void compressImage(std::vector<std::vector<std::vector<float> > >& vect, float newMin, float newMax);

float vectorESum(const std::vector<float> &source);

int getMaxElementIndex(const std::vector<float> &vect); 

void vectorClone(const std::vector<std::vector<std::vector<float> > > &source, std::vector<std::vector<std::vector<float> > > &dest);





#endif /* defined(____ConvNet__) */
