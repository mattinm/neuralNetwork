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

// Classes

class Layer{
public:
	virtual ~Layer(){};
	virtual int getType() const = 0;
	virtual void forwardprop(const Layer& prevLayer) = 0;
	virtual void backprop(Layer& prevLayer) = 0;
	virtual const std::vector<std::vector<std::vector<double> > >& getNeurons() const = 0;
	virtual std::vector<std::vector<std::vector<double> > >& getdNeurons() = 0;
};

class InputLayer : public Layer{
public:
	InputLayer();
	~InputLayer();
	int getType() const;
	InputLayer(const std::vector<std::vector<std::vector<double> > >& trainingImage, std::vector<std::vector<std::vector<double> > >* blankdNeurons);
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
	bool setImage(const std::vector<std::vector<std::vector<double> > >* trainingImage,std::vector<std::vector<std::vector<double> > >* blankdNeurons);
private:
	static const int i_type;
	bool i_resizeable;
	const std::vector<std::vector<std::vector<double> > >  *i_neurons;
	std::vector<std::vector<std::vector<double> > > *i_dneurons;
};

class ConvLayer : public Layer{
public:
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	~ConvLayer();
	int getType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
	void initWeights();
private:
	void init(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	void initRandomWeights();
	void initWeights(std::string weights);
	static const int c_type;
	std::vector<std::vector<std::vector<double> > > c_neurons;
	std::vector<std::vector<std::vector<double> > > c_dneurons;
	std::vector<std::vector<std::vector<std::vector<double> > > > c_weights;
	std::vector<std::vector<std::vector<std::vector<double> > > > c_dweights;
	std::vector<double> c_biases;
	std::vector<double> c_dbiases;
	int c_padding;
	int c_stride;
};

class MaxPoolLayer : public Layer{
public:
	MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride);
	~MaxPoolLayer();
	int getType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int m_type;
	std::vector<std::vector<std::vector<double> > > m_neurons;
	std::vector<std::vector<std::vector<double> > > m_dneurons;
	int m_stride;
	int m_poolSize;
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
	int getActivationType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int a_type;
	static const double RELU_CAP;
	static const double LEAKY_RELU_CONST;
	int a_activationType;
	std::vector<std::vector<std::vector<double> > > a_neurons;
	std::vector<std::vector<std::vector<double> > > a_dneurons;
};

class SoftmaxLayer : public Layer{
public:
	SoftmaxLayer(const Layer& prevLayer);
	~SoftmaxLayer();
	int getPredictedClass();
	std::vector<double> getError();
	void setError(std::vector<double> error);
	int getType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	void setTrueVal(int trueVal);
	void gradientCheck(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int s_type;
	std::vector<double> s_neurons;
	std::vector<std::vector<std::vector<double> > > s_3neurons;
	std::vector<double> s_dneurons;
	std::vector<std::vector<std::vector<double> > > s_3dneurons;
	int s_trueVal;
};

class Net{
public:
	Net(const char* filename);
	Net(int inputWidth, int inputHeight, int inputDepth);
	void init(int, int, int);
	~Net();
	void forwardprop();
	void backprop();
	static double stepSize;
	static const int CONV_LAYER = 0;
	static const int MAX_POOL_LAYER = 1;
	static const int ACTIV_LAYER = 2;
	static const int INPUT_LAYER = 3;
	static const int SOFTMAX_LAYER = 4;
	static const bool walkthrough = false;
	static const bool showErrors = false;
	static bool gradCheck;
	static const double GRADCHECK_H;

	void debug();

	int numLayers();
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addSoftmaxLayer();
	void addTrainingData(const std::vector<std::vector<std::vector<std::vector<double> > > >& trainingData, std::vector<double>& trueVals);
	void clear();
	bool setActivType(int activationType);
	void train(int epochs);
	void splitTrain(int epochs);
	void runTrainingData();
	void miniBatchTrain(int epochs, int batchSize);
	void addRealData(const std::vector<std::vector<std::vector<std::vector<double> > > >& realData);
	void run();
	std::vector<std::vector<std::vector<double> > >* getBlankVectorPointer();
	int getPredictedClass();
	void gradientCheck();
	double calcLoss(int indexOfTrueVal);
	void shuffleTrainingData(int times=1);

	bool save(const char* filename);
private:
	static int n_activationType;
	std::vector<Layer*> n_trainingData;
	std::vector<Layer*> n_realData;
	std::vector<double> n_results;
	std::vector<double> n_trainingDataTrueVals;
	std::vector<std::vector<std::vector<double> > > n_blankVector;
	InputLayer n_blankInput;
	std::vector<Layer*> n_layers;

	bool load(const char* filename);
};


// Other Functions

void padZeros(const std::vector<std::vector<std::vector<double> > > &source, int numZeros, std::vector<std::vector<std::vector<double> > > &dest);

void printVector(const std::vector<std::vector<std::vector<std::vector<double> > > > &vect);

void printVector(const std::vector<std::vector<std::vector<double> > > &vect);

void printVector(const std::vector<double>& vect);

void resize3DVector(std::vector<std::vector<std::vector<double> > > &vect, int width, int height, int depth);

void setAll3DVector(std::vector<std::vector<std::vector<double> > > &vect, double val);

void setAll4DVector(std::vector<std::vector<std::vector<std::vector<double> > > > &vect, double val);

void softmax(const std::vector<std::vector<std::vector<double> > > &vect, std::vector<double>& normedPredictionsContainer);

void maxSubtraction(std::vector<double>& vect);

void meanSubtraction(std::vector<double>& vect);

double mean(const std::vector<std::vector<std::vector<double> > >& vect);

double stddev(const std::vector<std::vector<std::vector<double> > >& vect);

double stddev(const std::vector<std::vector<std::vector<double> > >& vect, double mean);

void preprocess(std::vector<std::vector<std::vector<double> > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<double> > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

void compressImage(std::vector<std::vector<std::vector<std::vector<double> > > >& vect, double newMin, double newMax);

void compressImage(std::vector<std::vector<std::vector<double> > >& vect, double newMin, double newMax);

double vectorESum(const std::vector<double> &source);

void vectorClone(const std::vector<std::vector<std::vector<double> > > &source, std::vector<std::vector<std::vector<double> > > &dest);





#endif /* defined(____ConvNet__) */
