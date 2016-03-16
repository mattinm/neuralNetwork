//
//  ConvNet.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNet__
#define ____ConvNet__

#include <stdio.h>
#include <vector>

//using namespace std;

// Classes

class Layer{
public:
	virtual ~Layer(){};
	virtual void forwardprop(const Layer& prevLayer) = 0;
	virtual void backprop(Layer& prevLayer) = 0;
	virtual const std::vector<std::vector<std::vector<double> > >& getNeurons() const = 0;
	virtual std::vector<std::vector<std::vector<double> > >& getdNeurons() = 0;
};

class InputLayer : public Layer{
public:
	InputLayer();
	~InputLayer();
	InputLayer(const std::vector<std::vector<std::vector<double> > >& trainingImage);
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
	bool setImage(const std::vector<std::vector<std::vector<double> > >& trainingImage);
private:
	bool i_resizeable;
	const std::vector<std::vector<std::vector<double> > >* i_neurons;
	std::vector<std::vector<std::vector<double> > > i_dneurons;
};

class ConvLayer : public Layer{
public:
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	~ConvLayer();
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
	void initWeights();
private:
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
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
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
	static const int NUM_ACTIV_TYPES = 1;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	int a_activationType;
	std::vector<std::vector<std::vector<double> > > a_neurons;
	std::vector<std::vector<std::vector<double> > > a_dneurons;
};

class Net{
public:
	Net(int inputWidth, int inputHeight, int inputDepth);
	~Net();
	void forwardprop();
	void backprop();
	static double stepSize;

	int numLayers();
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addMaxPoolLayer(int poolSize, int stride);
	void addTrainingData(const std::vector<std::vector<std::vector<std::vector<double> > > >& trainingData, std::vector<double>& trueVals);
	void clear();
	bool setActivType(int activationType);
	void train(int epochs);
	void addRealData(const std::vector<std::vector<std::vector<std::vector<double> > > >& realData);
	void run();
	int getPredictedClass();
	double calcLoss(int indexOfTrueVal);
private:
	static int n_activationType;
	std::vector<Layer*> n_trainingData;
	std::vector<Layer*> n_realData;
	std::vector<double> n_results;
	std::vector<double> n_trainingDataTrueVals;
	InputLayer n_blankInput;
	std::vector<Layer*> n_layers;
};


// Other Functions

void padZeros(const std::vector<std::vector<std::vector<double> > > &source, int numZeros, std::vector<std::vector<std::vector<double> > > &dest);

void printVector(const std::vector<std::vector<std::vector<std::vector<double> > > > &vect);

void printVector(const std::vector<std::vector<std::vector<double> > > &vect);

void resize3DVector(std::vector<std::vector<std::vector<double> > > &vect, int width, int height, int depth);

void setAll3DVector(std::vector<std::vector<std::vector<double> > > &vect, double val);

void setAll4DVector(std::vector<std::vector<std::vector<std::vector<double> > > > &vect, double val);

void softmax(const std::vector<std::vector<std::vector<double> > > &vect, std::vector<double>& normedPredictionsContainer);

void meanSubtraction(std::vector<double>& vect);

double vectorESum(const std::vector<double> &source);




#endif /* defined(____ConvNet__) */
