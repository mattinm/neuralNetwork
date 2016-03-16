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
 *			FullyConnectedLayer???
 *
 *	Each Layer needs forwardprop and backprop
 *		Can we build in local dneurons in forwardprop? No?
 *		Multiply(and add) by forward derivatives in backprop
 * 		
 *		each layer finishes the derivatives for the layer before it
 *		layers never need to know what layer comes after??? even in backprop??? I think so.
 *
 *	Todo: make it so the i_dneurons for InputLayer can be added or deleted only when we are doing 
 *		the computation with that InputLayer. This should reduce size of n_trainingData by like 2.
 *
 *
 *************************************************************************************************/

#include "ConvNet.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <limits>

#define GETMAX(x,y) (x > y) ? x: y

using namespace std;

/***********************************
 * Class Implementations
 ***********************************/

/**********************
 * Net
 **********************/

double Net::stepSize = 1e-5;

int Net::n_activationType = 0;

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	vector<vector<vector<double> > > blankVector;
	resize3DVector(blankVector,inputWidth,inputHeight,inputDepth);
	n_blankInput.setImage(blankVector);
	n_layers.push_back(&n_blankInput);
}

Net::~Net()
{
	//do I need to run delete on the vectors in the layers????????
	for(int i=0; i< n_layers.size(); i++)
	{
		delete n_layers[i];
	}

	for(int i=0; i< n_trainingData.size(); i++)
	{
		delete n_trainingData[i];
	}
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

void Net::train(int epochs)
{
	//set 1 in the d_neurons in the last layer
			//or do we need to set it to the error? -> I don't think so.
	vector<vector<vector<double> > > lastLayerGradients = n_layers.back()->getdNeurons();
	setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//get error
			int predictedClass = getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

			//run backward pass
			backprop();

		}
		cout << "Epoch: " << e << ", Accuracy: " << (double)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;
	}
}

void Net::run()
{
	for(int r=0; r< n_realData.size(); r++)
	{
		n_layers[0] = n_realData[r];

		//run forward pass
		forwardprop();

		//get the results and save them into n_results
	}
}

int Net::numLayers() 
{
	return n_layers.size();
}

bool Net::addActivLayer()
{
	try
	{
		ActivLayer *activ = new ActivLayer(*n_layers.back(),n_activationType);
		n_layers.push_back(activ);
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
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addMaxPoolLayer(int poolSize, int stride)
{
	try
	{
		MaxPoolLayer *pool = new MaxPoolLayer(*(n_layers.back()),poolSize, stride);
		n_layers.push_back(pool);
		return true;
	}
	catch(...)
	{
		return false;
	}
}

//The true value will be the index of the correct class
void Net::addTrainingData(const vector<vector<vector<vector<double> > > >& trainingData, vector<double>& trueVals)
{
	const vector<vector<vector<vector<double> > > >& t = trainingData;	
	for(int n=0; n< t.size(); n++)
	{
		InputLayer *in = new InputLayer(t[n]);
		n_trainingData.push_back(in);
		n_trainingDataTrueVals.push_back(trueVals[n]);
	}
}

void Net::addRealData(const vector<vector<vector<vector<double> > > >& realData)
{
	const vector<vector<vector<vector<double> > > >& r = realData;
	n_results.resize(r.size());	
	for(int n=0; n< r.size(); n++)
	{
		InputLayer *in = new InputLayer(r[n]);
		n_realData.push_back(in);
	}
}

void Net::clear()
{
	n_layers.clear();
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

double Net::calcLoss(int indexOfTrueVal)
{
	// call softmax on the last layer of input data. The last layer should go down to a vector that is
	// 1x1xn or 1xnx1 or nx1x1
	vector<double> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions); // gets out predictions
	return -log(normedPredictions[indexOfTrueVal]);

}

int Net::getPredictedClass()
{
	vector<double> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions);
	int maxLoc = 0;
	for(int i=1; i<normedPredictions.size();i++)
	{
		if(normedPredictions[i] > normedPredictions[maxLoc])
		{
			maxLoc = i;
		}
	}
	return maxLoc;
}

/**********************
 * InputLayer
 **********************/

InputLayer::InputLayer()
{
	i_resizeable = true;
}

InputLayer::~InputLayer(){}

InputLayer::InputLayer(const vector<vector<vector<double> > >& trainingImage)
{
	const vector<vector<vector<double> > >& t = trainingImage;
	i_neurons = &trainingImage;
	resize3DVector(i_dneurons,t.size(),t[0].size(),t[0][0].size());
	i_resizeable = false;
}

void InputLayer::forwardprop(const Layer& prevLayer){};
void InputLayer::backprop(Layer& prevLayer){};

const vector<vector<vector<double> > >& InputLayer::getNeurons() const
{
	return *i_neurons;
}

vector<vector<vector<double> > >& InputLayer::getdNeurons()
{
	return i_dneurons;
}

bool InputLayer::setImage(const vector<vector<vector<double> > >& trainingImage)
{
	if(i_resizeable)
	{
		const vector<vector<vector<double> > >& t = trainingImage;
		i_neurons = &trainingImage;
		resize3DVector(i_dneurons,t.size(),t[0].size(),t[0][0].size());
		return true;
	}
	return false;
}

/**********************
 * ConvLayer
 **********************/

ConvLayer::ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
{
	// need to set size of neurons, dneurons, weights, dweights. Also make sure that the sizes and such match
	// up with the amount of padding and stride. If padding is greater than 0 will need to make new 3d vector 
	// to hold padded array (or use if statements?) DO IN FORWARD PROP AND MAKE LOCAL VARIABLE.
	
	const vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
	int prevWidth = prevNeurons.size();
	int prevHeight = prevNeurons[0].size();
	int prevDepth = prevNeurons[0][0].size();

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
	
	//Size of c_weights (and c_dweights) will be numFilters x filterSize x filterSize x numFilters
	c_weights.resize(numFilters);
	c_dweights.resize(numFilters);
	for(int i=0; i< numFilters; i++)
	{
		resize3DVector(c_weights[i], filterSize, filterSize, prevDepth);
		resize3DVector(c_dweights[i], filterSize, filterSize, prevDepth);
	}

	c_biases.resize(numFilters);

	//need some way to initialize, save, and load weights and biases
}

ConvLayer::~ConvLayer(){}

void ConvLayer::forwardprop(const Layer& prevLayer)
{
	vector<vector<vector<double> > > source;
	padZeros(prevLayer.getNeurons(),c_padding,source);
	double sum;
	int oX, oY;
	int subsetWidth = c_weights[0].size();
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i < source.size()-1; i+= c_stride) // row in source
		{
			oY = 0;
			for(int j=0; j < source[i].size()-1;j+=c_stride) // col in source
			{
				//now we go into the stride subset
				sum = 0;
				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < source[i][j].size(); k++) // depth in source
						{
							sum += source[i+s][j+r][k] * c_weights[f][s][r][k];
							//sum += source.at(i+s).at(j+r).at(k) * weights.at(f).at(s).at(r).at(k);

							//can I set some of the prevLayers.getdNeuron local values at this point???
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
}

//todo: make a special forwardprop and backprop for when padding = 0.
//todo: update weight and bias values

void ConvLayer::backprop(Layer& prevLayer)
{
	vector<vector<vector<double> > > p_dNeurons = prevLayer.getdNeurons();
	vector<vector<vector<double> > > padded_dNeurons;
	resize3DVector(padded_dNeurons,p_dNeurons.size() + 2*c_padding,p_dNeurons[0].size() + 2*c_padding, p_dNeurons[0][0].size());
	setAll3DVector(p_dNeurons, 0);
	setAll4DVector(c_dweights, 0);
	vector<vector<vector<double> > > source;
	padZeros(prevLayer.getNeurons(),c_padding,source);
	double sum;
	int oX, oY; // outX & outY
	int subsetWidth = c_weights[0].size();
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i < source.size()-1; i+= c_stride) // row in source
		{
			oY = 0;
			for(int j=0; j < source[i].size()-1;j+=c_stride) // col in source
			{
				//now we go into the stride subset
				sum = 0;
				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < source[i][j].size(); k++) // depth in source
						{
							//sum += source[i+s][j+r][k] * c_weights[f][s][r][k];

							// out[oX][oY][f] <- source[i+s][j+r][k] * weights[f][s][r][k]
							// dout[oX][oY][f] -> source & weights 
							padded_dNeurons[i+s][j+r][k] += c_dneurons[oX][oY][f] * c_weights[f][s][r][k];
							c_dweights[f][s][r][k]  	 += c_dneurons[oX][oY][f] * source[i+s][j+r][k];
						}
					}
				}
				// add bias
				//sum += c_biases[f];
				c_dbiases[f] = c_dneurons[oX][oY++][f];
				// add into c_neurons[i%stride, j%stride, f]
				//oY++;
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

	//we need to substract the gradient * the stepSize from the weights and biases

	//update the weights
	for(int f=0;f<c_weights.size();f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
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

const vector<vector<vector<double> > >& ConvLayer::getNeurons() const
{
	return c_neurons;
}

vector<vector<vector<double> > >& ConvLayer::getdNeurons()
{
	return c_dneurons;
}

/**********************
 * MaxPoolLayer
 **********************/

 MaxPoolLayer::MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride)
 {
 	// need to set size of neurons and dneurons and make sure it goes evenly across new neurons
 	const vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
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

	resize3DVector(m_neurons,newWidth,newHeight,newDepth);
	resize3DVector(m_dneurons,newWidth,newHeight,newDepth);

	//set size of m_dprevLayerMap
	//resize3DVector(m_dprevLayerMap,pWidth,pHeight,pDepth);

 }

MaxPoolLayer::~MaxPoolLayer(){}

void MaxPoolLayer::forwardprop(const Layer& prevLayer)
{
	vector<vector<vector<double> > > source = prevLayer.getNeurons();
	int oX=0, oY=0;//, di, dj; //, dk;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size()-1;i+=m_stride) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size()-1;j+=m_stride) // cols
			{
				double maxVal = source[i][j][k];
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
}

void MaxPoolLayer::backprop(Layer& prevLayer)
{
	//m_dneurons should have been set at this point by the next layer's backprop
	//set all prev dNeurons to 0
	vector<vector<vector<double> > > p_dNeurons = prevLayer.getdNeurons();
 	setAll3DVector(p_dNeurons, 0);
 	vector<vector<vector<double> > > source = prevLayer.getNeurons();

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
				double maxVal = source[i][j][k];
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


const vector<vector<vector<double> > >& MaxPoolLayer::getNeurons() const
{
	return m_neurons;
}

vector<vector<vector<double> > >& MaxPoolLayer::getdNeurons()
{
	return m_dneurons;
}

/**********************
 * ActivLayer
 **********************/


ActivLayer::ActivLayer(const Layer& prevLayer, const int activationType)
{
	int notFound = 1;
	//check if valid activation Type
	if(0 <= activationType && activationType < ActivLayer::NUM_ACTIV_TYPES)
	{ 
		notFound = 0;
	}
	
	if(notFound)
	{
		cout << "The activationType was not valid. Please use a valid activationType. Exp: ActivLayer::RELU";
		throw "Invalid activationType";
	}
	a_activationType = activationType;
	vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
	int w = prevNeurons.size();
	int h = prevNeurons[0].size();
	int d = prevNeurons[0][0].size();
	resize3DVector(a_neurons,w,h,d);
	resize3DVector(a_dneurons,w,h,d);
}

ActivLayer::~ActivLayer(){}

void ActivLayer::forwardprop(const Layer& prevLayer)
{
	vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();

	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					a_neurons[i][j][k] = GETMAX(0,prevNeurons[i][j][k]);
				}
			}
		}
	}
}

void ActivLayer::backprop(Layer& prevLayer)
{
	vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
	vector<vector<vector<double> > > p_dNeurons = prevLayer.getdNeurons();
	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					//a_neurons[i][j][k] = GETMAX(0,prevNeurons[i][j][k]);
					if(prevNeurons[i][j][k] >= 0)
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
}

const vector<vector<vector<double> > >& ActivLayer::getNeurons() const
{
	return a_neurons;
}

vector<vector<vector<double> > >& ActivLayer::getdNeurons()
{
	return a_dneurons;
}

/***********************************
 * Functions
 ***********************************/
void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth)
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

void setAll4DVector(vector<vector<vector<vector<double> > > > &vect, double val)
{
	for(int n=0; n< vect.size(); n++)
	{
		setAll3DVector(vect[n],val);
	}
}

void setAll3DVector(vector<vector<vector<double> > > &vect, double val)
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

void printVector(const vector<vector<vector<vector<double> > > > &vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		printVector(vect[n]);
		cout << endl;
	}
}

void printVector(const vector<vector<vector<double> > > &vect)
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

/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
void padZeros(const vector<vector<vector<double> > > &source, int numZeros, vector<vector<vector<double> > > &dest)
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

void softmax(const vector<vector<vector<double> > >& vect, vector<double>& normedPredictionsContainer)
{
	vector<double>& out = normedPredictionsContainer;
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
	for(i=0;i<width;i++)
	{
		for(j=0;j<height;j++)
		{
			for(k=0;k<height;k++)
			{
				out[*dir] = vect[i][j][k];
			}
		}
	}

	//now its in a 1D array

	//subtract the mean
	meanSubtraction(out);
	double denom = vectorESum(out);
	for(int n=0; n< out.size(); n++)
	{
		out[i] = exp(out[i])/denom;
	}
}

double vectorESum(const vector<double> &source)
{
	double sum = 0;
	for(int i=0; i < source.size(); i++)
	{
		sum += exp(source[i]);
	}
	return sum;
}

void meanSubtraction(vector<double>& vect)
{
	double mean = 0;
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

int main(void)
{
	
}
























