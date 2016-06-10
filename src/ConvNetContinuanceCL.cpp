/************************************
*
*	ConvNetTrainer
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetTrainer binaryTrainingImagesFile outname=<outputName.txt> epochs=<#epochs> gpu=<true/false>
*
* 		The outputName and epochs are optional keyword arguments. If no outname is specified the weights are not saved. Epochs defaults to 1. Defaults to using GPU
*
*************************************/


#include <iostream>
#include <vector>
#include "ConvNetCL.h"
#include <ctype.h>
#include <fstream>
#include <time.h>

using namespace std;

typedef vector<vector<vector<double> > > imVector;

long __ifstreamend;

/**********************
 *	Helper Functions
 ***********************/

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

char readChar(ifstream& in)
{
	char num[1];
	in.read(num,1);
	return num[0];
}
unsigned char readUChar(ifstream& in)
{
	char num[1];
	in.read(num,1);
	return num[0];
}

short readShort(ifstream& in)
{
	short num;
	in.read((char*)&num,sizeof(short));
	return num;
}
unsigned short readUShort(ifstream& in)
{
	unsigned short num;
	in.read((char*)&num,sizeof(unsigned short));
	return num;
}

int readInt(ifstream& in)
{
	int num;
	in.read((char*)&num,sizeof(int));
	return num;
}
unsigned int readUInt(ifstream& in)
{
	unsigned int num;
	in.read((char*)&num,sizeof(unsigned int));
	return num;
}

float readFloat(ifstream& in)
{
	float num;
	in.read((char*)&num,sizeof(float));
	return num;
}

double readDouble(ifstream& in)
{
	double num;
	in.read((char*)&num,sizeof(double));
	return num;
}

string secondsToString(time_t seconds)
{
	time_t secs = seconds%60;
	time_t mins = (seconds%3600)/60;
	time_t hours = seconds/3600;
	char out[100];
	if(hours > 0)
		sprintf(out,"%ld hours, %ld mins, %ld secs",hours,mins,secs);
	else if(mins > 0)
		sprintf(out,"%ld mins, %ld secs",mins,secs);
	else
		sprintf(out,"%ld secs",secs);
	string outString = out;
	return outString;
}

string secondsToString(float seconds)
{
	float secs = (int)seconds%60 + (seconds - (int)seconds);
	time_t mins = ((int)seconds%3600)/60;
	time_t hours = (int)seconds/3600;
	char out[100];
	if(hours > 0)
		sprintf(out,"%ld hours, %ld mins, %0.2f secs",hours,mins,secs);
	else if(mins > 0)
		sprintf(out,"%ld mins, %0.2f secs",mins,secs);
	else
		sprintf(out,"%0.2f secs",secs);
	string outString = out;
	return outString;
}

/**********************
 *	Functions
 ***********************/

// returns the true value
double getNextImage(ifstream& in, imVector& dest, short x, short y, short z, short sizeByte)
{
	resize3DVector(dest,x,y,z);
	for(int i=0; i < x; i++)
	{
		for(int j=0; j < y; j++)
		{
			for(int k=0; k < z; k++)
			{
				if(sizeByte == 1)
					dest[i][j][k] = (double)readUChar(in);
				else if(sizeByte == -1)
					dest[i][j][k] = (double)readChar(in);
				else if(sizeByte == 2)
					dest[i][j][k] = (double)readUShort(in);
				else if(sizeByte == -2)
					dest[i][j][k] = (double)readShort(in);
				else if(sizeByte == 4)
					dest[i][j][k] = (double)readUInt(in);
				else if(sizeByte == -4)
					dest[i][j][k] = (double)readInt(in);
				else if(sizeByte == 5)
					dest[i][j][k] = (double)readFloat(in);
				else if(sizeByte == 6)
					dest[i][j][k] = readDouble(in);
				else
				{
					cout << "Unknown sizeByte: " << sizeByte << ". Exiting" << endl;
					exit(0);
				}
			}
		}
	}

	//return the trueVal
	return (double)readUShort(in);
}

void convertBinaryToVector(ifstream& in, vector<imVector>& dest, vector<double>& trueVals, short sizeByte, short xSize, short ySize, short zSize)
{
	if(!in.is_open())
	{
		cout << "ifstream was not open. Exiting." << endl;
		exit(0);;
	}	

	//cout << "Size: " << sizeByte << " x: " << xSize << " y: " << ySize << " z: " << zSize << endl;
	while(in.tellg() != __ifstreamend)// && !in.eof())
	{
		if(in.tellg() > __ifstreamend)
		{
			cout << "The counter went past the max. There might be something wrong with the file format" << endl;
			break;
		}
		dest.resize(dest.size() + 1);
		trueVals.push_back(getNextImage(in,dest.back(),xSize,ySize,zSize,sizeByte));
	}

	cout << "Num images = " << dest.size() << endl;
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		cout << "Usage (Required to come first):\n   ./ConvNetContinuanceCL CNNConfigFile.txt binaryTrainingImagesFile";
		cout << "\nOptional arguments (must come after required args, everything before equals sign is case sensitive):\n";
		cout << "   outname=<outname.txt>    Sets the name for the outputted trained CNN. If not specified new weights will be saved over old file.\n";
		cout << "   testSet=<name.txt>       A binary training file to be used as a test/validation set. Never trained on.\n";
		cout << "   epochs=<#epochs>         Number of epochs to train for. Defaults to 1.\n";
		cout << "   device=<device#>         Which OpenCL device on to use. Integer. Defaults to GPU supporting doubles if present, else defaults to CPU.\n";
		cout << "   -train_as_is             Causes CNN to train using all images for every epoch. On by default. Can only use one train method at a time\n";
		cout << "   -train_equal_prop        Causes CNN to train using equal amounts of each class for each epoch. For classes with larger amounts of images,\n";
		cout << "                            the ones used will be randomly chosen each epoch. Can only use one train method at a time\n";
		cout << "   learningRate=<rate>        Sets the learningRate for the CNN.\n";
		cout << "   RELU_CAP=<cap>             Sets max value that can pass through the RELU\n";
		cout << "   LEAKY_RELU_CONST=<const>   Sets the constant for LEAKY_RELU\n";
		cout << "   l2Lambda=<lambda>          Sets the lambda for L2 Regularization\n";
		cout << "   MOMENT_CONST=<const>       Sets the constant decay for the Nesterov Momentum\n";
		cout << "   MAX_NORM_CAP=<cap>         Sets the max value a weight can be\n";
		return 0;
	}

	time_t starttime,endtime;

	string outputName;
	bool saveNewWeights = false;
	int epochs = -1;
	int device = -1;
	string testSetName;
	bool haveTest;
	int haveTrainMethod = 0;
	int trainMethod = TRAIN_AS_IS;
	double learningRate = -1; //-1s are sentinel values
	double reluCap = -1;
	double leaky = -1;
	double l2 = -1;
	double moment = -1;
	double maxNorm = -1;


	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("outname=") != string::npos)
			{
				outputName = arg.substr(arg.find("=") + 1);
				saveNewWeights = true;
			}
			else if(arg.find("epochs=") != string::npos || arg.find("epoch=") != string::npos)
				epochs = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("device=") != string::npos)
				device = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("testSet=") != string::npos)
			{
				testSetName = arg.substr(arg.find("=") + 1);
				haveTest = true;
			}
			else if(arg.find("-train_equal_prop") != string::npos)
			{
				trainMethod = TRAIN_EQUAL_PROP;
				haveTrainMethod++;
			}
			else if(arg.find("-train_as_is") != string::npos)
				haveTrainMethod++; // this is on by default
			else if(arg.find("learningRate=") != string::npos)
				learningRate = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("RELU_CAP=") != string::npos)
				reluCap = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("LEAKY_RELU_CONST=") != string::npos)
				leaky = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("l2Lambda=") != string::npos)
				l2 = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("MOMENT_CONST=") != string::npos)
				moment = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("MAX_NORM_CAP=") != string::npos)
				maxNorm = stod(arg.substr(arg.find("=")+1));
			else
			{
				printf("Unknown arg %s. Aborting.\n", argv[i]);
				return 0;
			}
		}
	}

	if(haveTrainMethod > 1)
	{
		printf("You cannot have multiple training methods simultaneously.\n");
		return 0;
	}

	ifstream in;
	in.open(argv[2]);

	if(!in.is_open())
	{
		cout << "Unable to open file \"" << argv[2] << "\". Exiting." << endl;
		return 0;
	}

	in.seekg(0, in.end);
	__ifstreamend = in.tellg();
	in.seekg(0, in.beg);

	short sizeByte = readShort(in);
	short xSize = readShort(in);
	short ySize = readShort(in);
	short zSize = readShort(in);

	if(xSize == 0 || ySize == 0 || zSize == 0)
	{
		cout << "None of the dimensions can be zero. Exiting." << endl;
		exit(0);
	}

	//set up net
	Net net(argv[1]);
	net.setDevice(device);
	net.setTrainingType(trainMethod);
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Net did not load correctly. Exiting." << endl;
		return 0;
	}
	

	//get images and add them
	vector<imVector> images(0);
	vector<double> trueVals(0);

	starttime = time(NULL);
	convertBinaryToVector(in,images,trueVals,sizeByte,xSize,ySize,zSize);
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	in.close();

	cout << "Adding training data" << endl;
	net.setTrainingData(images,trueVals);

	printf("Training Data Distribution:\n");
	net.printTrainingDistribution();
	printf("\n");

	//get test images if needed
	vector<imVector> testImages(0);
	vector<double> testTrueVals(0);
	if(haveTest)
	{
		ifstream testIn;
		in.open(testSetName.c_str());
		if(!in.is_open())
		{
			cout << "Unable to open test file \"" << testSetName << "\". Exiting." << endl;
			return 0;
		}
		in.seekg(0, in.end);
		__ifstreamend = in.tellg();
		in.seekg(0, in.beg);

		short tsizeByte = readShort(in);
		short txSize = readShort(in);
		short tySize = readShort(in);
		short tzSize = readShort(in);
		if(txSize != xSize || tySize != ySize || tzSize != zSize)
		{
			printf("Training and test images must be of same size.\n");
			return 0;
		}

		printf("Bringing in testing data from file: %s\n", testSetName.c_str());
		starttime = time(NULL);
		convertBinaryToVector(in,testImages,testTrueVals,tsizeByte,txSize,tySize,tzSize);
		endtime = time(NULL);
		cout << "Time to bring in test data: " << secondsToString(endtime - starttime) << endl;
		in.close();

		net.addTestData(testImages, testTrueVals);

		printf("Test Set Distribution:\n");
		net.printTestDistribution();
		printf("\n");
	}


	starttime = time(NULL);
	net.train(epochs);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ". - " << secondsToString((endtime-starttime)/(float)epochs) << " per epoch." << endl;

	if(saveNewWeights)
	{
		net.save(outputName.c_str());
	}
	else
	{
		net.save(argv[1]);
	}
}