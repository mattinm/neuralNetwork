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
#include "ConvNet.h"
#include <ctype.h>
#include <fstream>
#include <time.h>

using namespace std;

typedef vector<vector<vector<double> > > imVector;

long __ifstreamend;

/**********************
 *	Helper Functions
 ***********************/

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
		cout << "Usage: ./ConvNetTrainer CNNConfigFile.txt binaryTrainingImagesFile outname=<outputName.txt> epochs=<#epochs> gpu=<true/false> device=<device#>";
		cout << "\n\tThe outputName and epochs are optional keyword arguments. If no outname is specified the weights will be saved over the old weights. Epochs defaults to 1. Defaults to using GPU" << endl;
		return 0;
	}

	time_t starttime,endtime;

	string outputName;
	bool saveNewWeights = false;
	int epochs = 1;
	int device = -1;
	bool useGPU = true;

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
			{
				epochs = stoi(arg.substr(arg.find("=")+1));
			}
			else if(arg.find("gpu=") != string::npos)
			{
				if(arg.find("false") != string::npos || arg.find("False") != string::npos)
				{
					useGPU = false;
				}
			}
			else if(arg.find("device=") != string::npos)
			{
				device = stoi(arg.substr(arg.find("=")+1));
			}
		}
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
	if(!net.isActive())
	{
		cout << "Net did not load correctly. Exiting." << endl;
		return 0;
	}
	

	//get images and preprocess them
	vector<imVector> images(0);
	vector<double> trueVals(0);

	starttime = time(NULL);
	convertBinaryToVector(in,images,trueVals,sizeByte,xSize,ySize,zSize);
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	in.close();
	//preprocessByFeature(images);
	//preprocessCollective(images);
	preprocess(images);
	//meanSubtraction(images);

	net.shuffleTrainingData(10);

	cout << "Adding training data" << endl;
	net.addTrainingData(images,trueVals);

	net.printTrainingDistribution();

	net.setDevice(device);

	starttime = time(NULL);
	net.OpenCLTrain(epochs, useGPU);
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