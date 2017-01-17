#include "ConvNetCL.h"
#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

typedef vector<vector<vector<double> > > imVector;


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

int readBigEndianInt(ifstream& in)
{
	int out = 0;
	for(int i=3; i >= 0; i--)
		out |= (readUChar(in) << 8*(i));
	return out;
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

int convertDataType(int dataType)
{
	int convert = dataType;
	if(dataType == 0x08) convert = 1; 		//unsigned byte
	else if(dataType == 0x09) convert = -1;	//signed byte
	else if(dataType == 0x0B) convert = -2;	//signed short
	else if(dataType == 0x0C) convert = -4;	//signed int
	else if(dataType == 0x0D) convert = 5;	//float
	else if(dataType == 0x0E) convert = 6;	//double
	return convert;
}

double getNextImage(ifstream& in, ifstream& trueval_in, imVector& dest, int x, int y, int z, int sizeByteData, int sizeByteLabel)
{
	//get 1 image
	resize3DVector(dest,x,y,z);
	// printf("resize as %d x %d x %d\n", x,y,z);
	for(int i=0; i < x; i++)
	{
		for(int j=0; j < y; j++)
		{
			for(int k=0; k < z; k++)
			{
				if(sizeByteData == 1)
					dest[i][j][k] = (double)readUChar(in);
				else if(sizeByteData == -1)
					dest[i][j][k] = (double)readChar(in);
				else if(sizeByteData == 2)
					dest[i][j][k] = (double)readUShort(in);
				else if(sizeByteData == -2)
					dest[i][j][k] = (double)readShort(in);
				else if(sizeByteData == 4)
					dest[i][j][k] = (double)readUInt(in);
				else if(sizeByteData == -4)
					dest[i][j][k] = (double)readInt(in);
				else if(sizeByteData == 5)
					dest[i][j][k] = (double)readFloat(in);
				else if(sizeByteData == 6)
					dest[i][j][k] = readDouble(in);
				else
				{
					cout << "Unknown sizeByte for data: " << sizeByteData << ". Exiting" << endl;
					exit(0);
				}
			}
		}
	}

	//return the trueVal
	double trueVal = 0;
	if(sizeByteLabel == 1)
		trueVal = (double)readUChar(trueval_in);
	else if(sizeByteLabel == -1)
		trueVal = (double)readChar(trueval_in);
	else if(sizeByteLabel == 2)
		trueVal = (double)readUShort(trueval_in);
	else if(sizeByteLabel == -2)
		trueVal = (double)readShort(trueval_in);
	else if(sizeByteLabel == 4)
		trueVal = (double)readUInt(trueval_in);
	else if(sizeByteLabel == -4)
		trueVal = (double)readInt(trueval_in);
	else if(sizeByteLabel == 5)
		trueVal = (double)readFloat(trueval_in);
	else if(sizeByteLabel == 6)
		trueVal = readDouble(trueval_in);
	else
	{
		cout << "Unknown sizeByte for data: " << sizeByteLabel << ". Exiting" << endl;
		exit(0);
	}
	// printf("trueVal: %lf\n", trueVal);
	return trueVal;
}

//this function parses the magic number at the beginning of an idx file and therefore
//SHOULD ONLY BY USED AT THE BEGINNING OF AN IDX FILE
//this also advance the ifstream cursor past the magic number
void magic(ifstream& in, int& dataType, int& numDims)
{
	if(readUChar(in) + readUChar(in) != 0)
	{
		printf("bad stuff happenin\n");
		exit(0);
	}
	dataType = (int)readUChar(in);
	numDims = (int)readUChar(in);
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		// printf("Use as: ./MNIST_test path/to/NetConfig.txt saveName.txt dataBatchSize deviceNum -DE(optional) -DEType\n");
		printf("Use as: ./MNIST_test path/to/NetConfig.txt saveName.txt dataBatchSize deviceNum -ant(optional)\n");
		return 0;
	}
	int device = 0;
	// bool useDE = false;
	bool useAnt = false;
	// int detype = 0;
	int dataBatchSize = atoi(argv[3]);
	if(argc >= 5)
		device = atoi(argv[4]);
	// if(argc >= 6 && string(argv[5]) == "-DE")
	// {
	// 	useDE = true;
	// 	detype = atoi(argv[6]);
	// }
	if(argc >= 6 && string(argv[5]) == "-ant")
	{
		useAnt = true;
	}

	/**************************
	*
	* get all training metadata and set up for reading in images
	*
	**************************/
	ifstream training_label_in("train-labels.idx1-ubyte");
	ifstream training_data_in("train-images.idx3-ubyte");

	//NOTE: the numDims includes the number of images, so x amount of rgb images would have a numDims of 4
	int train_data_dataType, train_data_numDims, train_label_dataType, train_label_numDims;
	magic(training_label_in, train_label_dataType, train_label_numDims);
	magic(training_data_in, train_data_dataType, train_data_numDims);

	int train_data_convType = convertDataType(train_data_dataType);
	int train_label_convType = convertDataType(train_label_dataType);

	int numTraining = readBigEndianInt(training_label_in); //number of items in label
	if(readBigEndianInt(training_data_in) != numTraining)
	{
		printf("The training data and label files don't have the same amount of items");
		return 0;
	}
	vector<int> trainDims(3,1);
	if(train_data_numDims - 1 > 3)
	{
		printf("Can only handle at most 3 dimensions in the training data right now. Sorry.\n");
		return 0;
	}
	for(int i = 0; i < train_data_numDims - 1; i++)
		trainDims[i] = readBigEndianInt(training_data_in);

	/**************************
	*
	* get all test metadata and set up for reading in images
	*
	**************************/

	ifstream test_label_in("t10k-labels.idx1-ubyte");
	ifstream test_data_in("t10k-images.idx3-ubyte");
	int test_data_dataType, test_data_numDims, test_label_dataType, test_label_numDims;
	magic(test_label_in, test_label_dataType, test_label_numDims);
	magic(test_data_in, test_data_dataType, test_data_numDims);

	int test_data_convType = convertDataType(test_data_dataType);
	int test_label_convType = convertDataType(test_label_dataType);

	int numTest = readBigEndianInt(test_label_in);
	if(readBigEndianInt(test_data_in) != numTest)
	{
		printf("The test data and label files don't have the same amount of items\n");
		return 0;
	}
	vector<int> testDims(3,1);
	if(test_data_numDims - 1 > 3)
	{
		printf("Can only handle at most 3 dimensions in the test data right now. Sorry.\n");
		return 0;
	}
	for(int i = 0; i < test_data_numDims - 1; i++)
		testDims[i] = readBigEndianInt(test_data_in);

	printf("numTrain %d numTest %d\n", numTraining, numTest);
	printf("Train are %d x %d x %d\n", trainDims[0],trainDims[1],trainDims[2]);
	printf("Test are %d x %d x %d\n", testDims[0],testDims[1],testDims[2]);
	printf("Converted train data: %d label: %d\n", train_data_convType,train_label_convType);
	printf("Converted test data: %d label: %d\n", test_data_convType,test_label_convType);

	vector<imVector> training_data(numTraining), test_data(numTest);
	vector<double> training_true(numTraining), test_true(numTest);

	for(int i = 0; i < numTraining; i++)
	{
		training_true[i] = getNextImage(training_data_in, training_label_in, training_data[i],trainDims[0],trainDims[1],trainDims[2],train_data_convType, train_label_convType);
	}
	// int quicksize = 1000;
	// test_true.resize(quicksize);
	// test_data.resize(quicksize);
	for(int i = 0; i < numTest; i++)
	{
		test_true[i] = getNextImage(test_data_in, test_label_in, test_data[i], testDims[0],testDims[1],testDims[2], test_data_convType, test_label_convType);
	}

	training_label_in.close();
	training_data_in.close();
	test_label_in.close();
	test_data_in.close();


	Net net(argv[1]);
	// net.save("testnetsave.txt");
	// printf("saved\n");
	// return 0;
	net.preprocessCollectively();
	net.setSaveName(argv[2]);
	net.setTrainingType(TRAIN_AS_IS);
	net.setDevice(device);
	// net.set_learningRate(0);
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Something went wrong making the net. Exiting." << endl;
		return 0;
	}
	net.addTrainingData(training_data,training_true);

	if(useAnt)
		net.antTrain(10000, 10, dataBatchSize);
	// else if(useDE)
	// 	net.DETrain_sameSize(detype,1000,dataBatchSize);
	else
		net.train();

	net.addData(test_data);

	net.run();

	vector<int> predictions;

	net.getCalculatedClasses(predictions);

	int numCorrect = 0;

	for(int i = 0; i < predictions.size(); i++)
	{
		if(predictions[i] == test_true[i])
			numCorrect++;
	}

	printf("Results on test data: %lf%%\n", 100.0 * numCorrect/predictions.size());


	return 0;
}
