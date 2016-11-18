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

double getNextImage(ifstream& in, ifstream& trueval_in, imVector& dest, short x, short y, short z, short sizeByte)
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
	return (double)readUChar(trueval_in);
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("Use as: ./MNIST_test path/to/NetConfig.txt saveName.txt deviceNum -DE(optional)\n");
		return 0;
	}
	int device = 0;
	bool useDE = false;
	if(argc >= 4)
		device = atoi(argv[3]);
	if(argc >= 5 && string(argv[4]) == "-DE")
		useDE = true;

	ifstream training_label_in("train-labels.idx1-ubyte");
	ifstream training_data_in("train-images.idx3-ubyte");
	ifstream test_label_in("t10k-labels.idx1-ubyte");
	ifstream test_data_in("t10k-images.idx3-ubyte");

	readInt(training_label_in); //magic number
	int numTraining = readBigEndianInt(training_label_in); //number of items

	readInt(test_label_in); //magic number
	int numTest = readBigEndianInt(test_label_in);

	for(int i =0; i < 4; i++)
	{
		readInt(training_data_in);
		readInt(test_data_in);
	}

	printf("numTrain %d numTest %d\n", numTraining, numTest);

	vector<imVector> training_data(numTraining), test_data(numTest);
	vector<double> training_true(numTraining), test_true(numTest);

	for(int i = 0; i < numTraining; i++)
	{
		training_true[i] = getNextImage(training_data_in, training_label_in, training_data[i],28,28,1,1);
	}
	for(int i = 0; i < numTest; i++)
	{
		test_true[i] = getNextImage(test_data_in, test_label_in, test_data[i],28,28,1,1);
	}

	training_label_in.close();
	training_data_in.close();
	test_label_in.close();
	test_data_in.close();

	Net net(argv[1]);
	net.preprocessCollectively();
	net.setSaveName(argv[2]);
	net.setTrainingType(TRAIN_AS_IS);
	cout << boolalpha << net.setDevice(device) << endl;
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Something went wrong making the net. Exiting." << endl;
		return 0;
	}
	net.addTrainingData(training_data,training_true);

	if(useDE)
		net.DETrain_sameSize(100);
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
