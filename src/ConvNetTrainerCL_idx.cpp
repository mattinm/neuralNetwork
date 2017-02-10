#include "ConvNetCL.h"
#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>

// #include "opencv2/imgproc/imgproc.hpp" //used for showing images being read in from IDX
// #include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

typedef vector<vector<vector<double> > > imVector;

int imcount = 0;
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

double getNextImage_byCount(ifstream& in, ifstream& trueval_in, imVector& dest, int x, int y, int z, int sizeByteData, int sizeByteLabel, int numLabelCounts)
{
	//get 1 image
	resize3DVector(dest,x,y,z);
	// printf("resize as %d x %d x %d\n", x,y,z);
	for(int i=0; i < x; i++)
	{
		for(int j=0; j < y; j++)
		{
			//for(int k=0; k < z; k++)
			for(int k = z-1; k >=0; k--) //need to read in rgb idx as bgr b/c that's how opencv does it.
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
	double retVal = 0;
	for(int i = 1; i <= numLabelCounts; i++)
	{
		double count = 0;
		if(sizeByteLabel == 1)
			count = (double)readUChar(trueval_in);
		else if(sizeByteLabel == -1)
			count = (double)readChar(trueval_in);
		else if(sizeByteLabel == 2)
			count = (double)readUShort(trueval_in);
		else if(sizeByteLabel == -2)
			count = (double)readShort(trueval_in);
		else if(sizeByteLabel == 4)
			count = (double)readUInt(trueval_in);
		else if(sizeByteLabel == -4)
			count = (double)readInt(trueval_in);
		else if(sizeByteLabel == 5)
			count = (double)readFloat(trueval_in);
		else if(sizeByteLabel == 6)
			count = readDouble(trueval_in);
		else
		{
			cout << "Unknown sizeByte for data: " << sizeByteLabel << ". Exiting" << endl;
			exit(0);
		}
		if(count > 0)// && i==1) // second half restricts to white geese
			retVal = i;
	}


	//show image and trueVal
	// if(true)
	// {
	// 	Mat show(x,y,CV_8UC3);
	// 	for(int i = 0; i < x; i++)
	// 	{
	// 		for(int j = 0; j < y; j++)
	// 		{
	// 			Vec3b& outPix = show.at<Vec3b>(i,j);
	// 			outPix[0] = dest[i][j][2];
	// 			outPix[1] = dest[i][j][1];
	// 			outPix[2] = dest[i][j][0];
	// 		}
	// 	}
	// 	char name[10];
	// 	sprintf(name,"%d,%d",(int)retVal,imcount);
	// 	imshow(name,show);
	// 	waitKey(0);
	// 	imcount++;
	// }

	
	// printf("trueVal: %lf\n", trueVal);
	return retVal;
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
		printf("Use as: \n");
		printf("  REQUIRED FIRST - ./ConvNetTrainerCL_idx path/to/NetConfig.txt saveName.txt\n");
		printf("  In any order at end:\n");
		printf("		-device=<int>                           OpenCL Device to run on. Optional w/default 0.\n");
		printf("		-train_data=path/to/train/data.idx      Path to training data\n");
		printf("		-train_label=path/to/train/labels.idx   Path to training labels\n");
		printf("		-test_data=path/to/test/data.idx        Path to test data\n");
		printf("		-test_label=path/to/test/labels.idx     Path to test labels\n");
		return 0;
	}
	int device = 0;
	int cmd_train_count = 0;
	int cmd_test_count = 0;
	string train_data_path, test_data_path, train_label_path, test_label_path;
	char * netConfig_path = argv[1];
	char * saveName = argv[2];
	for(int i = 3; i < argc; i++)
	{
		string arg = string(argv[i]);
		if(arg.find("-device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-train_data=") != string::npos)
		{
			train_data_path = arg.substr(arg.find('=')+1);
			cmd_train_count++;
		}
		else if(arg.find("-train_label=") != string::npos)
		{
			train_label_path = arg.substr(arg.find('=')+1);
			cmd_train_count++;
		}
		else if(arg.find("-test_data=") != string::npos)
		{
			test_data_path = arg.substr(arg.find('=')+1);
			cmd_test_count++;
		}
		else if(arg.find("-test_label=") != string::npos)
		{
			test_label_path = arg.substr(arg.find('=')+1);
			cmd_test_count++;
		}
	}

	if(cmd_train_count == 1)
	{
		printf("Training data AND label paths are needed\n");
		return 0;
	}
	else if(cmd_train_count == 0)
	{
		printf("NO TRAINING WILL BE DONE, ONLY TESTING\n");
	}
	else if(cmd_train_count > 2)
	{
		printf("You appear to have put in more than 1 set of training data, which is not currently supported. Exiting.\n");
		return 0;
	}

	if(cmd_test_count == 1)
	{
		printf("You appear to have specified either the test labels or test data paths, but not both\n");
		return 0;
	}
	else if(cmd_test_count > 2)
	{
		printf("You appear to have put in more than 1 set of test data, which is not currently supported. Exiting.\n");
		return 0;
	}


	

	/**************************
	*
	* get all training metadata and set up for reading in images
	*
	**************************/
	int numTraining = 0, train_data_convType, train_label_convType;
	ifstream training_label_in, training_data_in;
	vector<int> trainDims(3,1), train_label_dims;
	if(cmd_train_count > 0)
	{
		training_label_in.open(train_label_path.c_str());
		training_data_in.open(train_data_path.c_str());

		//NOTE: the numDims includes the number of images, so x amount of rgb images would have a numDims of 4
		int train_data_dataType, train_data_numDims, train_label_dataType, train_label_numDims;
		magic(training_label_in, train_label_dataType, train_label_numDims);
		magic(training_data_in, train_data_dataType, train_data_numDims);

		train_data_convType = convertDataType(train_data_dataType);
		train_label_convType = convertDataType(train_label_dataType);

		numTraining = readBigEndianInt(training_label_in); //number of items in label
		if(readBigEndianInt(training_data_in) != numTraining)
		{
			printf("The training data and label files don't have the same amount of items");
			return 0;
		}
		if(train_data_numDims - 1 > 3)
		{
			printf("Can only handle at most 3 dimensions in the training data right now. Sorry.\n");
			return 0;
		}
		for(int i = 0; i < train_data_numDims - 1; i++)
			trainDims[i] = readBigEndianInt(training_data_in);

		train_label_dims.resize(train_label_numDims);
		for(int i = 0; i < train_label_numDims - 1; i++)
			train_label_dims[i] = readBigEndianInt(training_label_in);
	}

	/**************************
	*
	* get all test metadata and set up for reading in images
	*
	**************************/

	int numTest = 0, test_data_convType, test_label_convType;
	ifstream test_label_in, test_data_in;
	vector<int> testDims(3,1), test_label_dims;
	if(cmd_test_count > 0)
	{
		test_label_in.open(test_label_path.c_str());
		test_data_in.open(test_data_path.c_str());
		int test_data_dataType, test_data_numDims, test_label_dataType, test_label_numDims;
		magic(test_label_in, test_label_dataType, test_label_numDims);
		magic(test_data_in, test_data_dataType, test_data_numDims);

		test_data_convType = convertDataType(test_data_dataType);
		test_label_convType = convertDataType(test_label_dataType);

		numTest = readBigEndianInt(test_label_in);
		if(readBigEndianInt(test_data_in) != numTest)
		{
			printf("The test data and label files don't have the same amount of items\n");
			return 0;
		}
		if(test_data_numDims - 1 > 3)
		{
			printf("Can only handle at most 3 dimensions in the test data right now. Sorry.\n");
			return 0;
		}
		for(int i = 0; i < test_data_numDims - 1; i++)
			testDims[i] = readBigEndianInt(test_data_in);

		test_label_dims.resize(test_label_numDims);
		for(int i = 0; i < test_label_numDims - 1; i++)
			test_label_dims[i] = readBigEndianInt(test_label_in);
	}

	

	printf("numTrain %d numTest %d\n", numTraining, numTest);
	printf("Train are %d x %d x %d\n", trainDims[0],trainDims[1],trainDims[2]);
	if(cmd_test_count > 0)
		printf("Test are %d x %d x %d\n", testDims[0],testDims[1],testDims[2]);

	vector<imVector> training_data(numTraining), test_data(numTest);
	vector<double> training_true(numTraining), test_true(numTest);

	for(int i = 0; i < numTraining; i++)
	{
		training_true[i] = getNextImage_byCount(training_data_in, training_label_in, training_data[i],trainDims[0],trainDims[1],trainDims[2],train_data_convType, train_label_convType,train_label_dims[0]);
	}

	for(int i = 0; i < numTest; i++)
	{
		test_true[i] = getNextImage_byCount(test_data_in, test_label_in, test_data[i], testDims[0],testDims[1],testDims[2], test_data_convType, test_label_convType,test_label_dims[0]);
	}


	readChar(training_label_in);
	if(training_label_in.eof())
	//if(readChar(training_label_in) == EOF)
	{
		printf("EOF\n");
	}
	else 
	{
		printf("Not EOF\n");
	}

	training_label_in.close();
	training_data_in.close();
	test_label_in.close();
	test_data_in.close();


	vector<int> numcolorimages(3,0);
	vector<int> numtotalbyclass(3,0);
	for(int i = 0; i < numTraining; i++)
	{
		int colorFound = 0;
		for(int j = 0; j < training_data[i].size(); j++)
			for(int k = 0; k < training_data[i][j].size(); k++)
				for(int l = 0; l < training_data[i][j][k].size(); l++)
					if(training_data[i][j][k][l] != 0)
					{
						colorFound = 1;
					}
		numcolorimages[training_true[i]] += colorFound;
		numtotalbyclass[training_true[i]]++;
	}
	for(int i = 0; i < 3; i++)
		printf("Num color images class %d is %d of %d. %lf%%\n", i, numcolorimages[i], numtotalbyclass[i], 100. * numcolorimages[i]/numtotalbyclass[i]);




	Net net(netConfig_path);
	net.preprocessCollectively();
	net.setSaveName(saveName);
	net.setTrainingType(TRAIN_AS_IS);
	// net.setTrainingType(TRAIN_EQUAL_PROP);
	net.setDevice(device);
	// net.set_learningRate(0);
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Something went wrong making the net. Exiting." << endl;
		return 0;
	}
	if(cmd_train_count > 0)
	{
		net.addTrainingData(training_data,training_true);
		net.printTrainingDistribution();

		net.train();
	}
	if(cmd_test_count > 0)
	{
		net.addData(test_data);

		net.run();

		vector<int> predictions;

		net.getCalculatedClasses(predictions);

		int numCorrect = 0;
		vector<int> numCorrectClass(net.getNumClasses(),0);
		vector<int> numTotalClass(net.getNumClasses(),0);

		for(int i = 0; i < predictions.size(); i++)
		{
			if(predictions[i] == test_true[i])
			{
				numCorrect++;
				numCorrectClass[test_true[i]]++;
			}
			numTotalClass[test_true[i]]++;
		}

		printf("Results on test data: %lf%%\n", 100.0 * numCorrect/predictions.size());
		for(int i = 0; i < numCorrectClass.size(); i++)
		{
			printf("    Class %d: %d out of %d, %lf%%\n",i, numCorrectClass[i],numTotalClass[i],100.*numCorrectClass[i]/numTotalClass[i]);
		}
	}


	return 0;
}
