/************************************
*
*	ConvNetDriver
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetDriver cnnConfig.txt binaryImagesFile outputName.txt append(defaults to false)
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
	/*
	char num[2];
	in.read(num,2);
	return num[0] | num[1] << 8;*/

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

/**********************
 *	Functions
 ***********************/

void getNextImage(ifstream& in, imVector& dest, short x, short y, short z, short sizeByte)
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
}

void saveClassesToFile(const char* filename, const vector<int>& calcedClasses, bool append)
{
	ofstream out;
	char buffer[50];
	int size;
	if(append)
		out.open(filename,ios::app);
	else
		out.open(filename,ios::trunc);

	for(int i=0; i < calcedClasses.size(); i++)
	{
		sprintf(buffer,"%d\n",calcedClasses[i]);
		size = 0;
		while(buffer[size] != '\0')
			size++;
		out.write(buffer,size);
	}
	out.close();
}

void convertBinaryToVector(const char *filename, vector<imVector>& dest)
{
	ifstream in;
	in.open(filename, ios::binary);
	if(!in.is_open())
	{
		cout << "File \"" << filename << "\" not found. Exiting." << endl;
		exit(0);;
	}

	in.seekg(0, in.end);
	long end = in.tellg();
	in.seekg(0, in.beg);

	/*
	double testd = readDouble(in);
	int testi = readInt(in);
	unsigned int testui = readUInt(in);
	unsigned short testus = readUShort(in);
	char testc = readChar(in);
	unsigned char testuc = readUChar(in);
	float testf = readFloat(in);
	cout << "Double: " << testd << endl;
	cout << "int: " << testi << endl;
	cout << "uint: " << testui << endl;
	cout << "ushort: " << testus << endl;
	cout << "char: " << (int)testc << endl;
	cout << "uchar: " << (int)testuc << endl;
	cout << "float: " << testf << endl;*/

	short sizeByte = readShort(in);
	short xSize = readShort(in);
	short ySize = readShort(in);
	short zSize = readShort(in);	

	cout << "Size: " << sizeByte << " x: " << xSize << " y: " << ySize << " z: " << zSize << endl;
	while(in.tellg() != end)
	{
		dest.resize(dest.size() + 1);
		getNextImage(in,dest.back(),xSize,ySize,zSize,sizeByte);
	}
	in.close();

	cout << "Num images = " << dest.size() << endl;

	//printVector(dest);
}

int main(int argc, char** argv)
{
	if(argc != 4 && argc != 5)
	{
		cout << "Usage: ./ConvNetDriver cnnConfig.txt binaryImagesFile outputName.txt append(true or false, defaults to false)" << endl;
		return 0;
	}

	bool append = false;
	if(argc == 5)
	{
		int i = 0;
		while(argv[4][i] != '\0')
		{
			argv[4][i] = tolower(argv[4][i]);
			i++;
		}
		string arg(argv[4]);
		if(arg.find("true") != string::npos)
			append = true;
	}

	//set up net
	Net net(argv[1]);

	//get images and preprocess them
	vector<imVector> images(0);
	convertBinaryToVector(argv[2],images);
	preprocess(images);

	net.addRealData(images);

	vector<int> calcedClasses;

	time_t starttime = time(NULL);
	net.newRun(calcedClasses,false);
	time_t endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << endl;

	saveClassesToFile(argv[3], calcedClasses, append);
}