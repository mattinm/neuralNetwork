/************************************
*
*	ConvNetDriver
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetDriver binaryImagesFile outputName.txt
*
*************************************/


#include <iostream>
#include <vector>
#include "ConvNet.h"
#include <fstream>

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
	/*
	char num[2];
	in.read(num,2);
	return num[0] | num[1] << 8;*/

	unsigned short num;
	in.read((char*)&num,sizeof(unsigned short));
	return num;
}

int readInt(ifstream& in)
{
	/*
	char num[4];
	in.read(num,4);
	return (num[0]) | (num[1] << 8) | (num[2] << 16) | (num[3] << 24);
	*/

	int num;
	in.read((char*)&num,sizeof(int));
	return num;
}
unsigned int readUInt(ifstream& in)
{
	/*
	char num[4];
	in.read(num,4);
	return num[0] | num[1] << 8 | num[2] << 16 | num[3] << 24;
	*/

	unsigned int num;
	in.read((char*)&num,sizeof(unsigned int));
	return num;
}

float readFloat(ifstream& in)
{
	/*
	char num[4];
	in.read(num,4);
	return num[0] | num[1] << 8 | num[2] << 16 | num[3] << 24;*/

	float num;
	in.read((char*)&num,sizeof(float));
	return num;
}

double readDouble(ifstream& in)
{
	/*
	char num[8];
	in.read(num,8);
	return num[0] | num[1] << 8 | num[2] << 16 | num[3] << 24 | num[4] << 32 | num[5] << 40 | num[6] << 48 | num[7] << 56;
	*/

	double num;
	in.read((char*)&num,sizeof(double));
	return num;
}

/**********************
 *	Functions
 ***********************/

template<typename type>
void getNextImage(ifstream& in, vector<imVector>& dest, short x, short y, short z)
{

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

	char buffer[100];
	buffer[99] = '\0';
	//in.read(buffer,99);
	//cout << "buffer: " << buffer << endl;
	double testd = readDouble(in);
	int testi = readInt(in);
	unsigned int testui = readUInt(in);
	unsigned short testus = readUShort(in);
	char testc = readChar(in);
	unsigned char testuc = readUChar(in);
	float testf = readFloat(in);

	short sizeByte = readShort(in);
	short xSize = readShort(in);
	short ySize = readShort(in);
	short zSize = readShort(in);
	/*
	sizeByte = buffer[0] | buffer[1] << 8;
	xSize = buffer[2] | buffer[3] << 8;
	ySize = buffer[4] | buffer[5] << 8;
	zSize = buffer[6] | buffer[7] << 8;*/

	cout << "Double: " << testd << endl;
	cout << "int: " << testi << endl;
	cout << "uint: " << testui << endl;
	cout << "ushort: " << testus << endl;
	cout << "char: " << (int)testc << endl;
	cout << "uchar: " << (int)testuc << endl;
	cout << "float: " << testf << endl;

	cout << "Size: " << sizeByte << " x: " << xSize << " y: " << ySize << " z: " << zSize << endl; 



	in.close();
}

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout << "Usage: ./ConvNetDriver binaryImagesFile outputName.txt" << endl;
		return 0;
	}

	vector<imVector> images;
	convertBinaryToVector(argv[1],images);

}