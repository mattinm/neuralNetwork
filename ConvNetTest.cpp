/*********************************************
 *
 *	ConvNetTest
 *
 * 	Created by Connor Bowley on 3/16/15
 *
 *********************************************/

#include <iostream>
#include <vector>
#include <iomanip>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ConvNet.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace cv;

void convert1DArrayTo3DVector(const double *array, int width, int height, int depth, vector<vector<vector<double> > > &dest)
{
	//resize dest vector
	dest.resize(width);
	for(int i=0; i < width; i++)
	{
		dest[i].resize(height);
		for(int j=0; j < height; j++)
		{
			dest[i][j].resize(depth);
		}
	}
	
	for(int i=0; i < width;i++)
	{
		for(int j=0; j < height; j++)
		{
			for(int k=0; k < depth; k++)
			{
				dest[i][j][k] = *array++;
			}
		}
	}
}

void convertColorMatToVector(Mat m, vector<vector<vector<double> > > &dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}
	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	
	for(int i=0; i< m.rows; i++)
	{
		for(int j=0; j< m.cols; j++)
		{
			Vec3b curPixel = m.at<Vec3b>(i,j);
			dest[i][j][0] = curPixel[0];
			dest[i][j][1] = curPixel[1];
			dest[i][j][2] = curPixel[2];
		}
	}
}

void getImageInVector(const char* filename, vector<vector<vector<double> > >& dest)
{
	Mat image = imread(filename,1);
	convertColorMatToVector(image,dest);
}

int checkExtensions(const char* filename)
{
	const string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}

void oldestMain()
{
	string weights = "1,0,1,-1,0,0,-1,1,0,0,1,0,1,0,0,1,-1,-1,-1,0,0,1,1,-1,-1,1,0,0,1,-1,-1,-1,-1,-1,0,-1,1,1,-1,0,0,-1,0,1,-1,1,-1,1,1,1,-1,-1,0,1,_1,0";

	double testArray[] = {
		0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
		2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
		1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
		1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
		2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};

	vector<vector<vector<vector<double> > > > testVectors(1);
	vector<double> trueVals(1,1);
	
	convert1DArrayTo3DVector(testArray,5,5,3,testVectors[0]);


	cout << "Making Net" << endl;
	Net *testNet = new Net(5,5,3);
	testNet->addTrainingData(testVectors,trueVals);
	//cout << "Adding ConvLayer" << endl;
	testNet->addConvLayer(2,2,3,1,weights);
	testNet->addActivLayer();
	//testNet->addMaxPoolLayer(2,1);

	testNet->train(1);


	cout << "Done" << endl;
	//delete testNet;
}

void getTrainingImages(const char* folder, int trueVal, vector<vector<vector<vector<double> > > >& images, vector<double>& trueVals)
{
	const char* inPath = folder;
	bool isDirectory;
	struct stat s;
	if(stat(inPath,&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
		{
			isDirectory = true;
		}
		else if (s.st_mode & S_IFREG) // file
		{
			isDirectory = false;
		}
		else
		{
			printf("We're not sure what the file you inputted was.\nExiting\n");
			return;
		}
	}
	else
	{
		cout << "Error getting status of folder or file. \"" << folder << "\"\nExiting\n";
		return;
	}
	
	if(isDirectory)
	{
		DIR *directory;
		struct dirent *file;
		if((directory = opendir(inPath)))// != NULL)
		{
			string pathName = inPath;
			if(pathName.rfind("/") != pathName.length()-1)
			{
				pathName.append(1,'/');
			}
			char inPathandName[250];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					if(checkExtensions(file->d_name))
					{
						sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
						//images.push_back();
						images.resize(images.size() + 1);
						//cout << "getImageInVector(" << inPathandName << ",images[" << images.size()-1 << "])" << endl;
						getImageInVector(inPathandName,images.back());
						trueVals.push_back(trueVal);
					}
				}
			}
			closedir(directory);
		}
	}
	else
	{
		if(checkExtensions(inPath))
		{
			//images.push_back();
			images.resize(images.size() + 1);
			getImageInVector(inPath,images.back());
			trueVals.push_back(trueVal);
		}

	}
}

int firstCNN(int argc, char** argv)
{
	if(argc != 2)
	{
		cout << "A trainingImagesConfig file is needed. See the readMe for format." << endl;
		return -1;
	}
	//set up CNN
	Net net(32,32,3);
	net.setActivType(ActivLayer::LEAKY_RELU);
	net.addConvLayer(6,1,5,0);
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);
	net.addConvLayer(10,1,3,0);
	net.addActivLayer();
	net.addMaxPoolLayer(3,3);
	net.addConvLayer(2,1,4,0);
	net.addActivLayer();

	cout << "NeuralNet set up" << endl;
	
	cout << "Getting training images" << endl;

	vector<vector<vector<vector<double> > > > trainingImages;
	vector<double> trueVals;

	ifstream tiConfig;
	string line;
	tiConfig.open(argv[1]);
	if(!tiConfig.is_open())
	{
		cout << "Could not open the trainingImagesConfig file " << argv[1];
		return -1;
	}
	while(getline(tiConfig,line))
	{
		int loc = line.find(",");
		if(loc != string::npos)
		{
			cout << "Adding folder" << line.substr(0,loc) << endl;
			getTrainingImages(line.substr(0,loc).c_str(),stoi(line.substr(loc+1)),trainingImages,trueVals);
		}
		else
			cout << "Error in trainingImagesConfig at line:\n" << line << endl;
	}
	tiConfig.close();

	cout << trainingImages.size() << " training images added." << endl;

	assert(trainingImages.size() == trueVals.size());
	if(trainingImages.size() == 0)
	{
		cout << "No training images found. Exiting." << endl;
		return 0;
	}
	
	cout << "Doing image preprocessing (mean subtraction)." << endl;
	meanSubtraction(trainingImages);

	cout << "Adding training images to Network" << endl;
	net.addTrainingData(trainingImages,trueVals);

	cout << "Training Neural Network" << endl;
	net.train(100);

	cout << "Done" << endl;
	return 0;
}

int main(int argc, char** argv)
{
	firstCNN(argc,argv);
	//cout << "back in main" << endl;
	return 0;
}


















