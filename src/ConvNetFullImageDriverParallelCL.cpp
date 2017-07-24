/**
 * TODO: Convert to <thread> from <pthread.h> and then refactor
 */

#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include "ConvNetCL.h"
#include <ctype.h>
#include <fstream>
#include <time.h>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <unordered_map>
#include <cassert>
#include <random>
#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

#ifdef WIN32
# include <io.h>
#else
# include <unistd.h>
#endif

// #define _CNFIDPCL_DEBUG 0

using namespace cv;
using namespace std;
using namespace std::chrono;

mutex rowMutex;

int curFrame = 0;
int curRow = 0;
int curSubmittedFrame = 0;
// unsigned long cnn_time;

bool rgb=false;

typedef vector<vector<vector<double> > > imVector;

char *inPath;
int stride = 1;

Mat fullMat;

int numrowsmin, numcolsmin;
int numClasses;

vector<mutex> muxs(1);

vector<imVector> fullImages;
vector<Net*> nets;
vector<bool> deviceActive;
unordered_map<int, char> excludeDevices;

char* __netName;

vector<string> class_names;

int inputWidth, inputHeight;
int __rows, __cols;

bool __separate_outputs = false;

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

void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth)
{
	vect.resize(width);
	for(int i=0; i < width; i++)
	{
		vect[i].resize(height);
		for(int j=0; j < height; j++)
			vect[i][j].resize(depth);
	}
}

void setAll3DVector(vector<vector<vector<double> > > &vect, double val)
{
	for(int i=0; i < vect.size(); i++)
		for(int j=0; j < vect[i].size(); j++)
			for(int k=0; k < vect[i][j].size(); k++)
				vect[i][j][k] = val;
}


double vectorSum(const vector<double>& vect)
{
	double sum = 0;
	for(int i = 0; i < vect.size(); i++)
		sum += vect[i];
	return sum;
}

void squareElements(vector<vector<vector<double> > >& vect)
{
	for(int i = 0; i < vect.size(); i++)
		for(int j = 0; j < vect[i].size(); j++)
			for(int k = 0; k < vect[i][j].size(); k++)
				vect[i][j][k] = vect[i][j][k] * vect[i][j][k];
}

int getNextRow()
{
	#ifdef _CNFIDPCL_DEBUG
	printf("locking getNextRow\n");
	#endif
	lock_guard<mutex> guard(rowMutex);
	#ifdef _CNFIDPCL_DEBUG
	printf("locked\n");
	#endif
	int out = -1;
	if(curRow < fullMat.rows - inputHeight)
	{
		#ifndef _CNFIDPCL_DEBUG
		if(curRow != 0)
			printf("\33[2K\r"); //so we know where we are in an image but it doesn't stay on the terminal
		#endif
		out = curRow;
		curRow += stride;

		printf("Giving row %d of %d (%d)", out,__rows - inputHeight, __rows);
	}
	#ifdef _CNFIDPCL_DEBUG
	printf("returning getNextRow\n");
	#endif
	return out;

}

bool allElementsEquals(vector<double>& array)
{
	if(array.size() < 1)
		return true;
	for(int i=1; i < array.size(); i++)
	{
		if(array[0] != array[i])
			return false;
	}
	return true;
}

int getNumDevices()
{
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	return (int)deviceIdCount;
}

void combineImages()
{
	for(int d = 1; d < fullImages.size(); d++)
	{
		if(!deviceActive[d])
			continue;
		for(int i = 0; i < fullImages[d].size(); i++)
			for(int j = 0; j < fullImages[d][i].size(); j++)
				for(int k = 0; k < fullImages[d][i][j].size(); k++)
					fullImages[0][i][j][k] += fullImages[d][i][j][k];
	}
}

void __combineImagesThread(int d, int start)
{
	// uniform_int_distribution<int> dis(0,fullImages[d].size() - 1);
	// default_random_engine gen(time(NULL) * d);
	// int i = dis(gen);
	int i = start;
	for(int c = 0; c < fullImages[d].size(); c++)
	{
		lock_guard<mutex> guard(muxs[i]);
		for(int j = 0; j < fullImages[d][i].size(); j++)
			for(int k = 0; k < fullImages[d][i][j].size(); k++)
				fullImages[0][i][j][k] += fullImages[d][i][j][k];
		i = (i + 1) % fullImages[d].size();
	}
}

void combineImagesThreaded()
{
	static int firstActive = -1;
	if(firstActive == -1)
	{
		for(int d = 0; d < fullImages.size(); d++)
			if(deviceActive[d])
			{
				firstActive = d;
				break;
			}
		if(firstActive == -1)
		{
			printf("We do not appear to have any active devices. Exiting.\n");
			exit(1);
		}
	}

	if(muxs.size() != fullImages[firstActive].size())
	{
		vector<mutex> temp(fullImages[firstActive].size());
		muxs.swap(temp);
	}
	// printf("muxs size %lu vs %lu\n", muxs.size(), fullImages[0].size());
	int start = 0;
	int startInc = fullImages[firstActive].size() / fullImages.size();
	vector<thread> thr(fullImages.size());
	for(int d = 1; d < fullImages.size(); d++)
		if(deviceActive[d])
		{
			thr[d] = thread([=]{__combineImagesThread(d, start);});
			start += startInc;
		}
	for(int d = 1; d < fullImages.size(); d++)
		if(deviceActive[d])
			thr[d].join();
}


void breakUpRow(const int row, const int device)
{
	#ifdef _CNFIDPCL_DEBUG
	printf("start break up row\n");
	#endif
	const int i = row;
	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	//get all subimages from a row
	#ifdef _CNFIDPCL_DEBUG
	printf("get subimages\n");
	#endif
	for(int j=0; j<= numcolsmin; j+=stride) //NOTE: each j is a different subimage
	{
		assert(i + inputHeight <= fullMat.rows);
		assert(j + inputWidth <= fullMat.cols);
		imageRow.push_back((fullMat)(Range(i,i+inputHeight),Range(j,j+inputWidth)));
	}
	//set them as the data in the net
	#ifdef _CNFIDPCL_DEBUG
	printf("set data\n");
	#endif
	nets[device]->setData(imageRow);
	#ifdef _CNFIDPCL_DEBUG
	printf("run\n");
	#endif
	// auto starttime = system_clock::now();
	nets[device]->run();
	// auto endtime = system_clock::now();
	// cnn_time += duration_cast<chrono::microseconds>(endtime - starttime).count();
	#ifdef _CNFIDPCL_DEBUG
	printf("get confidences\n");
	#endif
	nets[device]->getConfidences(confidences); //gets the confidence for each category for each image
	assert(confidences.size() == imageRow.size());

	#ifdef _CNFIDPCL_DEBUG
	printf("add confidences into fullImages[%d]\n",device);
	#endif
	int curImage = 0;
	for(int j=0; j<= numcolsmin; j+=stride) //NOTE: each iteration of this loop is a different subimage
	{
		for(int ii=i; ii < i+inputHeight /*&& ii < __rows*/; ii++)
			for(int jj=j; jj < j+inputWidth /*&& jj < __cols*/; jj++)
				for(int cat = 0; cat < confidences[curImage].size(); cat++)
					fullImages[device][ii][jj][cat] += confidences[curImage][cat];
		curImage++;
	}
	assert(curImage == confidences.size());
	#ifdef _CNFIDPCL_DEBUG
	printf("end break up row\n");
	#endif
}

void __parallelImageRowProcessor(const int device)
{
	if(!deviceActive[device])
		return;
	do
	{
		int row = getNextRow();
		if(row < 0)
			break;
		breakUpRow(row, device);
	}while(true);
}

string getNameForVal(int trueVal)
{
	return class_names[trueVal];
}


/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
void breakUpImage(const char* imageName)
{
	// cnn_time = 0;
	#ifdef _CNFIDPCL_DEBUG
	printf("start breakUpImage\n");
	#endif
	//reset stuff
	curRow = 0;

	fullMat = imread(imageName,1);
	if(fullMat.empty())
	{
		printf("File '%s' was unable to open\n", imageName);
		return;
	}
	if(rgb)
		cvtColor(fullMat,fullMat,CV_BGR2RGB);

	__rows = fullMat.rows;
	__cols = fullMat.cols;

	for(int i = 0; i < nets.size(); i++)
	{
		if(i != 0 && !deviceActive[i])
			continue;
		resize3DVector(fullImages[i],__rows,__cols,numClasses);
		setAll3DVector(fullImages[i],0);
	}

	numrowsmin = __rows - inputHeight;
	numcolsmin = __cols - inputWidth;

	//calculate the rows in parallel.
	#ifdef _CNFIDPCL_DEBUG
	printf("starting __parallelImageRowProcessor\n");
	#endif
	thread* t = new thread[nets.size()];
	for(int i = 0; i < nets.size(); i++)
		t[i] = thread(__parallelImageRowProcessor, i);

	for(int i = 0; i < nets.size(); i++)
		t[i].join();
	#ifdef _CNFIDPCL_DEBUG
	printf("__parallelImageRowProcessor all joined\n");
	#endif

	delete[] t;

	// combineImages(); // combines into fullImages[0]
	#ifdef _CNFIDPCL_DEBUG
	printf("combine Images Threaded\n");
	#endif
	combineImagesThreaded();


	//process the data
	double sumsq;
	squareElements(fullImages[0]);
	for(int i=0; i < __rows; i++)
	{
		for(int j=0; j < __cols; j++)
		{
			sumsq = vectorSum(fullImages[0][i][j]);
			// assert(sumsq != 0);
			if(sumsq != 0)
				for (int k = 0; k < numClasses; k++)
					fullImages[0][i][j][k] /= sumsq;
		}
	}

	printf("\33[2K\r");
	if(__separate_outputs)
	{
		//make the output Mats
		vector<Mat*> outputMats(numClasses);
		for(int k = 0; k < numClasses; k++)
		{
			outputMats[k] = new Mat(__rows,__cols,CV_8UC3);
		}
		//calculate what output images should look like
		for(int k = 0; k < numClasses; k++)
		{
			for(int i=0; i < __rows; i++)
			{
				for(int j=0; j < __cols; j++)
				{
					//write the pixel
					Vec3b& outPix = outputMats[k]->at<Vec3b>(i,j);

					double pix = 255 * fullImages[0][i][j][k];
					outPix[0] = pix;  // blue
					outPix[1] = pix;  //green
					outPix[2] = pix;  // red
				}
			}
		}

		//output the mats
		char outName[500];
		string origName(imageName);
		size_t dot = origName.rfind('.');
		const string noExtension = origName.substr(0,dot);
		const string extension = origName.substr(dot+1);

		
		for(int k = 0; k < numClasses; k++)
		{
			sprintf(outName,"%s_prediction_%s.%s",noExtension.c_str(), getNameForVal(k).c_str(), extension.c_str());
			// if(infos.size() > k)
			// 	sprintf(outName,"%s_prediction_%s.%s",noExtension.c_str(), getNameForVal(k).c_str(), extension.c_str());
			// else
			// 	sprintf(outName,"%s_prediction_class%d.%s",noExtension.c_str(),k,extension.c_str());

			printf("Writing %s\n", outName);
			imwrite(outName,*(outputMats[k]));
		}

		//cleanup memory
		for(int m = 0; m < numClasses; m++)
			delete outputMats[m];
	}
	else
	{
		unsigned long depth = fullImages[0][0][0].size();
		Mat outputMat(__rows, __cols, CV_8UC3);
		for(int k = 0; k < numClasses; k++)
		{
			for(int i=0; i < __rows; i++)
			{
				for(int j=0; j < __cols; j++)
				{
					//write the pixel
					Vec3b& outPix = outputMat.at<Vec3b>(i,j);

					outPix[0] = 255 * fullImages[0][i][j][0];  // blue
					outPix[2] = 255 * fullImages[0][i][j][1];  // red

					if(depth > 2)
						outPix[1] = 255 * fullImages[0][i][j][2];
					else
						outPix[1] = 0;
				}
			}
		}

		char outName[500];
		string origName(imageName);
		size_t dot = origName.rfind('.');
		const char *noExtension = origName.substr(0,dot).c_str();
		const char *extension = origName.substr(dot).c_str();

		sprintf(outName,"%s_prediction%s",noExtension,extension);
		printf("Writing %s\n", outName);
		imwrite(outName, outputMat);
	}
	#ifdef _CNFIDPCL_DEBUG
	printf("end breakUpImage\n");
	#endif
}

int checkExtensions(char* filename)
{
	string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	if(name.rfind(".JPG")  == name.length() - 4) return 1;
	if(name.rfind(".JPEG") == name.length() - 5) return 1;
	if(name.rfind(".PNG")  == name.length() - 4) return 1;
	return 0;
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("Usage (Required to come first):\n ./ConvNetFullImageDriverParallelCL cnnFile.txt ImageOrFolderPath\n");
		printf("Optional args (must come after required args. Case sensitive.):\n");
		printf("   stride=<int>          Stride across image. Defaults to 1.\n");
		printf("   -separate_outputs     Puts the prediction for each class in a separate image. Default for Nets with more than 3 classes.\n");
		printf("   -rgb                  Has ConvNetCL read in the Mats as RGB instead of OpenCV standard BGR.\n");
		printf("   -excludeDevice=<int>  Excludes the specified OpenCL device from use. Repeatable.\n");
		return -1;
	}
	// chrono::system_clock::time_point starttime, endtime;
	time_t starttime, endtime;
	inPath = argv[2];

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("stride=") != string::npos)
				stride = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("-separate_outputs") != string::npos)
				__separate_outputs = true;
			else if(arg.find("-rgb") != string::npos)
				rgb = true;
			else if(arg.find("-excludeDevice=") != string::npos)
				excludeDevices[stoi(arg.substr(arg.find('=')+1))] = 1;
			else
			{
				printf("Unknown arg \"%s\". Aborting.\n", argv[i]);
				return 0;
			}
		}
	}

	assert(stride > 0);

	// printf("jump %d stride %d\n", jump, stride);

	__netName = argv[1];
	//set up net
	//Net net(argv[1]);
	//if(!net.isActive())
		//return 0;


	//go through all images in the folder
	bool isDirectory;
	struct stat s;
	if(stat(inPath,&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
			isDirectory = true;
		else if (s.st_mode & S_IFREG) // file
			isDirectory = false;
		else
		{
			printf("We're not sure what the file you inputted was.\nExiting\n");
			return -1;
		}
	}
	else
	{
		printf("Error getting status of folder or file.\nExiting\n");
		return -1;
	}

	vector<string> filenames(0);
	
	if(isDirectory)
	{
		DIR *directory;
		struct dirent *file;
		if((directory = opendir(inPath)))// != NULL)
		{
			string pathName = inPath;
			if(pathName.rfind("/") != pathName.length()-1)
				pathName.append(1,'/');
			char inPathandName[250];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					if(checkExtensions(file->d_name))
					{
						sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
						string ipan(inPathandName);
						if(ipan.find("_prediction") == string::npos)
						{
							//breakUpImage(inPathandName, net);
							filenames.push_back(ipan);
						}
					}
				}
			}
			//cout << "closing directory" << endl;
			closedir(directory);
			//cout << "directory closed" << endl;
		}
	}
	else
	{
		if(checkExtensions(inPath))
		{
			//breakUpImage(inPath, net);
			string ip(inPath);
			filenames.push_back(ip);
		}
	}

	printf("Found %lu image(s).\n", filenames.size());

	setbuf(stdout,0);

	//init all nets
	vector<thread> thr(getNumDevices());
	nets.resize(thr.size());
	deviceActive.resize(nets.size());
	for(int i = 0; i < thr.size(); i++)
	{
		if(excludeDevices.find(i) != excludeDevices.end())
		{
			deviceActive[i] = false;
			continue;
		}

		Net** loc = &(nets[i]);
		thr[i] = thread([=] { *loc = new Net(__netName); });
		deviceActive[i] = true;
		// nets[i] = new Net(__netName);
	}
	for(int i = 0; i < thr.size(); i++)
		if(deviceActive[i])
			thr[i].join();
	
	printf("joined\n");
	fullImages.resize(nets.size());

	int usable = -1;
	for(int i = 0; i < nets.size(); i++)
		if(deviceActive[i])
			usable = i;
	if(usable == -1)
	{
		printf("No usable nets were able to be made on non-excluded devices\n");
		return 0;
	}

	

	inputHeight = nets[usable]->getInputHeight();
	inputWidth = nets[usable]->getInputWidth();
	numClasses = nets[usable]->getNumClasses();
	if(numClasses > 3)
		__separate_outputs = true;

	nets[usable]->getClassNames(class_names);

	printf("Getting devices\n");
	//get the ones that work
	for(int i = 0 ; i < nets.size(); i++)
	{
		if(!deviceActive[i])
			continue;
		else if(nets[i]->setDevice(i) && nets[i]->finalize())
		{
			deviceActive[i] = true;
			printf("Thread using device %d\n",i);
		}
		else
		{
			deviceActive[i] = false;
			delete nets[i];
		}
	}

	usable = -1;
	for(int i = 0; i < nets.size(); i++)
		if(deviceActive[i])
			usable = i;
	if(usable == -1)
	{
		printf("No usable nets were able to be made and finalized on non-excluded devices\n");
		return 0;
	}

	for(int i=0; i < filenames.size(); i++)
	{
		starttime = time(NULL);
		// starttime = chrono::system_clock::now();
		cout << filenames[i] << " (" << i + 1 << " of " << filenames.size() << ")" << endl;
		breakUpImage(filenames[i].c_str());
		endtime = time(NULL);
		// endtime = chrono::system_clock::now();
		cout << " - Time for image: " << secondsToString(endtime - starttime) << endl;
		// cout << " - Time for image: " << (chrono::duration_cast<chrono::milliseconds>(endtime - starttime).count()) / 1000.0 << " seconds" << endl;
		// cout << "\t" << cnn_time / 1000000.0 << " seconds was in the CNN" << endl;
	}

	return 0;
	
}