
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "ConvNet.h"
#include <ctype.h>
#include <fstream>
#include <time.h>
#include <thread>
#include <pthread.h>
#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif


using namespace cv;
using namespace std;

pthread_mutex_t frameMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t submitMutex = PTHREAD_MUTEX_INITIALIZER;

int curFrame = 0;
int curSubmittedFrame = 0;

typedef vector<vector<vector<double> > > imVector;

struct Frame
{
	int frameNum;
	Mat mat;
};

vector<Frame*> waitingFrames(0);

char *inPath, *outPath;
int imageNum = 0;
int stride = 1;
bool __useGPU = true;

VideoCapture __video;
VideoWriter __outVideo;

//puts frame and frameNum in parameters
void getNextFrame(Mat& frame, unsigned int& frameNum)
{
	pthread_mutex_lock(&frameMutex);
	__video.read(frame);
	frameNum = count++;
	pthread_mutex_unlock(&frameMutex);
	//printf("Frame %ld of %.0lf\n", ++count, __video.get(CV_CAP_PROP_FRAME_COUNT));
	//breakUpImage(frame, net, outVideo);
}

void submitFrame(Mat& frame, unsigned int frameNum)
{
	pthread_mutex_lock(&submitMutex);
	if(frameNum = curSubmittedFrame)
	{
		__outVideo << frame;
		curSubmittedFrame++;
		while()
	}
	else
	{
		Frame *newframe = new Frame;
		newframe->mat = frame; newframe->frameNum
		waitingFrames.resize(waitingFrames.size() + 1);
		int i=0;
		while(waitingFrames[i].frameNum < frameNum)
			i++;
		for(int j=waitingFrames.size()-1; j >= i+1; j--)
		{
			waitingFrames[j] = waitingFrames[j-1];
		}
		waitingFrames[i] = newframe;

	}
	pthread_mutex_unlock(&submitMutex);
}

bool allElementsEquals(vector<double>& array)
{
	for(int i=1; i < array.size(); i++)
	{
		if(array[0] != array[i])
			return false;
	}
	return true;
}

int getNumDevices()
{
	cl_int error;
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	return (int)deviceIdCount
}

void _t_convertColorMatToVector(const Mat& m , vector<vector<vector<double> > > &dest, int row)
{
	for(int j=0; j< m.cols; j++)
	{
		const Vec3b& curPixel = m.at<Vec3b>(row,j);
		dest[row][j][0] = curPixel[0];
		dest[row][j][1] = curPixel[1];
		dest[row][j][2] = curPixel[2];
	}
}

void convertColorMatToVector(const Mat& m, vector<vector<vector<double> > > &dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}

	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	resize3DVector(dest,width2,height2,depth2);
	thread *t = new thread[width2];
	
	for(int i=0; i< width2; i++)
	{
		t[i] = thread(_t_convertColorMatToVector,ref(m),ref(dest),i);
	}

	for(int i=0; i< width2; i++)
	{
		t[i].join();
	}

	//delete t;
}

/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
void breakUpImage(Mat& image, Net& net, VideoWriter& outVideo)
{
	//cout << "starting breakUpImage" << endl;
	//Mat image = imread(imageName,1);
	int numrows = image.rows;
	int numcols = image.cols;
	//printf("%s rows: %d, cols: %d\n",imageName, numrows,numcols);
	int length = 0;
	char tempout[255];

	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	resize3DVector(fullImage,numrows,numcols,net.getNumCategories());
	setAll3DVector(fullImage,0);
	vector<imVector> imageRow(0); // this will hold all subimages from one row
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsm32 = numrows-32;
	int numcolsm32 = numcols-32;

	for(int i=0; i <= numrowsm32; i+=stride)
	{
		imageRow.resize(0);
		if(i != 0)
		{
			//cout << string(length,'\b');
		}
		//sprintf(tempout,"row %d of %d (%d)\n",i,numrowsm32,numrows);
		string tempstring(tempout); length = tempstring.length();
		//cout << "row " << i << " of " << numrows << endl;
		//cout << tempout;
		//get all subimages from a row
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(i,i+32),Range(j,j+32));
			//if((i == 0 || i == numrows-32) && j== 0)
				//cout << out << endl << endl;
			//printf("i: %d, j: %d\n",i,j);
			imageRow.resize(imageRow.size()+1);
			convertColorMatToVector(out,imageRow.back());
		}
		//set them as the data in the net
		preprocess(imageRow);
		net.setData(imageRow);
		net.newRun(calcedClasses, __useGPU);
		net.getConfidences(confidences); //gets the confidence for each category for each image
		//if((i == 0 || i == numrows-32))
		//cout << "row: " << i << " ";
		//printVector(confidences[0]);
		//cout << "conf got" << endl;
		int curImage = 0;
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=i; ii < i+32 && ii < numrows; ii++)
			{
				for(int jj=j; jj < j+32 && jj < numcols; jj++)
				{
					for(int cat = 0; cat < confidences[curImage].size(); cat++)
					{
						//printf("%d %d %d %d\n",i,j,jj,cat);
						fullImage[ii][jj][cat] += confidences[curImage][cat];
					}
				}
			}
			curImage++;
		}
	}
	//cout << endl;
	//printVector(fullImage);

	//now we have the confidences for every pixel in the image
	//so get the category for each pixel and make a new image from it
	Mat outputMat(numrows,numcols,CV_8UC3);
	for(int i=0; i < numrows; i++)
	{
		for(int j=0; j < numcols; j++)
		{
			Vec3b& outPix = outputMat.at<Vec3b>(i,j);
			int maxEle = getMaxElementIndex(fullImage[i][j]);
			if(allElementsEquals(fullImage[i][j]))
			{
				outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
			}
			else if(maxEle == 0)
			{
				outPix[0] = 255; outPix[1] = 0; outPix[2] = 0; // blue
			}
			else if(maxEle == 1)
			{
				outPix[0] = 0; outPix[1] = 0; outPix[2] = 255; // red
			}
		}
	}
	

	outVideo << outputMat;
}

void breakUpVideo(const char* videoName, Net& net)
{
	__video.open(videoName);
	if(!__video.isOpened())
	{
		cout << "Could not open video: " << videoName << endl;
		return;
	}

	if(__video.get(CV_CAP_PROP_FRAME_WIDTH) < 32 || __video.get(CV_CAP_PROP_FRAME_HEIGHT) < 32)
	{
		printf("The video %s is too small in at least one dimension. Minimum size is 32x32.\n",videoName);
		return;
	}



	char outName[255];
	string origName(videoName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	const char *extension = origName.substr(dot).c_str();

	sprintf(outName,"%s_prediction%s",noExtension,extension);
	//cout << "writing " << outName << endl;

	__outVideo.open(outName, 
	 video.get(CV_CAP_PROP_FOURCC),
	 video.get(CV_CAP_PROP_FPS), 
	 Size(__video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	int numDevices = getNumDevices();
	for(int i=0; i < numDevices; i++)
	{
		//start new thread for each device. Any thread for a device that does not support double will return early.

	}


	/*
	Mat frame;
	unsigned long count = 0;
	while(__video.read(frame))
	{
		printf("Frame %ld of %.0lf\n", ++count, __video.get(CV_CAP_PROP_FRAME_COUNT));
		breakUpImage(frame, net, outVideo);
	}
	/**/
}

int checkExtensions(char* filename)
{
	string name = filename;
	if(name.rfind(".avi")  == name.length() - 4) return 1;
	if(name.rfind(".mpeg") == name.length() - 5) return 1;
	if(name.rfind(".mp4")  == name.length() - 4) return 1;
	return 0;
}

int main(int argc, char** argv)
{
	if(argc < 3 || 5 < argc)
	{
		printf("use format: ./ConvNetVideoDriver cnnConfig.txt VideoOrFolderPath (stride=1) (gpu=true)\n");
		return -1;
	}

	inPath = argv[2];

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("stride=") != string::npos)
			{
				stride = stoi(arg.substr(arg.find("=")+1));
			}
			else if(arg.find("gpu=") != string::npos)
			{
				if(arg.find("false") != string::npos || arg.find("False") != string::npos)
				{
					__useGPU = false;
				}
			}
		}
	}


	//set up net
	Net net(argv[1]);
	if(!net.isActive())
		return 0;



	//go through all images in the folder
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

	for(int i=0; i < filenames.size(); i++)
	{
		breakUpVideo(filenames[i].c_str(),net);
	}

	//cout << "returning" << endl;
	return 0;
	
}