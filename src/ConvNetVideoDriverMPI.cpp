#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include "ConvNetCL.h"
#include <fstream>
#include <time.h>
#include <mpi.h>


using namespace cv;
using namespace std;

struct Frame
{
	Mat* mat;
	size_t frameNum;
	size_t red;
};

//all variables ending with 0 are only to be used by rank 0
size_t curFrame0 = 0;
size_t curFrame = 0;

int stride, jump;
char* netName;
int inputWidth, inputHeight;

//Reading and writing
VideoCapture __video;
ofstream __outcsv;


vector<Frame*> waitingFrames0(0);
vector<Frame*> waitingFrames(0);

//MPI variables
int my_rank;
int comm_size;

void print0(const char* message)
{
	if(my_rank == 0)
		printf("%s\n", message);
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

void breakUpVideo(char* videoPath)
{
	__video.open(videoPath);
	if(!__video.isOpened())
	{
		print0("Could not open video. Exiting.");
	}

	Net net(netName);
	//net.setConstantMem(true);

	inputWidth = net.getInputWidth();
	inputHeight = net.getInputHeight();

	if(__video.get(CV_CAP_PROP_FRAME_WIDTH) < inputWidth || __video.get(CV_CAP_PROP_FRAME_HEIGHT) < inputHeight)
	{
		if(my_rank == 0)
			printf("The video \"%s\" is too small in at least one dimension. Minimum size is %d x %d.\n",videoPath,inputWidth,inputHeight);
		return;
	}


}

int main(int argc, char** argv)
{
	time_t starttime, endtime;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if(argc < 3 || 5 < argc)
	{
		if(my_rank == 0)
			print0("use format: ./ConvNetVideoDriver cnnConfig.txt VideoOrFolderPath stride=<1> jump=<1>\n");
		MPI_Finalize();
		return 0;
	}
	if(comm_size < 2)
	{
		if(my_rank == 0)
			print0("Must have at least 2 processes. Exiting\n");
		MPI_Finalize();
		return 0;
	}

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("stride=") != string::npos)
				stride = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("jump=") != string::npos)
				jump = stoi(arg.substr(arg.find("=")+1));
			else
			{
				if(my_rank == 0)
					printf("Unknown arg \"%s\". Aborting.\n", argv[i]);
				return 0;
			}
		}
	}

	netName = argv[1];


	starttime = time(NULL);
	curFrame = 0;
	breakUpVideo(argv[2]);
	endtime = time(NULL);
	if(my_rank == 0)
		printf("Time for Video \"%s\": %s\n",argv[2], secondsToString(endtime - starttime).c_str());
}