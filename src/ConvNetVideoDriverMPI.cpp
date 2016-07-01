#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include "ConvNetCL.h"
#include <fstream>
#include <time.h>
#include <mpi.h>


//defines for MPI tags
#define REQUEST_FRAME 0
#define SEND_FRAME 1
#define SUBMIT_FRAME 2


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
size_t framesSent0 = 0;

bool done = false;

int stride, jump;
char* netName;
int inputWidth, inputHeight; // input width and height to the CNN
int fullWidth, fullHeight;   // width and height of the video frames

//Reading and writing
VideoCapture __video;
ofstream __outcsv;

bool firstGot0 = false;


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

void squareElements(vector<vector<vector<double> > >& vect)
{
	for(int i=0; i < vect.size(); i++)
		for(int j=0; j < vect[i].size(); j++)
			for(int k=0; k < vect[i][j].size(); k++)
				vect[i][j][k] = vect[i][j][k] * vect[i][j][k];
}

double vectorSum(const vector<double>& vect)
{
	double sum=0;
	for(int i=0; i<vect.size(); i++)
		sum += vect[i];
	return sum;
}

bool allElementsEqual(vector<double>& vect)
{
	for(int i=1; i < vect.size(); i++)
	{
		if(vect[0] != vect[i])
			return false;
	}
	return true;
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

void setAll3DVector(vector<vector<vector<double> > > &vect, double val)
{
	for(int i=0; i< vect.size(); i++)
		for(int j=0; j< vect[i].size(); j++)
			for(int k=0; k< vect[i][j].size(); k++)
				vect[i][j][k] = val;
}


unsigned int getNextFrame0()
{
	// pthread_mutex_lock(&frameMutex);
	//bool val = __video.read(frame);
	size_t outFrameNum;
	if(firstGot0)
	{
		bool val1 = true;
		for(int i=0; i < jump; i++)
			if(!__video.grab())
				val1 = false;
		//bool val2 = __video.retrieve(*(frame.mat));
		//frame.percentDone = __video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0;
		//frame.frameNum = curFrame;
		outFrameNum = curFrame0;
		curFrame0 += jump;
		// printf("got frame %d\n",curFrame);
		if(!val1)// || !val2) //this means the last frame was grabbed.
		{
			done = true;
		}
	}
	else // first frame only
	{
		//bool val = __video.read(*(frame.mat));
		bool val = __video.grab();
		//frame.percentDone = __video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0;
		// frame.frameNum = curFrame;
		outFrameNum = curFrame0;
		curFrame0+=jump;
		if(!val)
			done = true;
		firstGot0 = true;
		// printf("got first frame %d\n",curFrame-jump);
	}
	// pthread_mutex_unlock(&frameMutex);
	if(!done)
	{
		framesSent0++;
		printf("Giving frame %lu. %lf%%\n", outFrameNum, __video.get(CV_CAP_PROP_POS_AVI_RATIO)*100.0);
	}



	return outFrameNum;
}

bool getNextFrame(Frame& frame)
{
	// printf("getting next frame %d\n", my_rank);
	int rankArray[] = {my_rank};
	MPI_Send(rankArray, 1, MPI_INT, 0, REQUEST_FRAME, MPI_COMM_WORLD);

	frame.mat = new Mat(fullHeight,fullWidth,CV_8UC3);

	unsigned int receiveArray[2]; // {valid, frameNum}
	MPI_Recv(receiveArray, 2, MPI_UNSIGNED, 0, SEND_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if(receiveArray[0] == 0)
		return false;
	for(int i = curFrame; i <= receiveArray[1]; i++)
	{
		__video.grab();
	}
	__video.retrieve(*(frame.mat));
	frame.frameNum = receiveArray[1];
	curFrame = receiveArray[1];
	// printf("ending true next frame %d\n", my_rank);
	return true;

}

void manageNextFrames()
{
	int otherRank;
	unsigned int frameToSend; //this is a frame number
	unsigned int validity = 1;
	unsigned int send[2];
	while(!done)
	{
		//get next mpi msg from any source. they send their rank in the message
		MPI_Recv(&otherRank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		frameToSend = getNextFrame0();
		if(done)
			validity = 0;
		send[0] = validity;
		send[1] = frameToSend;

		MPI_Send(&send, 2, MPI_UNSIGNED, otherRank, SEND_FRAME, MPI_COMM_WORLD);
	}
	//once we break this loop there are comm_size - 2 processes that don't know we're done giving frames.
	//wait for the calls on each of these and send back stuff validity set to 0.
	for(int i = 0; i < comm_size - 2; i++)
	{
		MPI_Recv(&otherRank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&send, 2, MPI_UNSIGNED, otherRank, SEND_FRAME, MPI_COMM_WORLD);
	}
	//now everyone knows we're done and can start submitting.
}

void manageFrameSubmissions()
{
	// printf("start manage submit\n");
	waitingFrames0.resize(framesSent0);
	// printf("frames sent = %lu\n", framesSent0);
	double array[2]; //frameNum, red
	for(int i = 0; i < framesSent0; i++)
	{
		printf("%d\n", i);
		MPI_Recv(&array, 2, MPI_DOUBLE, MPI_ANY_SOURCE, SUBMIT_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		waitingFrames0[i] = new Frame();
		waitingFrames0[i]->frameNum = array[0];
		waitingFrames0[i]->red = array[1];
	}
	printf("start insert sort\n");
	// insertion sort because mostly sorted
	for(int i = 1; i < waitingFrames0.size(); i++)
	{
		unsigned int key = waitingFrames0[i]->frameNum;
		Frame* toInsert = waitingFrames0[i];
		int j = i - 1;
		while(j >= 0 && waitingFrames0[j]->frameNum > key)
		{
			waitingFrames0[j+1] = waitingFrames0[j];
			j--;
		}
		waitingFrames0[j+1] = toInsert;
	}
	printf("end insert sort\n");

	//Submit stuff to csv
	for(int i = 0; i < waitingFrames0.size(); i++)
	{
		__outcsv << waitingFrames0[i]->red << "," << waitingFrames0[i]->frameNum/10.0 << '\n';
		delete waitingFrames0[i];
	}

	// printf("end manage submit\n");
}

void storeFrame(Frame& frame)
{
	// printf("storing frame %d\n", my_rank);
	Frame *store = new Frame();
	store->frameNum = frame.frameNum;
	store->red = frame.red;
	store->mat = frame.mat;

	frame.mat = nullptr;

	waitingFrames.push_back(store);
	// printf("Storing frame %lu\n", frame.frameNum);
	// printf("ending storing frame %d\n", my_rank);
}

void submitFrames()
{
	printf("start submit\n");
	double array[2];
	for(int i = 0; i < waitingFrames.size(); i++)
	{
		array[0] = waitingFrames[i]->frameNum;
		array[1] = waitingFrames[i]->red;
		MPI_Send(array, 2, MPI_DOUBLE, 0, SUBMIT_FRAME, MPI_COMM_WORLD);
		delete waitingFrames[i]->mat;
		delete waitingFrames[i];
	}
	printf("end submit\n");
}

void breakUpImage(Net& net, Frame& frame)
{
	// printf("break up image\n");
	vector< vector< vector<double> > > fullImage;
	resize3DVector(fullImage, fullHeight, fullWidth, net.getNumClasses());
	setAll3DVector(fullImage, 0);

	int numrowsmin = fullHeight - inputHeight; // numrows minus input
	int numcolsmin = fullWidth  - inputWidth;  // numcols minus input
	int imageRowSize = 0;

	for(int j = 0; j <= numcolsmin; j += stride)
		imageRowSize++;
	// vector<Mat> imageRow(imageRowSize);
	// printf("image row size %d\n", imageRowSize);
	getchar();
	vector<Mat> imageRow(0);
	vector< vector<double> > confidences(0);

	for(int i = 0; i <= numrowsmin; i += stride)
	{
		// printf("start cv %d\n",my_rank);
		// cout << frame.mat << endl;
		for(int j = 0; j <= numcolsmin; j += stride)
		{
			const Mat out = (*(frame.mat))(Range(i, i+inputHeight),Range(j, j+inputWidth));
			// imageRow[j] = out;
			imageRow.push_back(out);
		}
		// printf("end cv %d\n",my_rank);

		// printf("start CL %d\n",my_rank);
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences);
		// printf("end CL %d\n",my_rank);

		int curImage = 0;
		for(int j = 0; j <= numcolsmin; j+=stride) //NOTE: each iteration of this loop is a different image
		{
			for(int ii = i; ii < i+inputHeight && ii < fullHeight; ii++)
				for(int jj = j; jj < j+inputWidth && jj < fullWidth; jj++)
					for(int cat = 0; cat < confidences[curImage].size(); cat++)
					{
						fullImage[ii][jj][cat] += confidences[curImage][cat];
						// printf("ii %d jj %d curImage %d cat %d\n", ii,jj,curImage,cat);
						// fullImage.at(ii).at(jj).at(cat) += confidences.at(curImage).at(cat);
					}
			curImage++;
		}
	}
	// printf("gone through frame\n");
	//at this point all subimages in the frame have been gone through
	squareElements(fullImage);

	Mat *outputMat = new Mat(fullHeight, fullWidth, CV_8UC3);

	frame.red = 0;
	double blue, red;
	for(int i = 0; i < fullHeight; i++)
	{
		for(int j = 0; j < fullWidth; j++)
		{
			double sumsq = vectorSum(fullImage[i][j]);
			for(int n = 0; n < fullImage[i][j].size(); n++)
				fullImage[i][j][n] /= sumsq;

			Vec3b& outPix = outputMat->at<Vec3b>(i,j);
			if(allElementsEqual(fullImage[i][j]))
			{
				outPix[0] = 0; outPix[1] = 0; outPix[2] = 0;
			}
			else
			{
				blue = 255*fullImage[i][j][0];
				red  = 255*fullImage[i][j][1];

				outPix[0] = blue; //blue pixel
				outPix[1] = 0;    //green pixel
				outPix[2] = red;  //red pixel

				if(red > 150)
					frame.red += (int)red;
			}
		}
	}

	delete frame.mat;
	frame.mat = outputMat;

	// printf("end break up image %d\n", my_rank);
	//frame.red is set in above loops
	//frame.frameNum shouldn't change
}

void breakUpVideo(char* videoPath)
{
	__video.open(videoPath);
	if(!__video.isOpened())
	{
		print0("Could not open video. Exiting.");
		return;
	}

	Net net(netName);
	//net.setConstantMem(true);

	// net.finalize is called below

	inputWidth = net.getInputWidth();
	inputHeight = net.getInputHeight();

	fullWidth = __video.get(CV_CAP_PROP_FRAME_WIDTH);
	fullHeight = __video.get(CV_CAP_PROP_FRAME_HEIGHT);

	if(__video.get(CV_CAP_PROP_FRAME_WIDTH) < inputWidth || __video.get(CV_CAP_PROP_FRAME_HEIGHT) < inputHeight)
	{
		if(my_rank == 0)
			printf("The video \"%s\" is too small in at least one dimension. Minimum size is %d x %d.\n",videoPath,inputWidth,inputHeight);
		return;
	}

	if(my_rank == 0)
	{
		char outNameCSV[255];
		string origName(videoPath);
		size_t dot = origName.rfind('.');
		const char *noExtension = origName.substr(0,dot).c_str();
		sprintf(outNameCSV,"%s_prediction.csv",noExtension);
		__outcsv.open(outNameCSV);

		//still need to do
		manageNextFrames();
		manageFrameSubmissions();

		__outcsv.close();
		return;
	}// rank 0 ends here (master)
	else 
	{// all other ranks (slaves)
		net.setDevice(0);
		// net.setConstantMem(true);
		if(!net.finalize())
			return;

		Frame frame;
		//frame.mat = new Mat(fullHeight, fullWidth, CV_8UC3);

		bool valid;
		while(true)
		{
			valid = getNextFrame(frame); // puts pointer to next frame in frame.mat
			if(!valid)
				break;
			breakUpImage(net, frame); // puts pointer to output frame in frame.mat
			storeFrame(frame);
		}
		submitFrames();
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
			print0("use format: ./ConvNetVideoDriverMPI cnnConfig.txt VideoOrFolderPath stride=<1> jump=<1>\n");
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

	MPI_Finalize();
}