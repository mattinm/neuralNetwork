//CNN code
#include "ConvNetCL.h"

//general
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//for looking at files and directories
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

//BOINC
#ifdef _BOINC_APP_
#include "diagnostics.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "proc_control.h"
#endif

using namespace cv;
using namespace std;

typedef vector<vector<vector<double> > > imVector;

const char* inPath;
char *outPath;
int stride = 1;

int cnnWidth, cnnHeight;

vector<string> filenames;
int curImage = 0;
int curRow = 0;
bool readimVector = false;


//BOINC FUNCTIONS
std::string getBoincFilename(std::string filename) throw(std::runtime_error) {
    std::string resolved_path = filename;
	#ifdef _BOINC_APP_
	    if(boinc_resolve_filename_s(filename.c_str(), resolved_path)) {
	        printf("Could not resolve filename %s\n",filename.c_str());
	        throw std::runtime_error("Boinc could not resolve filename");
	    }
	#endif
    return resolved_path;
}

void writeCheckpoint(imVector& fullImage) throw(std::runtime_error)
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	FileStorage outfile(resolved_checkpoint_name, FileStorage::WRITE);
	if(!outfile.isOpened())
		throw std::runtime_error("Checkpoint file could not be opened for writing.");

	printf("Writing Checkpoint: image: %d, last row completed: %d\n", curImage, curRow);

	outfile << "CUR_IMAGE" << curImage;
	outfile << "CUR_ROW" << curRow;

	outfile << "FILENAMES" << "[:";
	for(int i = 0; i < filenames.size(); i++)
		outfile << filenames[i].c_str();
	outfile << "]";

	int numrows = fullImage.size();
	int numcols = fullImage[0].size();
	int depth = fullImage[0][0].size();
	outfile << "NUMROWS" << numrows;
	outfile << "NUMCOLS" << numcols;
	outfile << "DEPTH" << depth;

	outfile << "IMVECTOR" << "[:"
	for(int i =0; i < fullImage.size(); i++)
		for(int j = 0; j < fullImage[i].size(); j++)
			for(int k = 0; k < fullImage[i][j].size(); k++)
				outfile << fullImage[i][j][k];
	outfile << "]";

	outfile.release();
}

bool readCheckpoint()
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	FileStorage infile(resolved_checkpoint_name, FileStorage::READ);
	if(!infile.isOpened())
		return false;

	infile["CUR_IMAGE"] >> curImage;
 
	return true;
}

void readSavedimVector(imVector& fullImage);
{

}
//END - BOINC FUNCTIONS

//VECTOR FUNCTIONS
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
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = val;
			}
		}
	}
}
//END - VECTOR FUNCTIONS

//FUNCTIONS FOR READING FILES AND DIRECTORIES
int checkExtensions(const char* filename)
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

bool getFiles(const char* inPath)
{
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
			return false;
		}
	}
	else
	{
		printf("Error getting status of folder or file.\nExiting\n");
		return false;
	}
	
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
							filenames.push_back(ipan);
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
			string ip(inPath);
			filenames.push_back(ip);
		}
	}
	return true;
}
// END - FUNCTIONS FOR READING FILES AND DIRECTORIES

//FUNCTIONS USED TO BREAK UP AN IMAGE
bool allElementsEquals(vector<double>& array)
{
	for(int i=1; i < array.size(); i++)
	{
		if(array[0] != array[i])
			return false;
	}
	return true;
}

void squareElements(vector<vector<vector<double> > >& vect)
{
	for(int i=0; i < vect.size(); i++)
		for(int j=0; j < vect[i].size(); j++)
			for(int k=0; k < vect[i][j].size(); k++)
				vect[i][j][k] = vect[i][j][k] * vect[i][j][k];
}

void breakUpImage(const char* imageName, Net& net)
{
	//cout << "starting breakUpImage" << endl;
	Mat image = imread(imageName,1);
	int numrows = image.rows;
	int numcols = image.cols;
	printf("%s rows: %d, cols: %d\n",imageName, numrows,numcols);


	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	int numClasses = net.getNumClasses();
	if(readimVector)
		readSavedimVector(fullImage)
	else
	{
		resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
		setAll3DVector(fullImage,0);
	}

	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsmcnn = numrows - cnnHeight;
	int numcolsmcnn = numcols - cnnWidth;
	if(numrows < inputHeight || numcols < inputWidth)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is %dx%d.\n",imageName,inputHeight,inputWidth);
		return;
	}
	for( ; curRow <= numrowsmcnn; curRow += stride) //curRow will be 0 unless set by checkpoint
	{
		imageRow.resize(0);
		printf("row %d of %d (%d)\n",i,numrowsm32,numrows);

		//get all subimages from a row
		for(int j=0; j<= numcolsmcnn; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(i,i+inputHeight),Range(j,j+inputWidth));
			imageRow.push_back(out);
		}

		//set them as the data in the net
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences); //gets the confidence for each category for each image

		int localCurImage = 0;
		for(int j=0; j<= numcolsmcnn; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=i; ii < i+inputHeight && ii < numrows; ii++)
				for(int jj=j; jj < j+inputHeight && jj < numcols; jj++)
					for(int cat = 0; cat < confidences[localCurImage].size(); cat++)
						fullImage[ii][jj][cat] += confidences[localCurImage][cat];
			localCurImage++;
		}

		//update fraction done and see if we need to checkpoint
		boinc_fraction_done(0);
		if(boinc_time_to_checkpoint())
		{
			writeCheckpoint(fullImage);
			boinc_checkpoint_completed();
		}
	}
	curRow = 0; // this is set here so the next image will be right.

	//now we have the confidences for every pixel in the image
	//so get the category for each pixel and make a new image from it
	squareElements(fullImage);
	vector<Mat*> outputMats;
	for(int k = 0; k < numClasses; k++)
	{
		for(int i=0; i < numrows; i++)
		{
			for(int j=0; j < numcols; j++)
			{
				//square ratios
				double sumsq = vectorSum(fullImage[i][j]);
				for(int n=0; n < fullImage[i][j].size(); n++)
					fullImage[i][j][n] /= sumsq;

				//write the pixel
				Vec3b& outPix = outputMat.at<Vec3b>(i,j);
				if(allElementsEquals(fullImage[i][j]))
				{
					outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
				}
				else
				{
					double white = 255 * fullImage[i][j][k];
					outPix[0] = white; // blue
					outPix[1] = white; // green
					outPix[2] = white; // red
				}
			}
		}
	}
	char outName[255];
	string origName(imageName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	const char *extension = origName.substr(dot).c_str();

	sprintf(outName,"%s_prediction%s",noExtension,extension);
	cout << "writing " << outName << endl;
	imwrite(outName, outputMat);
}
//END - FUNCTIONS USED TO BREAK UP AN IMAGE

int main(int argc, char** argv)
{
	if(argc < 3  || 5 < argc)
	{
		printf("use format: ./ConvNetFullImageDriverCL_BOINC cnnConfig.txt imageOrFolderPath (stride=1) (device=0)\n");
		return -1;
	}

	printf("Initializing BOINC\n");

	#ifdef _BOINC_APP_
	boinc_init_diagnostics(BOINC_DIAG_MEMORYLEAKCHECKENABLED);

	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.multi_thread = true;  // for multiple threads in OpenCL
	options.multi_process = true; // for multiple processes in OpenCL?
	options.normal_thread_priority = true; // so GPUs will run at full speed
	boinc_init_options(&options);
	boinc_init();
	#endif

	inPath = getBoincFilename(string(argv[2])).c_str();

	int device = 0;
	for(int i = 3; i < argc; i++)
	{
		string arg(argv[i]);
		if(arg.find("stride=") != string::npos)
			stride = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
	}

	string netPath = getBoincFilename(argv[1]);
	Net net(netPath.c_str());

	cnnWidth = net.getInputWidth();
	cnnHeight = net.getInputHeight();

	if(!net.setDevice(device) || !net.finalize())
	{
		printf("Net was unable to finalize on device %d\n", device);
		boinc_finish(-1);
		return -1;
	}

	bool noCheckpoint = true;

	#ifdef _BOINC_APP_
	if(readCheckpoint())
	{
		printf("Continuing from Checkpoint\n");
		noCheckpoint = false;
		readimVector = true;
	}
	else
	{
		printf("No Checkpoint found. Starting from beginning\n");
	}
	#endif

	if(noCheckpoint) // otherwise files will be put in vector by readCheckpoint
	{
		bool success = getFiles(inPath);
		if(!success)
		{
			printf("Error getting images to run over.\n");
			boinc_finish(-1);
			return -1;
		}
	}

	for(int i = curImage; i < filenames.size(); i++)
	{
		breakUpImage(filenames[i].c_str(), net);
	}

	boinc_finish(0);
	return 0;

}





