/***************************************
*
*
*	TrainingImageSplitterFileCreator takes a set of input images of specified classes, splits them into specified sized subimages and converts the subimages to a single binary file.
*
* 	All input images must be of the same size, specified in command line arguments. The format of the
*	output file is this:
*
*		sizeByte xsize ysize zsize image1 trueVal1 image2 trueVal2... imageN trueValN
*		short	 short short short 		  ushort		  ushort
*
*	The sizeByte says how large/what type each input is. 
*		1 - unsigned byte 		-1 - signed byte
*		2 - unsigned short 		-2 - signed short
* 		4 - unsigned int 		-4 - signed int
* 		5 - float
*		6 - double
*
*
*
*	Usage: 
*		Usage: ./TrainingImageSplitterFileCreator ImageConfigFile outfileName <stride=1>
*			stride is optional, defaults to 1.
*
*	ImageConfigFile format:
*		inWidth inHeight inDepth
*		sizeByte
*		folderOfImages1,trueVal1,<stride=1>
*		folderOfImages2,trueVal2,<stride=1>
*		...
*
*		stride parameter is optional for each folder. it will override the command line parameter stride for that folder
*
*
****************************************/


#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unordered_map>

using namespace std;
using namespace cv;

short xsize = -1, ysize = -1, zsize = -1;

int imageNum = 0;
int stride = 1;
int globalStride = 1;
long byteCount = 0;
long imageCount = 0;
unsigned short __globalTrueVal;
unordered_map<unsigned short, int> trueMap;

template<typename type>
void writeImage(Mat& image, ofstream& outfile)
{
	//Mat image = imread(inPathandName,1); //color image
	//cout << image << endl;
	//printf("x %d, y %d, z %d, ir %d, ic %d id %d",xsize,ysize,zsize,image.rows,image.cols,image.depth());
	if(image.rows != ysize || image.cols != xsize)// || image.depth() != zsize)
		return;
	//cout << "after return" << endl;

	type pixel[3];
	long size = sizeof(type) * 3;

	if(image.type() == CV_8UC3)
	{
		for(int i=0; i < xsize; i++)
		{
			for(int j=0; j < ysize; j++)
			{
				const Vec3b& curPixel = image.at<Vec3b>(i,j);
				pixel[0] = (type)curPixel[0];
				pixel[1] = (type)curPixel[1];
				pixel[2] = (type)curPixel[2];

				//cout << "writing" << endl;
				outfile.write(reinterpret_cast<const char *>(pixel),size);
			}
		}
		outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(unsigned short));
		unordered_map<unsigned short, int>::const_iterator got = trueMap.find(__globalTrueVal);
		if(got == trueMap.end()) // not found
			trueMap[__globalTrueVal] = 1;
		else // found
			trueMap[__globalTrueVal]++;
		byteCount += xsize * ysize * 3 * sizeof(type) + sizeof(unsigned short); //extra 2 for the ushort trueVal
		imageCount++;
		if(imageCount % 100000 == 0)
		{
			printf("Images: %ld, GB: %lf\n", imageCount, byteCount/1.0e9);
			if(byteCount/1.0e9 > 30)
			{
				outfile.close();
				exit(0);
			}
		}
	}
	else
	{
		cout << "Unsupported image type" << endl;
	}
	
}

template<typename type>
void breakUpImage(const char* imageName, ofstream& outfile)
{
	Mat image = imread(imageName,1);
	int numThisImage = 0;
	int numrows = image.rows;
	int numcols = image.cols;
	printf("%s rows: %d, cols: %d.     ",imageName, numrows,numcols);
	if(numrows < ysize || numcols < xsize)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is %dx%d.\n",imageName,xsize,ysize);
		return;
	}

	cout << "Breaking with stride = " << stride << " and true val " << __globalTrueVal << endl;
	for(int i=0; i <= numrows-ysize; i+=stride)
	{
		for(int j=0; j<= numcols-xsize; j+=stride)
		{
			Mat out = image(Range(i,i+ysize),Range(j,j+xsize));
			writeImage<type>(out,outfile);
			numThisImage++;
			imageNum++;
		}
	}
	printf("%d images created.\n", numThisImage);	
}

int checkExtensions(const char* filename)
{
	const string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}



template<typename type>
void getImages(const char* folder, ofstream& outfile)
{
	cout << "Getting images from " << folder << endl;
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
						//cout << "going to write an image" << endl;
						//writeImage<type>(inPathandName,outfile);
						breakUpImage<type>(inPathandName,outfile);
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
			breakUpImage<type>(inPath,outfile);
			//writeImage<type>(inPath,outfile);
		}

	}
}

int main (int argc, char** argv)
{
	if(argc != 3 && argc != 4)
	{
		cout << "Usage: ./TrainingImageSplitterFileCreator ImageConfigFile outfileName <stride=1>\nstride is optional, defaults to 1." << endl;
		return 0;
	}

	if(argc == 4)
	{
		string str(argv[3]);
		if(str.find("=") != string::npos)
		{
			globalStride = stoi(str.substr(str.find("=")+1));
		}
		else
		{
			globalStride = stoi(str);
		}
	}

	ifstream imageConfig;
	ofstream outfile;
	imageConfig.open(argv[1]);
	outfile.open(argv[2], ios::trunc | ios::binary);
	if(!imageConfig.is_open())
	{
		cout << "Could not open the ImageConfigFile" << endl;
		return -1;
	}
	string line;

	//get image sizes
	getline(imageConfig,line);

	int locx = line.find(" ");
	xsize = stoi(line.substr(0,locx));
	int locy = line.find(" ",locx+1);
	ysize = stoi(line.substr(locx+1, locy));
	int locz = line.find(" ",locy+1);
	zsize = stoi(line.substr(locy+1,locz));

	//get sizeByte
	getline(imageConfig,line);
	short sizeByte = stoi(line);

	cout << "SizeByte: " << sizeByte << " x: " << xsize << " y: " << ysize << " z: " << zsize << endl;

	/*
	double testd = 559.236;
	int testi = -400001;
	unsigned int testui = 5098;
	unsigned short testus = 4098;
	char testc = -120;
	unsigned char testuc = 230;
	float testf = -4.59;
	outfile.write(reinterpret_cast<const char *>(&testd),sizeof(double));
	outfile.write(reinterpret_cast<const char *>(&testi),sizeof(int));
	outfile.write(reinterpret_cast<const char *>(&testui),sizeof(unsigned int));
	outfile.write(reinterpret_cast<const char *>(&testus),sizeof(unsigned short));
	outfile.write(reinterpret_cast<const char *>(&testc),sizeof(char));
	outfile.write(reinterpret_cast<const char *>(&testuc),sizeof(unsigned char));
	outfile.write(reinterpret_cast<const char *>(&testf),sizeof(float));
	*/

	outfile.write(reinterpret_cast<const char *>(&sizeByte),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&xsize),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&ysize),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&zsize),sizeof(short));

	byteCount += 4 * sizeof(short);

	while(getline(imageConfig, line))
	{
		if(line.size() == 0 || line[0] == '#' || line[0] == '\n')
			continue;
		int comma1 = line.find(',');
		int comma2 = line.find(',',comma1+1);
		string folder = line.substr(0,comma1);
		unsigned short trueVal;
		if(comma2 != string::npos)
		 	trueVal = stoi(line.substr(comma1+1));
		else
			trueVal = stoi(line.substr(comma1+1,comma2));
		__globalTrueVal = trueVal;
		if(comma2 == string::npos)
			stride = globalStride;
		else
		{
			string stri = line.substr(comma2+1);
			if(stri.find("stride=") != string::npos)
			{
				stride = stoi(stri.substr(stri.find('=') + 1));
			}
			else
				stride = stoi(stri);
		}

		if(sizeByte == 1)
				getImages<unsigned char>(folder.c_str(),outfile);
		else if(sizeByte == -1)
				getImages<char>(folder.c_str(),outfile);
		else if(sizeByte == 2)
				getImages<unsigned short>(folder.c_str(),outfile);
		else if(sizeByte == -2)
				getImages<short>(folder.c_str(),outfile);
		else if(sizeByte == 4)
				getImages<unsigned int>(folder.c_str(),outfile);
		else if(sizeByte == -4)
				getImages<int>(folder.c_str(),outfile);
		else if(sizeByte == 5)
				getImages<float>(folder.c_str(),outfile);
		else if(sizeByte == 6)
				getImages<double>(folder.c_str(),outfile);
	}

	imageConfig.close();
	outfile.close();

	cout << "Total: " << imageNum << " images created" << endl;

	double sum = 0;
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		sum += it->second;
	}
	cout << "Distribution:" << endl;
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		cout << "True val " << it->first << ": " << it->second << "   " << it->second/sum * 100 << "%\n";
	}

	return 0;
}







