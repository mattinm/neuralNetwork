/***************************************
*
*
*	ImageFileCreator takes a set of input images and converts them to a single binary file.
*
* 	All input images must be of the same size, specified in command line arguments. The format of the
*	output file is this:
*
*		sizeByte xSize ySize zSize image1 image2 ... imageN
*		short	 short short short 
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
*		./ImageFileCreator ImageConfigFile outfileName
*
*	ImageConfigFile format:
*		inWidth inHeight inDepth
*		sizeByte
*		folderOfImages1
*		folderOfImages2
*		...
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

using namespace std;
using namespace cv;

short xsize = -1, ysize = -1, zsize = -1;

int checkExtensions(const char* filename)
{
	const string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}

template<typename type>
void writeImage(const char* inPathandName, ofstream& outfile)
{
	Mat image = imread(inPathandName,1); //color image
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
				const Vec3b curPixel = image.at<Vec3b>(i,j);
				pixel[0] = (type)curPixel[0];
				pixel[1] = (type)curPixel[1];
				pixel[2] = (type)curPixel[2];

				//cout << "writing" << endl;
				outfile.write(reinterpret_cast<const char *>(pixel),size);
			}
		}
	}
	
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
						writeImage<type>(inPathandName,outfile);
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
			writeImage<type>(inPath,outfile);
		}

	}
}

int main (int argc, char** argv)
{
	if(argc != 3)
	{
		cout << "Usage: ./ImageFileCreator ImageConfigFile outfileName" << endl;
		return 0;
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

	cout << "Size: " << sizeByte << " x: " << xsize << " y: " << ysize << " z: " << zsize << endl;

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

	if(sizeByte == 1)
		while(getline(imageConfig,line))
			getImages<unsigned char>(line.c_str(),outfile);
	else if(sizeByte == -1)
		while(getline(imageConfig,line))
			getImages<char>(line.c_str(),outfile);
	else if(sizeByte == 2)
		while(getline(imageConfig,line))
			getImages<unsigned short>(line.c_str(),outfile);
	else if(sizeByte == -2)
		while(getline(imageConfig,line))
			getImages<short>(line.c_str(),outfile);
	else if(sizeByte == 4)
		while(getline(imageConfig,line))
			getImages<unsigned int>(line.c_str(),outfile);
	else if(sizeByte == -4)
		while(getline(imageConfig,line))
			getImages<int>(line.c_str(),outfile);
	else if(sizeByte == 5)
		while(getline(imageConfig,line))
			getImages<float>(line.c_str(),outfile);
	else if(sizeByte == 6)
		while(getline(imageConfig,line))
			getImages<double>(line.c_str(),outfile);

	imageConfig.close();
	outfile.close();

	return 0;
}







