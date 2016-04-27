#include <iostream>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <dirent.h>

int xsize = -1, ysize = -1, zsize = -1;
int size = 1;
double* buffer = NULL;

void setSize()
{
	size = xsize * ysize * zsize;
}

void setBuffer()
{
	setSize();
	buffer = new double[size];
}

void writeImage(const char* inPathandName, ofstream& outfile)
{
	Mat image = imread(inPathandName,1); //color image
	if(image.rows != xsize || image.cols != ysize || image.depth() != zsize)
		return;
	
}

void getImages(const char* folder, ofstream& outfile)
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
						writeImage(inPathandName,outfile);
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

int main (int argc, char** argv)
{
	//set xsize ysize and zsize from argv
}