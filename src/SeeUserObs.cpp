#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <fstream>
#include "ConvNetCommon.h"
#include <cassert>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "MSILocations.h"


// #define FLAG_SHOW_MATCHED_BOXES //when not commmented will show image of each msi showing matched and unmatched boxes
// #define FLAG_SHOW_BLACKOUT_BOXES //when not commmented will show image of each msi showing blacked out boxes after the blackout stage
// #define FLAG_SHOW_MISSED_BACKGROUND

#define BACKGROUND -1
#define WHITE_PHASE 2
#define BLUE_PHASE 1000000
#define MATCHING_PERCENT 0.1
#define MATCHING_PERCENT_FOR_OUTPUT 0.5

using namespace cv;
using namespace std;
using namespace convnet;

struct OutImage
{
	Mat mat;
	int32_t species_id; 
};

// map<int, vector<Box> > locations;
map<int, MSI > locations;
vector<OutImage> forTrainingVariable;
vector<OutImage> forTrainingFixed;

//all species being excluded from calculations. Background is special and always in exclude.
//value doesn't matter, just whether or not it exists in the map.
unordered_map<int,int> exclude = {{BACKGROUND,1}}; 

void readInOriginalFilenames(const string& original_image_folder)
{
	bool isDirectory;
	struct stat s;
	unordered_map<int, string> filenamesByMSI;
	if(stat(original_image_folder.c_str(),&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
			isDirectory = true;
		else if (s.st_mode & S_IFREG) // file
			isDirectory = false;
		else
		{
			printf("We're not sure what the file you inputted for --original_image_folder was.\nExiting\n");
			exit(-1);
		}
		if(!isDirectory)
		{
			printf("The --original_image_folder should be a directory but it doesn't appear to be.\nExiting\n");
			exit(-1);
		}

		DIR *directory;
		struct dirent *file;
		if((directory = opendir(original_image_folder.c_str())))// != NULL)
		{
			string pathName = original_image_folder;
			if(pathName.rfind("/") != pathName.length()-1)
				pathName.append(1,'/');
			char inPathandName[500];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
					string ipan(inPathandName);
					if(ipan.find("_prediction") == string::npos)
					{
						// cout << "Found: " << ipan << " MSI - " << getMSI(ipan) << endl;
						filenamesByMSI[getMSI(ipan)] = ipan;

						// if(getMSI(ipan) == 5605)
						// {
						// 	cout << "\tFrom map: " << filenamesByMSI[getMSI(ipan)] << endl;
						// }
					}
				}
			}
			//cout << "closing directory" << endl;
			closedir(directory);
			//cout << "directory closed" << endl;
		}
		else
		{
			printf("directory problems dealing with --original_image_folder ??? Bye.\n");
			exit(-1);
		}

		for(auto it = locations.begin(); it != locations.end(); it++)
		{
			// if(it->first == 5605)
			// 	cout << "Trying 5605" << endl;
 
			if(filenamesByMSI.find(it->first) != filenamesByMSI.end()) // if we found the MSI in filenamesByMSI
			{
				// cout << "Putting " << it->first << endl;
				locations[it->first].original_image_path = filenamesByMSI[it->first];
			}

			//only worry if the original_image_path doesn't exist when looking for it

			// else
			// {
			// 	printf("Unable to find original image for MSI %d\n", it->first);
			// 	exit(-1);
			// }
		}
	}
	else
	{
		printf("Error getting status of folder for --original_image_folder.\nExiting\n");
		exit(-1);
	}
}

int getMaxIndex(const Vec3b& pix)
{
	if(pix[0] > pix[1] && pix[0] > pix[2])
		return 0;
	if(pix[1] > pix[2])
		return 1;
	return 2;
}

//returns species_id of what the pixel matches
//will not return species_id that is in exclude EXCEPT for BACKGROUND
inline int match(const Vec3b& pix)
{
	static bool keepWhite = exclude.find(WHITE_PHASE) == exclude.end();
	static bool keepBlue = exclude.find(BLUE_PHASE) == exclude.end();
	//STRATEGY: Highest pixel
	// if(pix[2] > pix[1] && pix[2] > pix[0]) //red channel max
	// 	return WHITE_PHASE;
	// else if(pix[1] > pix[0]) // green channel max
	// 	return BLUE_PHASE;

	//STRATEGY: Over 1/3
	if(keepWhite && pix[2] > 255 * .33) //red channel max
		return WHITE_PHASE;
	else if(keepBlue && pix[1] > 255 * .33) // green channel max
		return BLUE_PHASE;

	return BACKGROUND;
}

/*
possible dataTypes
0x08 - unsigned byte
0x09 - signed byte
0x0B - short
0x0C - int
0x0D - float
0x0E - double

dims will be converted to big endian on output
dim[0] should be number of items to output, then x, y, z dimesions
*/
void initIDX(ofstream& out, const string& outName, uint8_t dataType, const vector<int32_t>& dims)
{
	if(out.is_open())
		out.close();
	out.open(outName.c_str(), ios_base::binary | ios::trunc);
	//put magic number of 0 0 dataType numdims
	out.put(0); out.put(0);
	out.put(dataType);

	uint8_t numDims = (uint8_t)dims.size();
	out.put(numDims);


	//put all dims as Big Endian 32bit ints
	for(size_t i = 0; i < dims.size(); i++)
		for(int j = 24; j >= 0; j-=8)
			out.put(dims[i] >> j);
	// out.flush();
}

void writeMatToIDX(ofstream& out, const Mat& frame, uint8_t dataType)
{
	if(dataType != 0x08)
	{
		printf("Mats should be written to IDXs with a dataType of 0x08 (unsigned byte).\n");
		exit(0);
	}

	for(int i = 0; i < frame.rows; i++)
		for(int j = 0; j < frame.cols; j++)
		{
			const Vec3b& pix = frame.at<Vec3b>(i,j);
			out.put(pix[0]);
			out.put(pix[1]);
			out.put(pix[2]);
		}
	// out.flush();
}

void printImageMetaData(const vector<OutImage>& images)
{
	//count amount per species
	unordered_map<int, int> speciesCount;
	int count = 0;
	for(OutImage const &image : images)
	{
		if(image.species_id == WHITE_PHASE)
			count++;
		auto got = speciesCount.find(image.species_id);
		if(got == speciesCount.end()) // not found
			speciesCount[image.species_id] = 1;
		else // found
			speciesCount[image.species_id]++;
	}
	printf("white count %d\n",count);

	printf("Output IDX species counts and ratios:\n");
	for(auto it = speciesCount.begin(); it != speciesCount.end(); it++)
	{
		printf("   Species %7d: Count - %6d. Percent of Output Data - %5.2lf%%\n", it->first,it->second, 100.0 * it->second / images.size());
	}
	printf("  Total Image Count: %lu\n",images.size());
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("Usage: ./SeeUserObs obsfile image1 image2 ...\n");
		return 0;
	}
	int a = 1;

	readLocationsFile(argv[a++],locations);
	// for(auto it = locations.begin(); it != locations.end(); it++)
	// 	printf("%lu %d\n", sizeof(int), it->first);

	for(; a < argc; a++)
	{

		string filename(argv[a]);
		int msi = getMSI(filename); //got the msi

		//make sure we have locations for this msi
		if(locations.find(msi) == locations.end()) // if we don't have locations for the MSI
		{
			printf("Unable to find locations for MSI %d in the obsfile given. Skipping.\n", msi);
			continue;
		}

		Mat im = imread(argv[a],1); //got the image
		locations[msi].numPixels = im.rows * im.cols;
		int numBoxes = locations[msi].boxes.size();
		printf("Doing msi %5d: size %4d x %4d (w x h). Num obs: %3d\n", msi,im.cols,im.rows,numBoxes);

		//calculate # obs that match (white and blue)
		// if(numBoxes != 0) // if no boxes, matching and blackout aren't needed. But background misclassify checking is
		// {
			// printf("start box matching\n");
			for(int i = 0; i < numBoxes; i++)
			{
				locations[msi].boxes[i].hit = true;
				Box& box = locations[msi].boxes[i];

				//draw rectangles for visual confirmation of matching
				Scalar color;
				if(box.species_id == WHITE_PHASE)
					color = Scalar(0,0,255);
				else if(box.species_id == BLUE_PHASE)
				{
					color = Scalar(255,0,0);
				}
				else
					color = Scalar(0,0,0);
				rectangle(im,Point(box.x,box.y),Point(box.ex,box.ey),color,3);
			}
			imwrite("user_obs_image.png",im);
			//for showing images as they come with matched reds and green (not whether greens are matched) locations
			cv::Size mysizeMatched(750,750 * im.rows / im.cols);
			resize(im,im,mysizeMatched);
			imshow("image",im);

			waitKey(0); // if also showing missed background, show all at once
		// }
	}

	return 0;
}