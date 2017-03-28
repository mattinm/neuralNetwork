#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <fstream>
#include "ConvNetCommon.h"

#define FLAG_SHOW_MATCHED_BOXES //when not commmented will show image of each msi showing matched and unmatched boxes
#define FLAG_SHOW_BLACKOUT_BOXES //when not commmented will show image of each msi showing blacked out boxes after the blackout stage

#define BACKGROUND -1
#define WHITE_PHASE 2
#define BLUE_PHASE 1000000
#define MATCHING_PERCENT 0.1

using namespace cv;
using namespace std;
using namespace convnet;

struct OutImage
{
	Mat mat;
	int species_id; 
};

struct Box
{
	int species_id;
	int x;  // top left corner x
	int y;  // top left corner y
	int w;  // width
	int h;  // height
	int cx; // center point x
	int cy; // center point y
	int ex; // bottom right corner x
	int ey; // bottom right corner y
	bool matched = false; // whether we found a matching observation
	bool hit = false; //whether we looked at this msi at all

	Box();
	Box(int species_id, int x, int y, int w, int h);
	void load(int species_id, int x, int y, int w, int h);
	string toString();
};

Box::Box(){}

Box::Box(int species_id, int x, int y, int w, int h)
{
	load(species_id,x,y,w,h);
}

void Box::load(int species_id, int x, int y, int w, int h)
{
	this->species_id = species_id;
	this->x = x;
	this->y = y;
	this->w = w;
	this->h = h;
	this->cx = x + w/2;
	this->cy = y + h/2;
	this->ex = x + w;
	this->ey = y + h;
}

string Box::toString()
{
	char buf[200];
	sprintf(buf,"species: %d, x: %d, y: %d, w: %d, h: %d", species_id,x,y,w,h);
	return string(buf);
}

struct MSI
{
	int msi;
	vector<Box> boxes;

	//bmr => background_misclassified_percentages<species_id, ratio BG classified as species_id>
	//for the species_id != BACKGROUND, it is the misclassified ratio
	//for the species_id == BACKGROUND, it is the correctly classified ratio
	unordered_map<int,float> bmr; 

	MSI();
	MSI(int msi, int numBoxes);
	void init(int msi, int numBoxes);
};

MSI::MSI(){}

MSI::MSI(int msi, int numBoxes)
{
	init(msi,numBoxes);
}

void MSI::init(int msi, int numBoxes)
{
	this->msi = msi;
	this->boxes.resize(numBoxes);
}

// map<int, vector<Box> > locations;
map<int, MSI > locations;
vector<OutImage> forTrainingVariable;
vector<OutImage> forTrainingFixed;

void readFile(const char* filename)
{
	ifstream in(filename,ios::binary);
	int numMSIs = readInt(in);
	for(int i = 0; i < numMSIs; i++)
	{
		int msi = readInt(in);
		int numBoxes = readInt(in);
		// locations[msi] = vector<Box>(numBoxes);
		locations[msi].init(msi,numBoxes);
		for(int j = 0; j < numBoxes; j++)
			locations[msi].boxes[j].load(readInt(in),readInt(in),readInt(in),readInt(in),readInt(in));
	}
}

int getMSI(string filename)
{
	int startMSIIndex = filename.find("msi");
	int nextUnderscore = filename.find("_",startMSIIndex);
	return stoi(filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3));
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
inline int match(const Vec3b& pix)
{
	//STRATEGY: Highest pixel
	// if(pix[2] > pix[1] && pix[2] > pix[0]) //red channel max
	// 	return WHITE_PHASE;
	// else if(pix[1] > pix[0]) // green channel max
	// 	return BLUE_PHASE;

	//STRATEGY: Over 1/3
	if(pix[2] > 255 * .33) //red channel max
		return WHITE_PHASE;
	else if(pix[1] > 255 * .33) // green channel max
		return BLUE_PHASE;

	return BACKGROUND;
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("Usage: ./CNNtoObserver_comparator --idx_size=<int> --idx_name=name obsfile image1 image2 ...\n");
		printf("  --idx must come before other args if it is used\n");
		return 0;
	}
	int idx_size = -1;
	string outName = "";
	bool doOutput = false;
	ofstream out;
	int a = 1;
	while(a < argc && string(argv[a]).find('-') != string::npos)
	{
		string arg(argv[a]);
		if(arg.find("--idx_name=") != string::npos)
		{
			outName = arg.substr(arg.find('=')+1);
			doOutput = true;
		}
		else if(arg.find("--idx_size=") != string::npos)
		{
			idx_size = stoi(arg.substr(arg.find('=')+1));
			if(idx_size < 1)
			{
				printf("--idx_size must be a positive number.\n");
				return 0;
			}
		}
		a++;
	}

	if(idx_size == -1 && outName != "")
	{
		printf("Need to specify --idx_size when using --idx_name.\n");
		return 0;
	}
	if(outName == "" && idx_size != -1)
	{
		printf("Need to specify --idx_name when using --idx_size.\n");
		return 0;
	}

	//read in observations to global map
	if(a >= argc)
	{
		printf("Need an obsfile and images.");
		return 0;
	}
	readFile(argv[a]);

	// for(auto it = locations.begin(); it != locations.end(); it++)
	// {
	// 	printf("%d\n", it->first);
	// 	for(int i = 0; i < it->second.size(); i++)
	// 		printf("\t%s\n",it->second[i].toString().c_str());
	// }

	// namedWindow("image",WINDOW_NORMAL);
	// resizeWindow("image",600,600);

	for(; a < argc; a++)
	{
		string filename(argv[a]);
		int msi = getMSI(filename); //got the msi
		Mat im = imread(argv[a],1); //got the image
		printf("Doing msi %d: size %d x %d\n", msi,im.cols,im.rows);

		//make a copy to draw on
		#ifdef FLAG_SHOW_MATCHED_BOXES
		Mat draw = im.clone();
		#endif

		//calculate # obs that match (white and blue)
		int numBoxes = locations[msi].boxes.size();
		if(numBoxes == 0)
			continue;
		for(int i = 0; i < numBoxes; i++)
		{
			// printf("Box %d:\n", i);
			locations[msi].boxes[i].hit = true;
			Box& box = locations[msi].boxes[i];
			// cout<< box.toString() << endl;
			int size = box.w * box.h;
			unordered_map<int,int> phaseCount({{WHITE_PHASE,0},{BLUE_PHASE,0},{BACKGROUND,0}});
			for(int x = box.x; x < box.ex; x++)
				for(int y = box.y; y < box.ey; y++)
					phaseCount[match(im.at<Vec3b>(y,x))]++;


			if(box.species_id == WHITE_PHASE && (float)phaseCount[WHITE_PHASE]/size > MATCHING_PERCENT)
				locations[msi].boxes[i].matched = true;
			else if(box.species_id == BLUE_PHASE && (float)phaseCount[BLUE_PHASE]/size > MATCHING_PERCENT)
				locations[msi].boxes[i].matched = true;

			//if we are doing output AND if image is not matched, pull for training
			if(doOutput && locations[msi].boxes[i].matched == false)
			{
				int fx, fy, fex, fey; // fixed x start, fixed y start, fixed x end, fixed y end
				if(box.w >= idx_size)
				{
					fx = box.x;
					fex = box.ex;
				}
				else //box is too small in x direction, need padding
				{
					int paddingNeeded = idx_size - box.w;
					int leftAttempt = paddingNeeded/2; //we will attempt to put this much padding on the left
					int rightAttempt = paddingNeeded - leftAttempt; //we will attempt to put this much padding on the right
					//best case
					if(box.x - leftAttempt > 0 && box.ex + rightAttempt < im.cols)
					{
						fx = box.x - leftAttempt;
						fex = box.ex + rightAttempt;
					}
					//impossible case
					else if(im.cols < idx_size)
					{
						printf("Trying to make idx size %d x %d, but image is %d x %d. Impossible. I quit.\n", idx_size,idx_size,im.cols,im.rows);
						return -1;
					}
					//falling off left edge
					else if(box.x - leftAttempt < 0)
					{
						int leftActual = box.x; // box.x - 0 //We will actually put this much padding on the left
						int leftDiff = leftAttempt - leftActual;
						int rightActual = rightAttempt + leftDiff; // shouldn't fall off right side since we ruled out impossible case
						fx = 0; //box.x - leftActual
						fex = box.ex + rightActual;
					}
					//falling off right edge
					else
					{
						int rightActual = im.cols - 1 - box.ex; //We will actually put this much padding on the right
						int rightDiff = rightActual - rightAttempt;
						int leftActual = leftAttempt + rightDiff;
						fx = box.x - leftActual;
						fex = im.cols - 1; // box.ex + rightActual
					}

				}

				if(box.h >= idx_size)
				{
					fy = box.y;
					fey = box.ey;
				}
				else //box is too small in y direction, need padding
				{
					int paddingNeeded = idx_size - box.h;
					int upAttempt = paddingNeeded/2; //we will attempt to put this much padding on the top
					int downAttempt = paddingNeeded - upAttempt; //we will attempt to put this much padding on the bottom
					//best case
					if(box.y - upAttempt > 0 && box.ey + downAttempt < im.rows)
					{
						fy = box.y - upAttempt;
						fey = box.ey + downAttempt;
					}
					//impossible case. proper size would fall off both edges
					else if(im.rows < idx_size)
					{
						printf("Trying to make idx size %d x %d, but image is %d x %d. Impossible. I quit.\n", idx_size,idx_size,im.cols,im.rows);
						return -1;
					}
					//falling off top
					else if(box.y - upAttempt < 0)
					{
						int upActual = box.y; // box.y - 0 //We will actually put this much padding on the top
						int upDiff = upAttempt - upActual;
						int downActual = downAttempt + upDiff; // shouldn't fall off bottom since we ruled out impossible case
						fy = 0; //box.y - upActual
						fey = box.ey + downActual;
					}
					//falling off bottom
					else
					{
						int downActual = im.rows - 1 - box.ey; //We will actually put this much padding on the bottom
						int downDiff = downActual - downAttempt;
						int upActual = upAttempt + downDiff;
						fy = box.y - upActual;
						fey = im.rows - 1; // box.ey + rightActual
					}
				}
				                //im(rowRange,colRange) upper boundary not included (hence the +1)
				Mat forTraining = im(Range(fy,fey + 1),Range(fx,fex + 1));

				//need to put image and true val in forTrainingVariable vector
				OutImage varImage;
				varImage.mat = forTraining.clone(); //need to deep copy
				varImage.species_id = box.species_id;
				forTrainingVariable.push_back(varImage);
			}

			//draw rectangles for visual confirmation of matching
			#ifdef FLAG_SHOW_MATCHED_BOXES
			Scalar color;
			if(box.species_id == WHITE_PHASE && locations[msi].boxes[i].matched == true)
				color = Scalar(255,255,255);
			else if(box.species_id == WHITE_PHASE)
			{
				color = Scalar(0,0,0);
			}
			else
				color = Scalar(0,255,0);
			rectangle(draw,Point(box.x,box.y),Point(box.ex,box.ey),color,3);
			#endif


		}

		//for showing images as they come with matched reds and green (not whether greens are matched) locations
		#ifdef FLAG_SHOW_MATCHED_BOXES
		cv::Size mysizeMatched(750,750 * draw.rows / draw.cols);
		resize(draw,draw,mysizeMatched);
		imshow("image",draw);
		waitKey(0);
		#endif

		//remove (blackout) areas of observations
		for(int i = 0; i < numBoxes; i++)
		{
			// printf("Box %d:\n", i);
			Box& box = locations[msi].boxes[i];
			for(int x = box.x; x < box.ex; x++)
				for(int y = box.y; y < box.ey; y++)
				{
					Vec3b& pix = im.at<Vec3b>(y,x);
					pix[0] = 0;
					pix[1] = 0;
					pix[2] = 0;
				}
		}

		//for showing blacked out images
		#ifdef FLAG_SHOW_BLACKOUT_BOXES
		Mat blackout = im.clone();
		cv::Size mysizeBlackout(750,750 * blackout.rows / blackout.cols);
		resize(blackout,blackout,mysizeBlackout);
		imshow("image blackout",blackout);
		waitKey(0);
		#endif

		//now everything should be blue (hypothetically)
		//calc ratio and sizes of misclassified background area.

		//count background pixels and what they were classified as
		unordered_map<int,int> phaseCount({{WHITE_PHASE,0},{BLUE_PHASE,0},{BACKGROUND,0}}); //technically more of a speciesCount than a phaseCount
		int totalBGCount = 0;
		for(int y = 0; y < im.rows; y++)
		{
			for(int x = 0; x < im.cols; x++)
			{
				const Vec3b& pix = im.at<Vec3b>(y,x);
				phaseCount[match(pix)]++; //this could maybe be parallelized with a concurrent map or a vector of maps that are summed after
				if(pix[0] != 0 || pix[1] != 0 || pix[2] != 0) //ie, the pixel is not black
					totalBGCount++;
			}
		}

		//cacluate ratios
		for(auto it = phaseCount.begin(); it != phaseCount.end(); it++)
		{
			//it->first is the species_id
			//it->second is the count for the species
			locations[msi].bmr[it->first] = (float)it->second/totalBGCount;
		}

		//pull misclassified BG area for training
		//stride over full image
		int size = idx_size * idx_size;
		int stride = idx_size / 2;
		for(int y = 0; y < im.rows - idx_size; y += stride)
		{
			for(int x = 0; x < im.cols - idx_size; x += stride)
			{
				int cacluatedSpecies;
				map<int,int> phaseCount({{WHITE_PHASE,0},{BLUE_PHASE,0},{BACKGROUND,0}});
				for(int i = 0; i < idx_size; i++)
					for(int j = 0; j < idx_size; j++)
						phaseCount[match(im.at<Vec3b>(y+j,x+i))]++;

				for(auto it = phaseCount.begin(); it != phaseCount.end(); it++)
				{
					if((float)it->second/size > MATCHING_PERCENT)
					{
						cacluatedSpecies = it->first;
						break;
					}
				}

				if(cacluatedSpecies != BACKGROUND)
				{
					Mat forTraining = im(Range(y, y + idx_size),Range(x, x + idx_size));

					//need to put image and true val in forTrainingVariable vector
					OutImage fixedImage;
					fixedImage.mat = forTraining.clone(); //need to deep copy
					fixedImage.species_id = BACKGROUND;
					forTrainingFixed.push_back(fixedImage);
				}
			}
		}
	}

	//convert all the variable sized images for training to the fixed idx_size

	//calcuate aggregate matching obs
	int numWhite = 0, numBlue = 0, matchedWhite = 0, matchedBlue = 0;
	for(auto it = locations.begin(); it != locations.end(); it++)
	{
		int size = it->second.boxes.size();
		for(int i = 0; i < size; i++)
		{
			if(it->second.boxes[i].hit == false)
				break;
			if(it->second.boxes[i].species_id == WHITE_PHASE)
			{
				numWhite++;
				if(it->second.boxes[i].matched == true)
					matchedWhite++;
			}
			else if(it->second.boxes[i].species_id == BLUE_PHASE)
			{
				numBlue++;
				if(it->second.boxes[i].matched == true)
					matchedBlue++;
			}
		}
	}

	printf("Overall (white:blue): Matched %d:%d, Total %d:%d, Percent %.2lf:%.2lf\n", matchedWhite,matchedBlue,numWhite,numBlue,100.*matchedWhite/numWhite,100.*matchedBlue/numBlue);

	return 0;
}