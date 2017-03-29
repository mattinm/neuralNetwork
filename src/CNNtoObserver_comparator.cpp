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

// #define FLAG_SHOW_MATCHED_BOXES //when not commmented will show image of each msi showing matched and unmatched boxes
// #define FLAG_SHOW_BLACKOUT_BOXES //when not commmented will show image of each msi showing blacked out boxes after the blackout stage

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
	int32_t species_id; 
};

struct Box
{
	int32_t species_id;
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
	Box(int32_t species_id, int x, int y, int w, int h);
	void load(int32_t species_id, int x, int y, int w, int h);
	string toString();
};

Box::Box(){}

Box::Box(int32_t species_id, int x, int y, int w, int h)
{
	load(species_id,x,y,w,h);
}

void Box::load(int32_t species_id, int x, int y, int w, int h)
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
	sprintf(buf,"species: %7d, x: %d, y: %d, w: %d, h: %d, ex: %d, ey: %d", species_id,x,y,w,h,ex,ey);
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

//all species being excluded from calculations. Background is special and always in exclude.
//value doesn't matter, just whether or not it exists in the map.
unordered_map<int,int> exclude = {{BACKGROUND,1}}; 

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
	// printf("%s\n", filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3).c_str());
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
void initIDX(ofstream& out, const string& outName, uint8_t dataType, uint8_t numDims, const vector<int32_t>& dims)
{
	if(out.is_open())
		out.close();
	out.open(outName.c_str(), ios_base::binary | ios::trunc);
	//put magic number of 0 0 dataType numdims
	out.put(0); out.put(0);
	out.put(dataType);
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
	if(argc < 2)
	{
		printf("Usage: ./CNNtoObserver_comparator  obsfile image1 image2 ...\n");
		printf("  Optional args MUST come before required args\n");
		printf("  --idx_name=<string> A base name (no extension) for IDX output files. Actual files will be baseName_data.idx and baseName_label.idxn");
		printf("  --idx_size=<int>    The size in pixels of the width/height for the output IDX. All output images will be square.\n");
		printf("  --exclude=<int>     The species_id of the species to exclude from calculations and the output IDX. Can be used multiple times.\n");
		return 0;
	}
	int32_t idx_size = -1;
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
		else if(arg.find("--exclude=") != string::npos)
		{
			exclude[stoi(arg.substr(arg.find('=')+1))] = 1;
		}
		else
		{
			printf("Unknown arg '%s'. Exiting.\n", argv[a]);
			return -1;
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
	readFile(argv[a++]);

	// for(auto it = locations.begin(); it != locations.end(); it++)
	// {
	// 	printf("%d\n", it->first);
	// 	for(int i = 0; i < it->second.size(); i++)
	// 		printf("\t%s\n",it->second[i].toString().c_str());
	// }

	// namedWindow("image",WINDOW_NORMAL);
	// resizeWindow("image",600,600);

	int bgStride = idx_size / 2;
	int fgStride = 1000;

	for(; a < argc; a++)
	{
		string filename(argv[a]);
		int msi = getMSI(filename); //got the msi
		Mat im = imread(argv[a],1); //got the image
		int numBoxes = locations[msi].boxes.size();
		printf("Doing msi %5d: size %4d x %4d (w x h). Num obs: %3d\n", msi,im.cols,im.rows,numBoxes);

		//make a copy to draw on
		#ifdef FLAG_SHOW_MATCHED_BOXES
		Mat draw = im.clone();
		#endif

		//calculate # obs that match (white and blue)
		if(numBoxes != 0) // if no boxes, matching and blackout aren't needed. But background misclassify checking is
		{
			// printf("start box matching\n");
			for(int i = 0; i < numBoxes; i++)
			{
				locations[msi].boxes[i].hit = true;
				Box& box = locations[msi].boxes[i];

				//some boxes are dumb and go outside the bounds of the image. We must manually fix it.
				if(box.x < 0) { box.x = 0; box.w = box.ex - box.x; }
				if(box.ex >= im.cols) { box.ex = im.cols - 1; box.w = box.ex - box.x; }
				if(box.y < 0) { box.y = 0; box.h = box.ey - box.y; }
				if(box.ey >= im.rows) { box.ey = im.rows - 1; box.h = box.ey - box.y; }

				//don't do boxes of excluded species
				if(exclude.find(box.species_id) != exclude.end()) //if the species is found in exclude
					continue;

				// printf("Box %d: x %d y %d ex %d ey %d w %d h %d\n", i,box.x,box.y,box.ex,box.ey,box.w,box.h);
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
						// printf("BOX TOO SMALL!!!\n%s\n",box.toString().c_str());
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
							int rightDiff = rightAttempt - rightActual;
							int leftActual = leftAttempt + rightDiff;
							fx = box.x - leftActual;
							fex = im.cols - 1; // box.ex + rightActual
							// printf("fall right: RAct %d RD %d LAct %d fx %d, fex %d\n", rightActual, rightDiff, leftActual, fx,fex);
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
							int downDiff = downAttempt - downActual;
							int upActual = upAttempt + downDiff;
							fy = box.y - upActual;
							fey = im.rows - 1; // box.ey + rightActual
						}
					}
					                //im(rowRange,colRange) upper boundary not included (hence the +1)
					Mat forTraining = im(Range(fy,fey + 1),Range(fx,fex + 1));

					// printf("adding missed positive species %d: size %d x %d\n", box.species_id,forTraining.cols,forTraining.rows);

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

			// printf("start blackout\n");
			//remove (blackout) areas of observations (ONLY NEED IF DOING OUTPUT IDX or IF SHOWING BLACKOUT IMAGES)
			#ifndef FLAG_SHOW_BLACKOUT_BOXES //don't want if statement if showing blackout boxes
			if(doOutput)
			#endif
			{
				for(int i = 0; i < numBoxes; i++)
				{
					// printf("Box %d:\n", i);
					Box& box = locations[msi].boxes[i];

					//don't blackout boxes of excluded species
					if(exclude.find(box.species_id) != exclude.end()) //if the species is found in exclude
						continue;

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
			}
		}

		//now everything should be blue (hypothetically)
		//calc ratio and sizes of misclassified background area.

		//count background pixels and what they were classified as

		// printf("start bmr\n");
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
		if(doOutput)
		{
			//stride over full image and grab boxes that don't match bg or excluded species
			// printf("Image: rows %d cols %d\n", im.rows,im.cols);
			int size = idx_size * idx_size;
			for(int y = 0; y < im.rows - idx_size; y += bgStride)
			{
				for(int x = 0; x < im.cols - idx_size; x += bgStride)
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
	
					// if(cacluatedSpecies != BACKGROUND)
					if(exclude.find(cacluatedSpecies) == exclude.end()) //exclude always includes BG
					{
						// printf("box: y %d ey %d x %d ex %d\n", y,y+idx_size,x,x+idx_size);
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
	}

	if(doOutput)
	{
		//convert all the variable sized images for training to the fixed idx_size
		printf("For training variable size %lu\n", forTrainingVariable.size());
		for(OutImage &image : forTrainingVariable)
		{
			// bool pushed = false;
			for(int y = 0; y < image.mat.rows - idx_size; y += fgStride)
				for(int x = 0; x < image.mat.cols - idx_size; x += fgStride)
				{
					// pushed = true;
					Mat forTraining = image.mat(Range(y, y + idx_size),Range(x, x + idx_size));
	
					//need to put image and true val in forTrainingVariable vector
					OutImage fixedImage;
					fixedImage.mat = forTraining.clone(); //need to deep copy
					fixedImage.species_id = image.species_id;
					forTrainingFixed.push_back(fixedImage);
				}
			// if(!pushed)
			// {
			// 	printf("  !!!!!!!!!!! not pushed\n");
			// 	printf(" cols %d rows %d xcap %d ycap %d\n", image.mat.cols,image.mat.rows,image.mat.cols - idx_size,image.mat.cols - idx_size);
			// }

			image.mat.release(); // release memory to try to keep mem usage lower
		}

		printf("\n");
		printImageMetaData(forTrainingFixed);
		printf("\n");

		//write out IDX
		ofstream outdata,outlabel;
		vector<int32_t> dimsdata = {(int32_t)forTrainingFixed.size(),idx_size,idx_size,3}; //num images, rows, cols, depth
		vector<int32_t> dimslabel = {(int32_t)forTrainingFixed.size()}; //num images
		initIDX(outdata,outName+string("_data.idx"),0x08,4,dimsdata); //idx named outName with data type of unsigned byte
		initIDX(outlabel,outName+string("_label.idx"),0x0C,4,dimslabel); //idx named outName with data type of int
		for(OutImage &image : forTrainingFixed)
		{
			assert(image.mat.cols == idx_size && image.mat.rows == idx_size);
			writeMatToIDX(outdata,image.mat,0x08);
			outlabel.write(reinterpret_cast<const char *>(&(image.species_id)),4);
		}
	}

	//calcuate aggregate matching obs
	int numWhite = 0, numBlue = 0, matchedWhite = 0, matchedBlue = 0;
	unordered_map<int,int> num = { {WHITE_PHASE,0},{BLUE_PHASE,0} };
	unordered_map<int,int> matched = { {WHITE_PHASE,0},{BLUE_PHASE,0} };
	for(auto it = locations.begin(); it != locations.end(); it++)
	{
		for(int i = 0; i < it->second.boxes.size(); i++)
		{
			if(it->second.boxes[i].hit == false)
				break;
			if(exclude.find(it->second.boxes[i].species_id) != exclude.end()) //the current species_id is in exclude 
				continue;
			
			num[it->second.boxes[i].species_id]++;
			if(it->second.boxes[i].matched)
				matched[it->second.boxes[i].species_id]++;
			// if(it->second.boxes[i].species_id == WHITE_PHASE)
			// {
			// 	numWhite++;
			// 	if(it->second.boxes[i].matched == true)
			// 		matchedWhite++;
			// }
			// else if(it->second.boxes[i].species_id == BLUE_PHASE)
			// {
			// 	numBlue++;
			// 	if(it->second.boxes[i].matched == true)
			// 		matchedBlue++;
			// }
		}
	}

	printf("Species Accuracy:\n");
	for(auto it = num.begin(); it != num.end(); it++)
	{
		if(exclude.find(it->first) != exclude.end()) // if species is excluded, skip
			continue;
		int count = it->second;
		int match = matched[it->first];
		int miss  = count - match;
		printf("   Species %7d: Match - %3d Miss - %3d Total - %3d Match Percent %5.2f\n", it->first,match,miss,count, 100.0 * match/count);
	}


	// printf("Overall (white:blue): Matched %d:%d, Missed %d:%d Total %d:%d, Percent %.2lf:%.2lf\n", matchedWhite,matchedBlue,numWhite-matchedWhite,numBlue-matchedBlue,numWhite,numBlue,100.*matchedWhite/numWhite,100.*matchedBlue/numBlue);

	return 0;
}