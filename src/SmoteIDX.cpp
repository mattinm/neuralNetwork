#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <limits>
#include <random>
#include <chrono>
#include "opencv2/imgproc/imgproc.hpp" //used for showing images being read in from IDX
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

typedef std::vector<std::vector<std::vector<double> > > imVector;
bool showImages = false;
int imcount = 0;

/**********************
 *	Global Variables
 ***********************/
unordered_map<double, unsigned int> sample_counts; //[class] -> num images of class contained

/**********************
 *	Helper Functions
 ***********************/

inline double clamp(double val, double low, double high)
{
	if(val < low)  return low;
	if(val > high) return high;
	return val;
}

void resize3DVector(imVector& vect, int width, int height, int depth)
{
	vect.resize(width);
	for (auto&& i : vect) {
		i.resize(height);
		for (auto&& j : i) {
			j.resize(depth);
		}
	}
}

char readChar(ifstream& in)
{
	char num[1];
	in.read(num,1);
	return num[0];
}
unsigned char readUChar(ifstream& in)
{
	char num[1];
	in.read(num,1);
	return num[0];
}

short readShort(ifstream& in)
{
	short num;
	in.read((char*)&num,sizeof(short));
	return num;
}
unsigned short readUShort(ifstream& in)
{
	unsigned short num;
	in.read((char*)&num,sizeof(unsigned short));
	return num;
}

int readInt(ifstream& in)
{
	int num;
	in.read((char*)&num,sizeof(int));
	return num;
}

int readBigEndianInt(ifstream& in)
{
	int out = 0;
	for(int i=3; i >= 0; i--)
		out |= (readUChar(in) << 8*(i));
	return out;
}
unsigned int readUInt(ifstream& in)
{
	unsigned int num;
	in.read((char*)&num,sizeof(unsigned int));
	return num;
}

float readFloat(ifstream& in)
{
	float num;
	in.read((char*)&num,sizeof(float));
	return num;
}

double readDouble(ifstream& in)
{
	double num;
	in.read((char*)&num,sizeof(double));
	return num;
}

/**********************
 *	IDX Functions
 ***********************/

double getNextImage(ifstream& in, ifstream& trueval_in, imVector& dest, int x, int y, int z, int sizeByteData, int sizeByteLabel)
{
	//get 1 image
	resize3DVector(dest,x,y,z);
	// printf("resize as %d x %d x %d\n", x,y,z);
	for(int i=0; i < x; i++)
	{
		for(int j=0; j < y; j++)
		{
			for(int k=0; k < z; k++)
			{
				if(in.eof())
				{
					printf("early end of data file! imcount = %d\n",imcount);
					exit(-1);
				}
				if(sizeByteData == 0x08)
					dest[i][j][k] = (double)readUChar(in);
				else if(sizeByteData == 0x09)
					dest[i][j][k] = (double)readChar(in);
				else if(sizeByteData == 0x0B)
					dest[i][j][k] = (double)readShort(in);
				else if(sizeByteData == 0x0C)
					dest[i][j][k] = (double)readInt(in);
				else if(sizeByteData == 0x0D)
					dest[i][j][k] = (double)readFloat(in);
				else if(sizeByteData == 0x0E)
					dest[i][j][k] = readDouble(in);
				else
				{
					cout << "Unknown sizeByte for data: " << sizeByteData << ". Exiting" << endl;
					exit(0);
				}
			}
		}
	}

	if(trueval_in.eof())
	{
		printf("early end of label file!\n");
		exit(-1);
	}

	//return the trueVal
	double trueVal = 0;
	if(sizeByteLabel == 0x08)
		trueVal = (double)readUChar(trueval_in);
	else if(sizeByteLabel == 0x09)
		trueVal = (double)readChar(trueval_in);
	else if(sizeByteLabel == 0x0B)
		trueVal = (double)readShort(trueval_in);
	else if(sizeByteLabel == 0x0C)
		trueVal = (double)readInt(trueval_in);
	else if(sizeByteLabel == 0x0D)
		trueVal = (double)readFloat(trueval_in);
	else if(sizeByteLabel == 0x0E)
		trueVal = readDouble(trueval_in);
	else
	{
		cout << "Unknown sizeByte for data: " << sizeByteLabel << ". Exiting" << endl;
		exit(0);
	}

	// string retval = to_string((long)trueVal);
	// if(excludes.find(retval) != excludes.end())
	// 	retval = "-1";
	// cout << retval << endl;

	//show image and trueVal
	if(showImages)
	{
		Mat show(x,y,CV_8UC3);
		for(int i = 0; i < x; i++)
		{
			for(int j = 0; j < y; j++)
			{
				Vec3b& outPix = show.at<Vec3b>(i,j);
				outPix[0] = dest[i][j][0];
				outPix[1] = dest[i][j][1];
				outPix[2] = dest[i][j][2];
			}
		}
		char name[10];
		sprintf(name,"%d",(int)trueVal);
		imshow(name,show);
		waitKey(0);
		printf("Count: %d true: %lf\n",imcount, trueVal);
	}
	imcount++;

	//add to counts
	if(sample_counts.find(trueVal) == sample_counts.end()) // if not found
		sample_counts[trueVal] = 1;
	else
		sample_counts[trueVal]++;
	
	return trueVal;
}

void magic(ifstream& in, int& dataType, int& numDims)
{
	if(readUChar(in) + readUChar(in) != 0)
	{
		printf("Incorrect magic number format\n");
		exit(0);
	}
	dataType = (int)readUChar(in);
	numDims = (int)readUChar(in);
}

void writeVectorToIDX(ofstream& out, const imVector& frame, uint8_t dataType)
{
	if(dataType != 0x08)
	{
		printf("imVectors should be written to IDXs with a dataType of 0x08 (unsigned byte).\n");
		exit(0);
	}

	for(int i = 0; i < frame.size(); i++)
		for(int j = 0; j < frame[i].size(); j++)
			for(int k = 0; k < frame[i][j].size(); k++)
				out.put(frame[i][j][k]);
}

void initIDX(ofstream& out, const char *outName, uint8_t dataType, const vector<int32_t>& dims)
{
	if(out.is_open())
		out.close();
	out.open(outName, ios_base::binary | ios::trunc);
	//put magic number of 0 0 dataType numdims
	out.put(0); out.put(0);
	out.put(dataType);

	uint8_t numDims = (uint8_t)dims.size();
	out.put(numDims);

	//put all dims as Big Endian 32bit ints
	for(size_t i = 0; i < dims.size(); i++)
		for(int j = 24; j >= 0; j-=8)
			out.put(dims[i] >> j);
}

void initIDX(ofstream& out, const string& outName, uint8_t dataType, const vector<int32_t>& dims)
{
	initIDX(out,outName.c_str(), dataType,dims);
}

streamsize getSize(int data_type)
{
	if(data_type == 0x08 || data_type == 0x09)
		return sizeof(char); //trueVal = (double)readUChar(trueval_in);
	else if(data_type == 0x0B)
		return sizeof(short); //trueVal = (double)readShort(trueval_in);
	else if(data_type == 0x0C)
		return sizeof(int); //trueVal = (double)readInt(trueval_in);
	else if(data_type == 0x0D)
		return sizeof(float); // trueVal = (double)readFloat(trueval_in);
	else if(data_type == 0x0E)
		return sizeof(double); //trueVal = readDouble(trueval_in);
	else return 0;
}

/**********************
 *	SMOTE Functions
 ***********************/

double getDistance(imVector& a, imVector& b)
{
	double distance = 0;
	if(a.size() != b.size()){printf("Incompatible imVectors for distance\n");return 2;}
	for(int i = 0; i < a.size(); i++)
	{
		if(a[i].size() != b[i].size()){printf("Incompatible imVectors for distance\n");return 2;}
		for(int j = 0; j < a[i].size(); j++)
		{
			if(a[i][j].size() != b[i][j].size()){printf("Incompatible imVectors for distance\n");return 2;}
			for(int k = 0; k < a[i][j].size(); k++)
				// distance += abs(a[i][j][k] - b[i][j][k]);
				distance += pow(a[i][j][k] - b[i][j][k],2);
		}
	}
	printf("distance = %lf\n", distance);
	return distance;
}

void getCandidateForReplacement(vector<double>& distances, int& outIndex, double& outDistance)
{
	if(distances.size() < 1)
	{
		outIndex = -1;
		outDistance = -1;
		return;
	}
	int oi = 0;
	double od = distances[0];
	for(int i = 1; i < distances.size(); i++)
	{
		if(distances[i] > od)
		{
			od = distances[i];
			oi = i;
		}
	}

	outIndex = oi;
	outDistance = od;
}

/**********************
 *	Main Function
 ***********************/

int main(int argc, char** argv)
{
	if(argc < 5)
	{
		printf("Usage: SmoteIDX data.idx label.idx save_base_name smote_percent(100, 200, etc)\n\tDo not put .idx on the save_base_name\n");
		return 0;
	}

	ifstream in_data, in_label;

	int smote_percent = atoi(argv[4]);
	char* saveName = argv[3];


	/**************************
	*
	* get idx metadata for reading
	*
	**************************/
	int num_data, data_type, label_type, data_num_dims, label_num_dims;
	in_data.open(argv[1]);
	in_label.open(argv[2]);

	magic(in_data,data_type,data_num_dims);
	magic(in_label,label_type,label_num_dims);

	num_data = readBigEndianInt(in_data);
	if(num_data != readBigEndianInt(in_label))
	{
		printf("Inconsistent number of data items between data and labels.\n");
		return 1;
	}
	if(data_num_dims - 1 > 3)
	{
		printf("Can only handle at most 3 dimensions in the training data right now. Sorry.\n");
		return 0;
	}

	vector<int> label_dims, data_dims(3,1);

	for(int i = 0; i < data_num_dims - 1; i++)
		data_dims[i] = readBigEndianInt(in_data);
	for(int i = 0; i < label_num_dims - 1; i++)
		label_dims.push_back(readBigEndianInt(in_label));

	printf("Data File:  %d images, dims %dx%dx%d\n", num_data,data_dims[0],data_dims[1],data_dims[2]);
	printf("Label File: %d images\n", num_data);


	/**************************
	*
	* get data
	*
	**************************/

	vector<imVector> sample(num_data);
	vector<double> sample_labels(num_data);

	for(int i = 0; i < num_data; i++)
		sample_labels[i] = getNextImage(in_data, in_label, sample[i], data_dims[0], data_dims[1], data_dims[2], data_type, label_type);



	/**************************
	*
	* smote it
	*
	**************************/

	if(sample_counts.size() == 0)
	{
		printf("Error: no data was counted\n");
		return 1;
	}

	int minority_sample_size = sample_counts.begin()->second;
	double minority_label = sample_counts.begin()->first;
	for(auto it = ++(sample_counts.begin()); it != sample_counts.end(); it++)
	{
		if(it->second < minority_sample_size)
		{
			minority_sample_size = it->second;
			minority_label = it->first;
		}
	}

	printf("Minority class is label %lf with %u samples\n", minority_label, minority_sample_size);

	vector<imVector*> minorities(minority_sample_size);
	for(int i = 0, j = 0; i < num_data; i++)
		if(sample_labels[i] == minority_label)
			minorities[j++] = &(sample[i]);

	int synthetic_created = 0;
	const int sythetic_per_image = smote_percent / 100;
	const int synthetic_needed = sythetic_per_image * minority_sample_size;
	const int nnn = 2; //num nearest neighbors

	if(nnn > num_data - 1) //if we can't have that many nearest neighbors because we don't have that much data (unlikely)
	{
		printf("Can't have %d nearest neighbors with only %d data samples.\n", nnn, num_data);
		return 1;
	}

	vector<imVector> synthetic(synthetic_needed);
	for(int i = 0; i < synthetic_needed; i++)
		resize3DVector(synthetic[i],data_dims[0],data_dims[1],data_dims[2]);
	vector<double> synthetic_labels(synthetic_needed);

	vector<imVector*> nns(nnn);
	vector<double> nns_distance(nnn);

	int toReplace;
	double toReplace_distance;

	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	uniform_int_distribution<uint> dis(0,nnn-1);
	uniform_real_distribution<double> gap(0.0,1.0);

	cv::Size size(100,100);

	for(int s = 0; s < minority_sample_size; s++)
	{
		for(int n = 0; n < nnn; n++)
			nns_distance[n] = numeric_limits<double>::max();

		//get nearest neighbors
		for(int nei = 0; nei < minority_sample_size; nei++)
		{
			if(nei == s) continue;
			double distance = getDistance(*minorities[s],*minorities[nei]);
			getCandidateForReplacement(nns_distance,toReplace,toReplace_distance); // this fills toReplace and toReplace_distance

			if(distance < toReplace_distance) //do the replacement
			{
				nns_distance[toReplace] = distance;
				nns[toReplace] = minorities[nei];
			}
		}

		// for(int q = 0; q < nns_distance.size(); q++)
		// 	printf("neighbor %d - distance %lf\n", q,nns_distance[q]);

		Mat sam(data_dims[0],data_dims[1],CV_8UC3);
		for(int i = 0; i < data_dims[0]; i++)
		{
			for(int j = 0; j < data_dims[1]; j++)
			{
				Vec3b& outPix = sam.at<Vec3b>(i,j);
				outPix[0] = minorities[s]->at(i)[j][0];
				outPix[1] = minorities[s]->at(i)[j][1];
				outPix[2] = minorities[s]->at(i)[j][2];

				// outPix[0] = minorities[s]->at(i)[j][0];
				// outPix[1] = minorities[s]->at(i)[j][0];
				// outPix[2] = minorities[s]->at(i)[j][0];
			}
		}
		char name[100];
		sprintf(name,"minority");
		resize(sam,sam,size);
		imshow(name,sam);

		// waitKey(0);
		int maxdiff = 0, maxdiffloc[] = {-1,-1};

		//create synthetic samples
		for(int syn = 0; syn < sythetic_per_image; syn++)
		{
			int curNei = dis(gen);
			for(int i = 0; i < minorities[s]->size(); i++)
			{
				for(int j = 0; j < minorities[s]->at(i).size(); j++)
				{
					// double diff = 0;
					// for(int k = 0; k < minorities[s]->at(i)[j].size(); k++)
					// 	// diff += pow(abs(minorities[s]->at(i)[j][k] - nns[curNei]->at(i)[j][k]),0.5);
					// 	diff += minorities[s]->at(i)[j][k] - nns[curNei]->at(i)[j][k];

					// if(diff > maxdiff)
					// {
					// 	maxdiff = diff;
					// 	maxdiffloc[0] = i;
					// 	maxdiffloc[1] = j;
					// }
					// double add = gap(gen) * diff / minorities[s]->at(i)[j].size();
					
					// for(int k = 0; k < minorities[s]->at(i)[j].size(); k++)
					// {
					// 	printf("(%d,%d,%d): diff %lf gap %lf old %lf new %lf neighbor distance %lf\n",i,j,k, diff,add,minorities[s]->at(i)[j][k],minorities[s]->at(i)[j][k] + add,nns_distance[curNei]);
					// 	synthetic[synthetic_created][i][j][k] = clamp(minorities[s]->at(i)[j][k] + add, 0, 255);
					// }
					

					for(int k = 0; k < minorities[s]->at(i)[j].size(); k++)
					{
						double diff = minorities[s]->at(i)[j][k] - nns[curNei]->at(i)[j][k];
						if(diff > maxdiff)
						{
							maxdiff = diff;
							maxdiffloc[0] = i;
							maxdiffloc[1] = j;
						}
						double add = gap(gen) * diff;
						printf("(%d,%d,%d): diff %lf gap %lf old %lf new %lf neighbor distance %lf\n",i,j,k, diff,add,minorities[s]->at(i)[j][k],minorities[s]->at(i)[j][k] + add,nns_distance[curNei]);
						synthetic[synthetic_created][i][j][k] = clamp(minorities[s]->at(i)[j][k] + add, 0, 255);
					}
					printf("\n");
				}
			}

			printf("max diff %d at (%d,%d)\n", maxdiff, maxdiffloc[0],maxdiffloc[1]);

			Mat nei(data_dims[0],data_dims[1],CV_8UC3);
			for(int i = 0; i < data_dims[0]; i++)
			{
				for(int j = 0; j < data_dims[1]; j++)
				{
					Vec3b& outPix = nei.at<Vec3b>(i,j);
					outPix[0] = nns[curNei]->at(i)[j][0];
					outPix[1] = nns[curNei]->at(i)[j][1];
					outPix[2] = nns[curNei]->at(i)[j][2];

					// outPix[0] = nns[curNei]->at(i)[j][0];
					// outPix[1] = nns[curNei]->at(i)[j][0];
					// outPix[2] = nns[curNei]->at(i)[j][0];
				}
			}
			char name2[100];
			sprintf(name2,"neighbor");
			resize(nei,nei,size);
			imshow(name2,nei);

			// waitKey(0);

			Mat show(data_dims[0],data_dims[1],CV_8UC3);
			for(int i = 0; i < data_dims[0]; i++)
			{
				for(int j = 0; j < data_dims[1]; j++)
				{
					Vec3b& outPix = show.at<Vec3b>(i,j);
					outPix[0] = synthetic[synthetic_created][i][j][0];
					outPix[1] = synthetic[synthetic_created][i][j][1];
					outPix[2] = synthetic[synthetic_created][i][j][2];

					// outPix[0] = synthetic[synthetic_created][i][j][0];
					// outPix[1] = synthetic[synthetic_created][i][j][0];
					// outPix[2] = synthetic[synthetic_created][i][j][0];

				}
			}
			char name[100];
			sprintf(name,"synthetic of class %d",(int)minority_label);
			resize(show,show,size);
			imshow(name,show);

			waitKey(0);

			synthetic_created++;
		}


	}

	assert(synthetic_created == synthetic.size());

	/**************************
	*
	* make new idx
	*
	**************************/

	char dname[255], lname[255];
	sprintf(dname,"%s_data.idx",saveName);
	sprintf(lname,"%s_labels.idx",saveName);

	ofstream out_data, out_label;
	int32_t total_num = (int32_t)(synthetic_created + num_data);
	vector<int32_t> dimsdata = {total_num,data_dims[0],data_dims[1],data_dims[2]}; //num images, rows, cols, depth
	vector<int32_t> dimslabel = {total_num}; //num images
	initIDX(out_data,dname,0x08,dimsdata);
	initIDX(out_label,lname,label_type,dimslabel);
	for(int i = 0; i < num_data; i++)
	{
		//write data
		writeVectorToIDX(out_data,sample[i],0x08);

		//write label. This will only work for little endian order machines
		if(label_type != 0x0D && label_type != 0x0E) // integer type labels
		{
			long num = (long)sample_labels[i];
			streamsize size = getSize(label_type);
			out_label.write(reinterpret_cast<const char *>(&num),size);
		}
		else if(label_type == 0x0D) // float
		{
			float num = sample_labels[i];
			out_label.write(reinterpret_cast<const char *>(&num),sizeof(float));
		}
		else if(label_type == 0x0E) // double
		{
			out_label.write(reinterpret_cast<const char *>(&sample_labels[i]),sizeof(double));
		}
		else
		{
			printf("Unknown label type: %x\n",label_type);
			return 1;
		}
	}

	for(int i = 0; i < synthetic_created; i++)
	{
		//write data
		writeVectorToIDX(out_data,synthetic[i],0x08);

		//write label. This will only work for little endian order machines
		if(label_type != 0x0D && label_type != 0x0E) // integer type labels
		{
			long num = (long)minority_label;
			streamsize size = getSize(label_type);
			out_label.write(reinterpret_cast<const char *>(&num),size);
		}
		else if(label_type == 0x0D) // float
		{
			float num = minority_label;
			out_label.write(reinterpret_cast<const char *>(&num),sizeof(float));
		}
		else if(label_type == 0x0E) // double
		{
			out_label.write(reinterpret_cast<const char *>(&minority_label),sizeof(double));
		}
		else
		{
			printf("Unknown label type: %x\n",label_type);
			return 1;
		}
	}
}