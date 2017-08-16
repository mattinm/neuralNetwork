#include "MSILocations.h"
#include <fstream>
#include "ConvNetCommon.h"

using namespace std;

Box::Box(){}

Box::Box(int32_t species_id, int32_t x, int32_t y, int32_t w, int32_t h, uint64_t user_high, uint64_t user_low)
{
	load(species_id,x,y,w,h,user_high,user_low);
}

void Box::load(int32_t species_id, int32_t x, int32_t y, int32_t w, int32_t h, uint64_t user_high, uint64_t user_low)
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
	this->user_high = user_high;
	this->user_low = user_low;
}

string Box::toString()
{
	char buf[200];
	sprintf(buf,"species: %7d, x: %d, y: %d, w: %d, h: %d, ex: %d, ey: %d", species_id,x,y,w,h,ex,ey);
	return string(buf);
}

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

void readLocationsFile(const char* filename, map<int, MSI> &locations)
{
	ifstream in(filename,ios::binary);
	int numMSIs = convnet::readInt(in);
	for(int i = 0; i < numMSIs; i++)
	{
		int msi = convnet::readInt(in);
		// if(msi > 5000)
		// 	printf("found msi > 5000 - %d\n",msi);
		int numBoxes = convnet::readInt(in);
		// locations[msi] = vector<Box>(numBoxes);
		locations[msi].init(msi,numBoxes);
		for(int j = 0; j < numBoxes; j++)
			locations[msi].boxes[j].load(
				convnet::readInt(in),
				convnet::readInt(in),
				convnet::readInt(in),
				convnet::readInt(in),
				convnet::readInt(in),
				convnet::readUInt64(in),
				convnet::readUInt64(in));
	}
}

int getMSI(string filename)
{
	int startMSIIndex = filename.rfind("msi");
	int nextUnderscore = filename.find("_",startMSIIndex);
	if(nextUnderscore == string::npos)
		nextUnderscore = filename.find(".",startMSIIndex);
	// printf("%s\n", filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3).c_str());
	try
	{
		return stoi(filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3));
	} catch(...)
	{
		int lastSlash = filename.rfind('/') + 1;
		try
		{
			return stoi(filename.substr(lastSlash,filename.rfind('.') - lastSlash));
		}
		catch(...)
		{
			printf("Error with stoi on '%s' substrings '%s' and '%s'\n", filename.c_str(), filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3).c_str(), filename.substr(lastSlash,filename.rfind('.') - lastSlash).c_str());
			exit(1);
		}
	}
	return -1;
}