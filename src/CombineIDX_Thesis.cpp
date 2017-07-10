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
#include "IDX.h"

#define BACKGROUND -1
#define WHITE_PHASE 2
#define BLUE_PHASE 1000000

using namespace std;
using namespace cv;

typedef std::vector<std::vector<std::vector<double> > > imVector;
bool showImages = false;

int main(int argc, char** argv)
{
	if(argc < 6)
	{
		printf("./CombineIDX_Thesis original_data.idx original_labels.idx retrain_data.idx retrain_labels.idx originalPercentage outbasename\n");
		printf("    originalPercentage is a double from 0.0 - 1.0 and is how much of the new training set should be from the original data\n");
		printf("    outbasename is a string for a filename. Actual files will be outbasename_data.idx outbasename_label.idx\n");
		printf("  Optional: Must come after required\n");
		printf("    --exclude=<int> Changes label of all excluded classes to -1 (background).\n");
		return 0;
	}

	//read in cmd line args into IDX objects and originalPercentage
	IDX<unsigned char> odata_idx(argv[1]),
	                   rdata_idx(argv[3]);
	IDX<int> olabel_idx(argv[2]),
	         rlabel_idx(argv[4]);
	printf("Brought in all idxs\n");

	odata_idx.printMetaData();
	olabel_idx.printMetaData();
	rdata_idx.printMetaData();
	rlabel_idx.printMetaData();
	double originalPercentage = atof(argv[5]);
	string outbasename(argv[6]);

	unordered_map<int,char> excludes;
	for(int i = 7; i < argc; i++)
	{
		string arg = string(argv[i]);
		if(arg.find("--exclude=") != string::npos)
			excludes[stoi(arg.substr(arg.find('=')+1))] = 1;
		else
		{
			printf("Unknown arg '%s'. Exiting\n", argv[i]);
			return 1;
		}
	}
	printf("got excludes\n");
	if(excludes.size() > 0)
	{
		for(int i = 0; i < olabel_idx.data()->size(); i++)
			if(excludes.find(olabel_idx[i][0]) != excludes.end())
			{
				olabel_idx[i][0] = -1;
			}
		for(int i = 0; i < rlabel_idx.data()->size(); i++)
			if(excludes.find(rlabel_idx[i][0]) != excludes.end())
			{
				rlabel_idx[i][0] = -1;
			}
	}

	//get labels in vectors for ease of use
	vector<int> olabels, rlabels;
	olabel_idx.getFlatData(olabels);
	rlabel_idx.getFlatData(rlabels);

	printf("got flat data\n");
	//fix any excluded classes


	//figure out how many of each class we have
	printf("figure out have\n");
	unordered_map<int, int> ohave, rhave; // how many of each class we have
	unordered_map<int, vector<int> > olocs, rlocs; // the locations of each class
	for(int i = 0; i < olabels.size(); i++)
	{
		ohave[olabels[i]]++;
		olocs[olabels[i]].push_back(i);
	}
	for(int i = 0; i < rlabels.size(); i++)
	{
		rhave[rlabels[i]]++;
		rlocs[rlabels[i]].push_back(i);
	}

	//figure out how many of each class we need, ideally we will use all the retrain data we have
	printf("figure out need\n");
	unordered_map<int, int> oneed, rneed = rhave; // how many of each class we need for the final idx
	double ratio = originalPercentage / (1 - originalPercentage);
	for(auto it = rneed.begin(); it != rneed.end(); it++)
	{
		if(it->first == WHITE_PHASE) // play it special to test theory
		{
			oneed[it->first] = ohave[it->first];
			rneed[it->first] = 0;
		}
		else
		{
			oneed[it->first] = (int)(ratio * it->second);
			//if don't have of a class in the original, that is strange and probably not right, but we will just use all of the retrain and ignore the ratio for that class
			if(ohave[it->first] == 0) 
			{
				printf("We have no examples of class %d in the original training set.\n", it->first);
				oneed[it->first] = 0;
			}
			else if(oneed[it->first] > ohave[it->first]) // if we need more than we have for the ratio, use less rneed
			{
				rneed[it->first] = (int)(ohave[it->first] / ratio);
				oneed[it->first] = ohave[it->first];
			}
		}
	}

	//using how many we need and have, figure out how many to erase
	printf("figure out erase\n");
	unordered_map<int, int> oerase, rerase;
	for(auto it = oneed.begin(); it != oneed.end(); it++)
		oerase[it->first] = ohave[it->first] - oneed[it->first];
	for(auto it = rneed.begin(); it != rneed.end(); it++)
		rerase[it->first] = rhave[it->first] - rneed[it->first];

	//randomly select indexes to erase
	printf("get indexes\n");
	vector<int> oindexes, rindexes; // need vector for IDX::erase
	unordered_map<int, char> oused, rused; // keep track of used indexes so no duplicates

	default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	

	for(auto it = oerase.begin(); it != oerase.end(); it++)
	{
		printf("From original: class %d - erase %d items\n", it->first, it->second);
		uniform_int_distribution<int> dis(0,ohave[it->first] - 1); //this is inclusive on both sides, hence the -1
		for(int i = 0; i < it->second; i++)
		{
			//gen rand number
			int index;
			do
			{
				// get random number in range of [0, num data for this class] then convert to actual index location
				index = olocs[it->first][dis(gen)]; 
			}
			while(oused.find(index) != oused.end()); //check against used numbers
			
			//add to used map and index to erase vector
			oused[index] = 1;
			oindexes.push_back(index);
		}
	}
	for(auto it = rerase.begin(); it != rerase.end(); it++)
	{
		printf("From retrain: class %d - erase %d items\n", it->first, it->second);
		uniform_int_distribution<int> dis(0,rhave[it->first] - 1); //this is inclusive on both sides, hence the -1
		for(int i = 0; i < it->second; i++)
		{
			//gen rand number
			int index;
			do
			{
				// get random number in range of [0, num data for this class] then convert to actual index location
				index = rlocs[it->first][dis(gen)]; 
			}
			while(rused.find(index) != rused.end()); //check against used numbers
			
			//add to used map and index to erase vector
			rused[index] = 1;
			rindexes.push_back(index);
		}
	}


	//erase data
	printf("erase\n");
	odata_idx.erase(oindexes);
	olabel_idx.erase(oindexes);
	rdata_idx.erase(rindexes);
	rlabel_idx.erase(rindexes);

	//combine idxs
	printf("combine\n");
	odata_idx += rdata_idx;
	olabel_idx += rlabel_idx;

	printf("New IDX metadata\n");
	odata_idx.printMetaData();
	olabel_idx.printMetaData();

	//write out new idxs
	printf("write\n");
	odata_idx.write<unsigned char>(outbasename + string("_data.idx"));
	olabel_idx.write<int>(outbasename + string("_label.idx"));

}
