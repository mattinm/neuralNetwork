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

using namespace std;
using namespace cv;

typedef std::vector<std::vector<std::vector<double> > > imVector;
bool showImages = false;

int main(int argc, char** argv)
{
	IDX idx(argv[1]);//, idx2(argv[1]);
	idx.printMetaData();
	// idx2.printMetaData();
	// idx += idx2;
	// idx.write("appendedtestIDX.idx", 0x0B);
}
