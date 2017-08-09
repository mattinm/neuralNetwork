#include "opencv2/imgproc/imgproc.hpp" //used for showing images being read in from IDX
#include "opencv2/highgui/highgui.hpp"
#include "IDX.h"
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if(argc != 7)
	{
		printf("Usage: ./DataVisualizer data.idx label.idx mat_width mat_height classNum baseOutputName\n");
		return 0;
	}

	IDX<unsigned char> data_idx(argv[1]);
	IDX<int> label_idx(argv[2]);
	int width = atoi(argv[3]), height = atoi(argv[4]), depth = 3;
	int classNum = atoi(argv[5]);
	const char * baseOutputName = argv[6];

	vector<int> labels;
	label_idx.getFlatData(labels);

	assert(label_idx.getNumData() == data_idx.getNumData());
	for(int i = labels.size() - 1; i >= 0; i--)
	{
		if(labels[i] != classNum)
		{
			label_idx.erase(i);
			data_idx.erase(i);
		}
	}

	vector<vector<unsigned char> >* data = data_idx.data();
	if(data->size() == 0)
	{
		printf("No data left. Exiting\n");
		return 0;
	}
	vector<int32_t> dims;
	data_idx.getDims(dims);
	if(dims.size() != 3)
	{
		printf("Error: must have 3D dims because going to image.");
		return 1;
	}
	if(dims[2] != 3)
	{
		printf("Error: only rgb images are currently supported\n");
		return 2;
	}

	int usable_height = height - height % dims[0];
	int usable_width = width - width % dims[1];

	printf("uh = %d uw = %d\n", usable_height,usable_width);

	size_t pixels_needed = data->size() * data->at(0).size();
	int num_mats = ceil((float)pixels_needed / (usable_width * usable_height * 3));
	vector<Mat> mats(num_mats);

	int curImage = 0;

	printf("Making %d images\n", num_mats);

	int endheight = usable_height - dims[0] + 1;
	int endwidth = usable_width - dims[1] + 1;
	for(int m = 0; m < mats.size(); m++)
	{
		printf("m %d\n", m);
		mats[m].create(height,width,CV_8UC3);
		printf("rows %d cols %d\n", mats[m].rows, mats[m].cols);
		for(int j = 0; j < endheight; j += dims[1])
		{
			for(int i = 0; i < endwidth; i += dims[0])
			{
				if(curImage >= data->size())
					break;
				int curPixel = 0;
				for(int ii = 0; ii < dims[1]; ii++)
				{
					for(int jj = 0; jj < dims[0]; jj++)
					{
						Vec3b& pix = mats[m].at<Vec3b>(j + jj, i + ii);//(i + ii,j + jj);
						pix[2] = data->at(curImage)[jj * dims[1] * depth + ii * depth + 0];
						pix[1] = data->at(curImage)[jj * dims[1] * depth + ii * depth + 1];
						pix[0] = data->at(curImage)[jj * dims[1] * depth + ii * depth + 2];
					}
				}
				curImage++;
			}
		}
		stringstream ss;
		ss << baseOutputName << "_class" << classNum << "im" << m << ".png";
		imwrite(ss.str(), mats[m]);
	}


}