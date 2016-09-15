//ConvNet
#include "ConvNetCL.h"

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//Other
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

int window_width;

static int POSITION(int x, int y, int z) //assumes 3 channels
{
	return ((y * window_width * 3) + x*3) + z;
}

static int POSITION(int x, int y) {
    return ((y * window_width) + x);
}

/*void vectorResize(vector<vector<double> >& vect, int x, int y)
{
	vect.resize(x);
	for(int i = 0; i < y; i++)
		vect[i].resize(y);
}*/

bool seamcarve_vf(int numSeams, const Mat& source, Mat& dest)
{
	static int owidth = source.cols;
	static int oheight = source.rows;
	window_width = source.cols;
	int window_height = source.rows;

	if(owidth != window_width || oheight != window_height)
	{
		printf("The current dimensions do not match the original\n");
		return false;
	}

	if(window_width < numSeams)
		return false;

	unsigned long window_size = source.rows * source.cols;
	//get average of all color channels and put in vector
	static int *image = new int[window_size * 3];
	static int *dirs = new int[window_size]; // 0 is up, 1 is up-right, -1 is up-left
	static float *greyscale = new float[window_size];
	static float *vals = new float[window_size];
	static float *costs = new float[window_size];
	static int *seam = new int[window_height];

	//get average of all color channels and use as value for each pix.
	//also put unaltered into the image ptr
	for(int y = 0; y < window_height; y++)
		for(int x = 0; x < window_width; x++)
		{
			const Vec3b& pix = source.at<Vec3b>(y,x);
			int pos = POSITION(x,y);
			image[POSITION(x,y,0)] = pix[0];
			image[POSITION(x,y,1)] = pix[1];
			image[POSITION(x,y,2)] = pix[2];

			greyscale[pos] = (pix[0] + pix[1] + pix[2])/3;
		}

	//calc gradient (val) for each pixel
	for(int y = 0; y < window_height; y++)
		for(int x = 0; x < window_width; x++)
		{
			float result = 0;
			int pos = POSITION(x,y);
			if(x > 0) 					result += fabs(greyscale[pos] - greyscale[POSITION(x-1, y)]);
			if(x  < window_width - 1)	result += fabs(greyscale[pos] - greyscale[POSITION(x+1, y)]);
            if (y > 0)                  result += fabs(greyscale[pos] - greyscale[POSITION(x, y-1)]);
            if (y < window_height - 1)  result += fabs(greyscale[pos] - greyscale[POSITION(x, y+1)]);

            vals[pos] = result;
		}

	int count = 0;
	while(count < numSeams)
	{
		//init top row
		for(int x = 0; x < (window_width - count); x++)
		{
			int pos = POSITION(x,window_height - 1);
			costs[pos] = vals[pos];
			//dirs[pos] = 0; // doesn't really matter
		}

		//calc rest of costs and dirs
		for(int y = window_height - 2; y >= 0; y--)
		{
			//do left side
			if(costs[POSITION(0, y+1)] < costs[POSITION(1,y+1)])
			{
				costs[POSITION(0, y)] = vals[POSITION(0,y)] + costs[POSITION(0, y+1)];
				dirs[POSITION(0,y)] = 0; //up
			}
			else
			{
				costs[POSITION(0, y)] = vals[POSITION(0,y)] + costs[POSITION(1, y+1)];
				dirs[POSITION(0,y)] = 1;
			}

			//middle
			int x;
			for(x = 1; x < window_width - count - 1; x++)
			{
				float cost_left  = costs[POSITION(x-1, y+1)];
				float cost_up    = costs[POSITION(x  , y+1)];
				float cost_right = costs[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);

				if(cost_left < cost_up  && cost_left < cost_right) // cost_left is min
				{
					costs[mypos] = vals[mypos] + cost_left;
					dirs[mypos] = -1;
				}
				else if(cost_right < cost_up) //cost_right is min
				{
					costs[mypos] = vals[mypos] + cost_right;
					dirs[mypos] = 1;
				}
				else // up is min
				{
					costs[mypos] = vals[mypos] + cost_up;
					dirs[mypos] = 0;
				}
			}

			//right side
			int pos = POSITION(x, y);
			if (costs[POSITION(x, y+1)] < costs[POSITION(x-1, y+1)]) {
                costs[pos] = vals[pos] + costs[POSITION(x, y+1)];
                dirs [pos] = 0;
            } else {
                costs[pos] = vals[pos] + costs[POSITION(x-1, y+1)];
                dirs [pos] = -1;
            }		
        }//end calc costs and dirs

        //calc seam to remove
        float min_val = 20000000;
        for(int x = 0; x < window_width - count; x++)
        {
        	if(costs[POSITION(x,0)] < min_val)
        	{
        		min_val = costs[POSITION(x,0)];
        		seam[0] = x;
        	}
        }
        for(int y = 1; y < window_height; y++)
        	seam[y] = seam[y-1] + dirs[POSITION(seam[y-1], y-1)];

        //remove one seam
        for(int y = 0; y < window_height; y++)
        {
        	int x;
        	for(x = seam[y]; x < window_width - count - 1; x++)
        	{
        		image[POSITION(x,y,0)] = image[POSITION(x+1,y,0)];
        		image[POSITION(x,y,1)] = image[POSITION(x+1,y,1)];
        		image[POSITION(x,y,2)] = image[POSITION(x+1,y,2)];
        	}

        	image[POSITION(x,y,0)] = 0;
        	image[POSITION(x,y,1)] = 0;
        	image[POSITION(x,y,2)] = 0;

        	vals[POSITION(x,y)] = 0;
        }
        count++;
	}

	//put new image into mat
	dest.create(source.rows-numSeams,source.cols, CV_8UC3);
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION(j,i,0)];
			pix[1] = image[POSITION(j,i,1)];
			pix[2] = image[POSITION(j,i,2)];
		}
	}

	return true;
}
