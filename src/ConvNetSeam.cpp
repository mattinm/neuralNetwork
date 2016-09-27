#include "ConvNetSeam.h"

//ConvNet
//#include "ConvNetCL.h"

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

int owidth = -1, oheight = -1;

int window_width;
int *image;
int *vdirs, *hdirs;
float *greyscale;
float *vals;
float *vcosts, *hcosts;
int *vseam, *hseam;
bool __seam_inited = false;

static inline int POSITION(int x, int y, int z) //assumes 3 channels
{
	return ((y * window_width * 3) + x*3) + z;
}

static inline int POSITION(int x, int y) {
    return ((y * window_width) + x);
}

void seamcarve_cleanup()
{
	printf("Cleaning mem\n");
	delete image;
	delete vdirs;
	delete hdirs;
	delete greyscale;
	delete vals;
	delete vcosts;
	delete hcosts;
	delete vseam;
	delete hseam;
}


void calcGreyscale(const Mat& source, int window_height)
{
	//most of the stuff is global pointers so it'll be ok

	//get average of all color channels and use as value for each pix.
	//also put unaltered into the image ptr

	//x is the row (height), y is the col(width)
	for(int y = 0; y < window_width; y++)
		for(int x = 0; x < window_height; x++)
		{
			const Vec3b& pix = source.at<Vec3b>(x,y);
			int pos = POSITION(y,x);
			image[POSITION(y,x,0)] = pix[0];
			image[POSITION(y,x,1)] = pix[1];
			image[POSITION(y,x,2)] = pix[2];

			greyscale[pos] = (pix[0] + pix[1] + pix[2])/3;
		}
}

void calcGradient(int window_height)
{
	//calc gradient (val) for each pixel
	//y is row (height), x is column (width)
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
}

//seamcarve both ways, vertical then horizontal
bool seamcarve_both_vth(int vseams, int hseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_vf(vseams,source,temp);
	if(!good) return false;
	return seamcarve_hf(hseams,temp,dest);
}

//seamcarve both ways, horizontal then vertical
bool seamcarve_both_htv(int hseams, int vseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_hf(hseams,source,temp);
	if(!good) return false;
	return seamcarve_vf(vseams,temp,dest);
}

bool seamcarve_hf(int numSeams, const Mat& source, Mat& dest)
{
	window_width = source.cols;
	int window_height = source.rows;
	unsigned long window_size = source.rows * source.cols;

	//only redo the memory stuff if it changes
	if(owidth != window_width || oheight != window_height)
	{
		// printf("Doing mem in hf\n");
		if(__seam_inited) // if something, delete it
		{	
			// printf("Cleaning mem\n");
			seamcarve_cleanup();
		}
		__seam_inited = true;

		owidth = window_width;
		oheight = window_height;

		image = new int[window_size * 3];
		vdirs = new int[window_size]; // 0 is right, 1 is down-right, -1 is up-right
		// printf("hdirs should init with size of %lu\n", window_size);
		hdirs = new int[window_size];
		greyscale = new float[window_size];
		vals = new float[window_size];
		vcosts = new float[window_size];
		hcosts = new float[window_size];
		vseam = new int[window_height];
		hseam = new int[window_width];
	}

	if(window_height < numSeams)
		return false;

	calcGreyscale(source, window_height);//only need height bc width is global
	calcGradient(window_height);

	//x is col (width), y is row (height)

	int count = 0;
	// printf("Starting loop\n");
	while(count < numSeams)
	{
		//show current image
		// dest.create(source.rows-count,source.cols, CV_8UC3);
		// printf("rows: %d, cols: %d\n", source.rows-count,source.cols);
		// //i is row, j is col
		// for(int i = 0; i < dest.rows; i++)
		// {
		// 	for(int j = 0; j < dest.cols; j++)
		// 	{
		// 		Vec3b& pix = dest.at<Vec3b>(i,j);
		// 		pix[0] = image[POSITION(j,i,0)];
		// 		pix[1] = image[POSITION(j,i,1)];
		// 		pix[2] = image[POSITION(j,i,2)];
		// 	}
		// }

		// imshow("test",dest);
		// waitKey(1);
		// printf("Press enter for next frame\n");
		// getchar();

		// printf("count: %d\n",count);
		//init right row
		// printf("Right row\n");
		for(int y = 0; y < window_height - count; y++)
		{
			int pos = POSITION(window_width - 1, y);
			hcosts[pos] = vals[pos];
		}

		// printf("Rest costs/dirs\n");
		//calc rest of costs and dirs
		for(int x = window_width - 2; x >= 0; x--)
		{
			// printf("x: %d\n", x);
			// printf("begin ad: %x\n",&(vdirs[304]));
			// vdirs[304] = 0;
			//do top pixel
			if(hcosts[POSITION(x+1,0)] < hcosts[POSITION(x+1,1)])
			{
				// printf("In if\n");
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,0)];
				// printf("dir to right %d\n",POSITION(x,0));
				//vdirs[304] = 0;
				// printf("set 0\n");
				// int newIndex = POSITION(x,0);
				// printf("if ad: %x\n", &(vdirs[newIndex]));
				hdirs[POSITION(x,0)] = 0; // right
				// printf("done dirs\n");
			}
			else
			{
				// printf("In else %d\n",x);
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,1)];
				hdirs[POSITION(x,0)] = 1; //down
				// vdirs[304] = 0;
				// printf("set 0 else\n");
			}
			//vdirs[304] = 0;
			// printf("ad: %x\n", &(vdirs[304]));
			// printf("Middle\n");
			//middle
			int y;
			for(y = 1; y < window_height - count - 1; y++)
			{
				float cost_up   = hcosts[POSITION(x+1, y-1)];
				float cost_left = hcosts[POSITION(x+1, y  )]; //really goes to the right
				float cost_down = hcosts[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);
				// if(mypos == 304)
				// 	printf("----------------hf middle 304\n");
				// else 
				// 	printf("not 304\n");

				if(cost_up < cost_left && cost_up < cost_down) // cost_up is min
				{
					hcosts[mypos] = vals[mypos] + cost_up;
					hdirs[mypos] = -1;
				}
				else if(cost_down < cost_left) // down is min
				{
					hcosts[mypos] = vals[mypos] + cost_down;
					hdirs[mypos] = 1;
				}
				else //straight left/right is min
				{
					hcosts[mypos] = vals[mypos] + cost_left;
					hdirs[mypos] = 0;
				}
			}
			// printf("Bottom\n");
			//bottom pixel
			int pos = POSITION(x,y);
			if(hcosts[POSITION(x+1, y)] < hcosts[POSITION(x+1,y-1)])
			{
				hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y)];
				hdirs[pos] = 0;
			}
			else
			{
				hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y-1)];
				hdirs [pos] = -1;
			}
		}//end calc costs and dirs

		// printf("Calc seam\n");

		//calc seams to remove
		float min_val = 20000000;
		for(int y = 0; y < window_height - count; y++)
		{
			if(hcosts[POSITION(0,y)] < min_val)
			{
				min_val = hcosts[POSITION(0,y)];
				hseam[0] = y;
			}
		}
		for(int x = 1; x < window_width; x++)
			hseam[x] = hseam[x-1] + hdirs[POSITION(x-1,hseam[x-1])];

		//remove one horizontal seam
		for(int x = 0; x < window_width; x++)
		{
			int y;
			for(y = hseam[x]; y < window_height - count - 1; y++)
			{
				image[POSITION(x,y,0)] = image[POSITION(x,y+1,0)];
				image[POSITION(x,y,1)] = image[POSITION(x,y+1,1)];
				image[POSITION(x,y,2)] = image[POSITION(x,y+1,2)];

				vals[POSITION(x,y)] = vals[POSITION(x,y+1)];
			}

			image[POSITION(x,y,0)] = 0;
			image[POSITION(x,y,1)] = 0;
			image[POSITION(x,y,2)] = 0;

			vals[POSITION(x,y)] = 0;
		}
		count++;
	}


	//put new image in dest mat
	dest.create(source.rows-count,source.cols, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
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

	// // printf("Put in new mat\n");
	// //put new image into mat
	// // printf("rows %d cols %d\n", source.rows-numSeams,source.cols);
	// dest.create(source.rows - numSeams, source.cols, CV_8UC3);
	// for(int i = 0; i < dest.cols; i++)
	// {
	// 	for(int j = 0; j < dest.rows; j++)
	// 	{
	// 		// printf("i %d j %d\n",i,j );
	// 		Vec3b& pix = dest.at<Vec3b>(j,i);
	// 		// printf("got pix\n");
	// 		pix[0] = image[POSITION(i,j,0)];
	// 		pix[1] = image[POSITION(i,j,1)];
	// 		pix[2] = image[POSITION(i,j,2)];
	// 		// printf("got POSITION\n");
	// 	}
	// }
	// printf("complete\n");

	return true;
}

bool seamcarve_vf(int numSeams, const Mat& source, Mat& dest)
{
	window_width = source.cols;
	int window_height = source.rows;
	unsigned long window_size = source.rows * source.cols;

	//only redo the memory stuff if it changes
	if(owidth != window_width || oheight != window_height)
	{
		printf("Doing mem in vf\n");
		if(__seam_inited) // if something, delete it
		{	
			seamcarve_cleanup();
		}		
		__seam_inited = true;

		owidth = window_width;
		oheight = window_height;

		image = new int[window_size * 3];
		vdirs = new int[window_size]; // 0 is up, 1 is up-right, -1 is up-left
		hdirs = new int[window_size];
		greyscale = new float[window_size];
		vals = new float[window_size];
		// printf("window_size: %lu\n", );
		vcosts = new float[window_size];
		hcosts = new float[window_size];
		vseam = new int[window_height];
		hseam = new int[window_width];
	}

	if(window_width < numSeams)
		return false;

	calcGreyscale(source, window_height);
	calcGradient(window_height);

	// namedWindow("test",WINDOW_AUTOSIZE);

	//printf("Starting loop\n");

	int count = 0;
	while(count < numSeams)
	{
		//show current image
		// dest.create(source.rows,source.cols-count, CV_8UC3);
		// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
		// //i is row, j is col
		// for(int i = 0; i < dest.rows; i++)
		// {
		// 	for(int j = 0; j < dest.cols; j++)
		// 	{
		// 		Vec3b& pix = dest.at<Vec3b>(i,j);
		// 		pix[0] = image[POSITION(j,i,0)];
		// 		pix[1] = image[POSITION(j,i,1)];
		// 		pix[2] = image[POSITION(j,i,2)];
		// 	}
		// }

		// imshow("test",dest);
		// waitKey(1);
		// printf("Press enter for next frame\n");
		// getchar();

		//x is the col, y is the row

		//init bottom row
		// printf("Init bottom row. %d cols\n", (window_width - count));
		for(int x = 0; x < (window_width - count); x++)
		{
			int pos = POSITION(x,window_height - 1);
			//printf("%d\n", pos);
			vcosts[pos] = vals[pos];
			//dirs[pos] = 0; // doesn't really matter
		}

		//calc rest of costs and dirs
		for(int y = window_height - 2; y >= 0; y--)
		{
			//do left side
			if(vcosts[POSITION(0, y+1)] < vcosts[POSITION(1,y+1)])
			{
				vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(0, y+1)];
				vdirs[POSITION(0,y)] = 0; //up
			}
			else
			{
				vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(1, y+1)];
				vdirs[POSITION(0,y)] = 1;
			}

			//middle
			int x;
			for(x = 1; x < window_width - count - 1; x++)
			{
				float cost_left  = vcosts[POSITION(x-1, y+1)];
				float cost_up    = vcosts[POSITION(x  , y+1)];
				float cost_right = vcosts[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);
				// if(mypos == 304)
				// 	printf("Got 304\n");

				if(cost_left < cost_up  && cost_left < cost_right) // cost_left is min
				{
					vcosts[mypos] = vals[mypos] + cost_left;
					vdirs[mypos] = -1;
				}
				else if(cost_right < cost_up && cost_right < cost_left) //cost_right is min
				{
					vcosts[mypos] = vals[mypos] + cost_right;
					vdirs[mypos] = 1;
				}
				else // up is min
				{
					vcosts[mypos] = vals[mypos] + cost_up;
					vdirs[mypos] = 0;
				}
			}

			//right side
			int pos = POSITION(x, y);
			if (vcosts[POSITION(x, y+1)] < vcosts[POSITION(x-1, y+1)]) {
                vcosts[pos] = vals[pos] + vcosts[POSITION(x, y+1)];
                vdirs [pos] = 0;
            } else {
                vcosts[pos] = vals[pos] + vcosts[POSITION(x-1, y+1)];
                vdirs [pos] = -1;
            }		
        }//end calc costs and dirs

        //calc seam to remove
        float min_val = 20000000;
        for(int x = 0; x < window_width - count; x++)
        {
        	//printf("vcosts[%d] = %f\n", x,vcosts[POSITION(x,0)]);
        	if(vcosts[POSITION(x,0)] < min_val)
        	{
        		min_val = vcosts[POSITION(x,0)];
        		vseam[0] = x;
        	}
        }
        for(int y = 1; y < window_height; y++)
        	vseam[y] = vseam[y-1] + vdirs[POSITION(vseam[y-1], y-1)];

        //remove one seam
        for(int y = 0; y < window_height; y++)
        {
        	int x;
        	// printf("x %d, y %d\n", vseam[y], y);
        	for(x = vseam[y]; x < window_width - count - 1; x++)
        	{
        		image[POSITION(x,y,0)] = image[POSITION(x+1,y,0)];
        		image[POSITION(x,y,1)] = image[POSITION(x+1,y,1)];
        		image[POSITION(x,y,2)] = image[POSITION(x+1,y,2)];

        		vals[POSITION(x,y)] = vals[POSITION(x+1,y)];
        	}

        	image[POSITION(x,y,0)] = 0;
        	image[POSITION(x,y,1)] = 0;
        	image[POSITION(x,y,2)] = 0;

        	vals[POSITION(x,y)] = 0;
        }
        count++;
	}

	dest.create(source.rows,source.cols-count, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
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

	//put new image into mat
	// dest.create(source.rows,source.cols-numSeams, CV_8UC3);
	// for(int i = 0; i < dest.rows; i++)
	// {
	// 	for(int j = 0; j < dest.cols; j++)
	// 	{
	// 		Vec3b& pix = dest.at<Vec3b>(j,i);
	// 		pix[0] = image[POSITION(i,j,0)];
	// 		pix[1] = image[POSITION(i,j,1)];
	// 		pix[2] = image[POSITION(i,j,2)];
	// 	}
	// }

	return true;
}

