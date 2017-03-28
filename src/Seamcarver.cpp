#include "Seamcarver.h"

//#include <cmath>
#include <iostream>
//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//would be needed for OpenCL kernels
//#include <fstream>

Seamcarver::Seamcarver(){}

Seamcarver::~Seamcarver()
{
	destroyMem();
}

int Seamcarver::POSITION(int x, int y)
{
	if(((y * window_width) + x) >= window_size || ((y * window_width) + x) < 0)
		printf("carve position overflow\n");
	return ((y * window_width) + x);
}

int Seamcarver::POSITION(int x, int y, int z)
{
	if(((y * window_width * 3) + x*3) + z >= window_size*3 || ((y * window_width * 3) + x*3) + z < 0)
		printf("carve position3 overflow\n");
	return ((y * window_width * 3) + x*3) + z;
}

void Seamcarver::init(const cv::Mat& source)
{
	if(source.empty())
		printf("empty mat given to seamcarver\n");
	window_width = source.cols;
	window_height = source.rows;
	window_size = window_width * window_height;
	makeMem();
	calcGreyscale(source);
	calcGradient();
}

//prereq: have window_width/height/size correct from init
void Seamcarver::makeMem()
{

	if(window_width <= owidth && window_height <= oheight)
		return;
	if(owidth != -1)
	{
		destroyMem(); // sets owidth and oheight back to -1, so change to new width and height
	}
	if(owidth == -1) // will always be true at first run and if above if statement runs
	{
		owidth = window_width;
		oheight = window_height;
	}
	// printf("making mem\n");

	// printf("make Mem w %d h %d s %d\n", window_width, window_height, window_size);
	
	image = new int[window_size * 3];
	vdirs = new char[window_size]; // 0 is right, 1 is down-right, -1 is up-right
	hdirs = new char[window_size]; // 0 is 
	greyscale = new float[window_size];
	vals = new float[window_size];
	vcosts = new float[window_size];
	hcosts = new float[window_size];
	vseam = new int[window_height];
	hseam = new int[window_width];
	if(image == nullptr || vdirs == nullptr || hdirs == nullptr || greyscale == nullptr || vals == nullptr || vcosts == nullptr || hcosts == nullptr
		|| vseam == nullptr || hseam == nullptr)
		printf("nullptr in make mem!\n");
}

void Seamcarver::destroyMem()
{
	if(owidth != -1)
	{
		// printf("destroyMem\n");
		delete image;
		delete vdirs;
		delete hdirs;
		delete greyscale;
		delete vals;
		delete vcosts;
		delete hcosts;
		delete vseam;
		delete hseam;

		owidth = -1;
		oheight = -1;
	}
}

void Seamcarver::calcGreyscale(const cv::Mat& source)
{
	//get average of all color channels and use as value for each pix.
	//also put unaltered into the image array

	//x is the row (height), y is the col(width)
	for(int y = 0; y < window_width; y++)
		for(int x = 0; x < window_height; x++)
		{
			const cv::Vec3b& pix = source.at<cv::Vec3b>(x,y);
			int pos = POSITION(y,x);
			image[POSITION(y,x,0)] = pix[0];
			image[POSITION(y,x,1)] = pix[1];
			image[POSITION(y,x,2)] = pix[2];

			greyscale[pos] = (pix[0] + pix[1] + pix[2])/3;
		}
}

void Seamcarver::calcGradient()
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
            if(y > 0)                   result += fabs(greyscale[pos] - greyscale[POSITION(x, y-1)]);
            if(y < window_height - 1)   result += fabs(greyscale[pos] - greyscale[POSITION(x, y+1)]);

            vals[pos] = result;
		}
}

bool Seamcarver::carve_v(const cv::Mat& source, int numSeams, cv::Mat& dest)
{
	// printf("start carve\n");
	if(source.empty())
	{
		printf("carve_v source empty\n");
		return false;
	}
	if(source.cols <= numSeams) //if asked to carve more than possible, don't
	{
		printf("carve failed\n");
		return false;
	}

	init(source);

	unsigned int count = 0;
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
        float min_val = 200000000;
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
			cv::Vec3b& pix = dest.at<cv::Vec3b>(i,j);
			pix[0] = image[POSITION(j,i,0)];
			pix[1] = image[POSITION(j,i,1)];
			pix[2] = image[POSITION(j,i,2)];
		}
	}

	// printf("end carve\n");

	return true;
}

bool Seamcarver::carve_h(const cv::Mat& source, int numSeams, cv::Mat& dest)
{
	if(source.rows <= numSeams)
		return false;

	init(source);

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

		//init right col
		for(int y = 0; y < window_height - count; y++)
		{
			int pos = POSITION(window_width - 1, y);
			hcosts[pos] = vals[pos];
		}

		//calc rest of costs and dirs
		for(int x = window_width - 2; x >= 0; x--)
		{

			//do top pixel
			if(hcosts[POSITION(x+1,0)] < hcosts[POSITION(x+1,1)])
			{
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,0)];
				hdirs[POSITION(x,0)] = 0; // right
			}
			else
			{
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,1)];
				hdirs[POSITION(x,0)] = 1; //down
			}

			//middle
			int y;
			for(y = 1; y < window_height - count - 1; y++)
			{
				float cost_up   = hcosts[POSITION(x+1, y-1)];
				float cost_left = hcosts[POSITION(x+1, y  )]; //really goes to the right
				float cost_down = hcosts[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);

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
			cv::Vec3b& pix = dest.at<cv::Vec3b>(i,j);
			pix[0] = image[POSITION(j,i,0)];
			pix[1] = image[POSITION(j,i,1)];
			pix[2] = image[POSITION(j,i,2)];
		}
	}

	return true;
}

bool Seamcarver::carve_b(const cv::Mat& source, int vseams, int hseams, cv::Mat& dest)
{
	if(source.rows <= hseams || source.cols <= vseams)
		return false;

	init(source);

	int vcount = 0, hcount = 0;
	while(vcount < vseams || hcount < hseams)
	{
		float vmin = 20000000, hmin = 20000000;
		//calc horizontal stuff
		if(hcount < hseams)
		{
			//init right col
			for(int y = 0; y < window_height - hcount; y++)
			{
				int pos = POSITION(window_width - 1, y);
				hcosts[pos] = vals[pos];
			}

			//calc rest of costs and dirs
			for(int x = window_width - 2; x >= 0; x--)
			{

				//do top pixel
				if(hcosts[POSITION(x+1,0)] < hcosts[POSITION(x+1,1)])
				{
					hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,0)];
					hdirs[POSITION(x,0)] = 0; // right
				}
				else
				{
					hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,1)];
					hdirs[POSITION(x,0)] = 1; //down
				}

				//middle
				int y;
				for(y = 1; y < window_height - hcount - 1; y++)
				{
					float cost_up   = hcosts[POSITION(x+1, y-1)];
					float cost_left = hcosts[POSITION(x+1, y  )]; //really goes to the right
					float cost_down = hcosts[POSITION(x+1, y+1)];
					int mypos = POSITION(x,y);

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
			for(int y = 0; y < window_height - hcount; y++)
			{
				if(hcosts[POSITION(0,y)] < hmin)
				{
					hmin = hcosts[POSITION(0,y)];
					hseam[0] = y;
				}
			}
		}
		//end calc horizontal stuff

		//calc vertical stuff
		if(vcount < vseams)
		{
			//init bottom row
			// printf("Init bottom row. %d cols\n", (window_width - count));
			for(int x = 0; x < (window_width - vcount); x++)
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
				for(x = 1; x < window_width - vcount - 1; x++)
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
	        for(int x = 0; x < window_width - vcount; x++)
	        {
	        	//printf("vcosts[%d] = %f\n", x,vcosts[POSITION(x,0)]);
	        	if(vcosts[POSITION(x,0)] < vmin)
	        	{
	        		vmin = vcosts[POSITION(x,0)];
	        		vseam[0] = x;
	        	}
	        }
	    }
		//end calc vertical stuff

		
		if((hcount < hseams && hmin < vmin))// || vcount >= vseams) //cheaper to do horizontal
		{
			for(int x = 1; x < window_width; x++)
				hseam[x] = hseam[x-1] + hdirs[POSITION(x-1,hseam[x-1])];

			//remove one horizontal seam
			for(int x = 0; x < window_width; x++)
			{
				int y;
				for(y = hseam[x]; y < window_height - hcount - 1; y++)
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
			hcount++;
		}
		else //cheaper to do vertical
		{
			for(int y = 1; y < window_height; y++)
	        	vseam[y] = vseam[y-1] + vdirs[POSITION(vseam[y-1], y-1)];

	        //remove one seam
	        for(int y = 0; y < window_height; y++)
	        {
	        	int x;
	        	// printf("x %d, y %d\n", vseam[y], y);
	        	for(x = vseam[y]; x < window_width - vcount - 1; x++)
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
	        vcount++;
		}
	}
	dest.create(source.rows-hcount,source.cols-vcount, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			cv::Vec3b& pix = dest.at<cv::Vec3b>(i,j);
			pix[0] = image[POSITION(j,i,0)];
			pix[1] = image[POSITION(j,i,1)];
			pix[2] = image[POSITION(j,i,2)];
		}
	}

	return true;
}