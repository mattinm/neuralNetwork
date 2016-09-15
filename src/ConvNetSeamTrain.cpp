/***************************
*
* This program:
* 	1a. Brings in a video from database.
*	1b. OR brings in a file of already seamcarved images with truth labels.
*	2.  Trains a cnn over the images.
*	3.  Makes file saying where training images came from.
*
*
****************************/


//ConvNet
#include <ConvNetCL.h>
#include <ConvNetSeam.h>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//MySQL
#include <mysql.h>

//Other
#include <iostream>
#include <fstream>
#include <vector>


int main(int argc, const char **argv)
{
	if(argc == 1)
	{
		printf("Usage: ./ConvNetSeamTrain \n");
		printf(" -cnn=<cnn_config>        Sets CNN architecture.\n");
		printf(" -video=<video_id>        Picks a video to use for training. Must be in database. Can be used multiple times.\n");
		printf(" -species=<species_num>   Sets species to grab videos of\n");
		printf(" -max_videos=<int>        Max videos to bring in for training.\n");
		printf(" -max_time=<double>       Max number of hours of video to train on\n");
		printf(" -images=<path_to_images> Picks path_to_images for training. Can be used multiple times.\n");
		printf(" -device=<device_num>     OpenCL device to run CNN on\n");

		printf(" -train_as_is\n");
		printf(" -train_equal_prop\n");
		return 0;
	}

	//1a. Bring in video from database
	
}