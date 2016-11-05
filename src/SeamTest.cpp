#include <stdio.h>
#include <iostream>
#include "ConvNetSeam.h"
#include <random>
#include <time.h>
#include "ConvNetEvent.h"

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	Event e("parent behavior", 65, 120);

	printf("%s %d %d %s %s\n", e.type.c_str(), e.starttime, e.endtime, e.starttime_string.c_str(), e.endtime_string.c_str());










	// time_t starttime;
	// namedWindow("Original image",WINDOW_AUTOSIZE);
	// namedWindow("Seamcarved image",WINDOW_AUTOSIZE);
	// namedWindow("Horizontal seamcarved image",WINDOW_AUTOSIZE);
	// const Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	// int vseams = image.cols - atoi(argv[2]);
	// int hseams = image.rows - atoi(argv[3]);
	// imshow("Original image",image);
	// waitKey(1);

	// seamcarve_setDevice(0);

	// Size cvSize = Size(120,120);


	// Mat seamImage;
	// printf("Starting seamcarve_vf\n");
	// if(!seamcarve_vf(vseams,image,seamImage))
	// 	printf("Seamcarve vf returned false\n");;
	// imshow("Seamcarved image",seamImage);
	// printf("vseam row %d col %d\n\n", seamImage.rows, seamImage.cols);
	// waitKey(1);

	// Mat revseamImage;
	// printf("Starting seamcarve_vf\n");
	// if(!seamcarve_vfRandom(vseams,image,revseamImage))
	// 	printf("Seamcarve vf rev returned false\n");;
	// imshow("RevSeamcarved image",revseamImage);
	// printf("vseamrev row %d col %d\n\n", revseamImage.rows, revseamImage.cols);
	// waitKey(1);

	// Mat forimage;
	// seamcarve_vf(vseams, image, forimage);
	// int len = 15;
	// starttime = time(NULL);
	// for(int i = 0; i < len; i++)
	// {
	// 	seamcarve_vf(vseams,image,forimage);
	// }
	// printf("Time for %d runs OpenCL. %lu\n", len, time(NULL) - starttime);
	// starttime = time(NULL);
	// for(int i = 0; i < len; i++)
	// {
	// 	seamcarve_vf_cpu(vseams,image,forimage);
	// }
	// printf("Time for %d runs CPU only. %lu\n", len, time(NULL) - starttime);

	// Mat hseamImage, hseamImagecpu;
	// printf("Starting seamcarve_hf\n");
	// seamcarve_hf(hseams,image,hseamImage);
	// printf("hseam row %d col %d\n\n", hseamImage.rows, hseamImage.cols);
	// imshow("Horizontal seamcarved image",hseamImage);
	// printf("Starting seamcarve_hf_cpu\n");
	// seamcarve_hf_cpu(hseams,image,hseamImagecpu);
	// printf("hseam row %d col %d\n\n", hseamImagecpu.rows, hseamImagecpu.cols);
	// imshow("Horizontal seamcarved image cpu",hseamImagecpu);

	//crop
	// default_random_engine gen(time(NULL));
	// int inputSize = 120;
	// uniform_int_distribution<int> disI(0,image.rows - inputSize);
	// uniform_int_distribution<int> disJ(0,image.cols - inputSize);
	// int si = disI(gen);
	// int sj = disJ(gen);
	// Mat crop(image,Range(si,si+inputSize),Range(sj,sj+inputSize));
	// imshow("Cropped image",crop);
	// printf("crop row %d col %d\n\n", crop.rows, crop.cols);

	// //distort down
	// Mat distort;
	// resize(image,distort,cvSize);
	// imshow("Distort down",distort);
	// printf("distort down row %d col %d\n\n", distort.rows, distort.cols);

	
	////scale down
	// Mat temp, ttemp;
	// if(vseams > 0) //width > height. landscape
	// {
	// 	//vertical seams, fast
	// 	seamcarve_vf(image.cols - image.rows,image,ttemp);//bring us to square
	// 	resize(ttemp, temp,cvSize);
	// }
	// // imshow("Scaled down almost",ttemp);
	// imshow("Scaled down image",temp);
	// printf("scale down row %d col %d\n\n", temp.rows, temp.cols);

	// Mat vthseamImage;
	// printf("Starting seamcarve_both_vth\n");
	// seamcarve_both_vth(vseams,hseams,image,vthseamImage);
	// printf("vthseam row %d col %d\n\n", vthseamImage.rows, vthseamImage.cols);
	// imshow("VTH",vthseamImage);

	// Mat htvseamImage;
	// printf("Starting seamcarve_both_htv\n");
	// seamcarve_both_htv(hseams,vseams,image,htvseamImage);
	// printf("hthseam row %d col %d\n\n", htvseamImage.rows, htvseamImage.cols);
	// imshow("HTV",htvseamImage);

	// // if(countNonZero(htvseamImage != vthseamImage) == 0)
	// // {
	// // 	printf("htv and vth are equal\n");
	// // }
	// // else
	// // {
	// // 	printf("htv and vth are not equal\n");
	// // }

	// // int len = 2;
	// // vector<Mat> tests(len);
	//  Mat rawseamImage;

	// printf("Starting seamcarve_both_raw\n");
	// // printf("Starting carving\n");
	// // int starttime = time(NULL);
	// // for(int i = 0; i < len; i++)
	// 	seamcarve_both_raw(vseams,hseams,image,rawseamImage);//tests[i]);
	// // printf("Time for %d carves: %lu sec\n", len, time(NULL)-starttime);
	// printf("rawseam row %d col %d\n\n", rawseamImage.rows, rawseamImage.cols);
	// imshow("RawSeam",rawseamImage);

	// Mat scaledseamImage;
	// printf("Starting seamcarve_both_scaled\n");
	// seamcarve_both_scaled(vseams,hseams,image,scaledseamImage);
	// printf("scaledseam row %d col %d\n", scaledseamImage.rows, scaledseamImage.cols);
	// imshow("scaledSeam",scaledseamImage);

	

	// waitKey(0);
	// seamcarve_cleanup();

	//getchar();
}