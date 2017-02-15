/*

Seamcarver.h
Created by Connor Bowley on 2/15/17

Makes a Seamcarver object that can be used to seamcarve an OpenCV Mat.
Only adjusts memory used if more memory is needed, specifically asked to, or in the destructor.
*/

#ifndef ___Seamcarver__
#define ___Seamcarver__

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Seamcarver{
private:
	int* image;
	char *vdirs, *hdirs;
	float *greyscale, *vals, *vcosts, *hcosts;
	int *vseam, *hseam;

	int owidth = -1, oheight = -1;
	int window_width=0, window_height=0, window_size=0;

public:
	Seamcarver();
	~Seamcarver();
	bool carve_v(const cv::Mat& source, int numSeams, cv::Mat& dest);
	bool carve_h(const cv::Mat& source, int numSeams, cv::Mat& dest);
	bool carve_b(const cv::Mat& source, int vseams, int hseams, cv::Mat& dest);
	void destroyMem();

private:
	int POSITION(int x, int y);
	int POSITION(int x, int y, int z);
	void makeMem();
	void calcGreyscale(const cv::Mat& source);
	void calcGradient();
	void init(const cv::Mat& source);
};

#endif /* defined(___Seamcarver__)*/