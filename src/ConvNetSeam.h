

#ifndef ____ConvNetSeam__
#define ____ConvNetSeam__

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

bool seamcarve_vf(int numSeams, const cv::Mat& source, cv::Mat& dest);

#endif /* defined(____ConvNetSeam__)*/