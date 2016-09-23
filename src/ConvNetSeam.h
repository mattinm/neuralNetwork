

#ifndef ____ConvNetSeam__
#define ____ConvNetSeam__

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

bool seamcarve_vf(int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_hf(int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_vth(int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_htv(int hseams, int vseams, const cv::Mat& source, cv::Mat& dest);
void seamcarve_cleanup();

#endif /* defined(____ConvNetSeam__)*/