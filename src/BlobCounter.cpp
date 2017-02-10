//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if(argc == 1)
	{
		printf("Use: ./BlobCounter im1.jpg im2.png ...\n");
		return 0;
	}
	// printf("for\n");
	for(int i = 1; i < argc; i++)
	{
		Mat im = imread(argv[i], 1);
		// imshow("orig",im);
		// waitKey(0);

		// printf("hi\n");
		// printf("argv[%d] %s\n", i,argv[i]);

		if((im.empty()))
		{
			printf("%s_EMPTY: count 0\n", argv[i]);
			continue;
		}


		medianBlur(im,im,3);
		Mat hsv_im;
		cvtColor(im,hsv_im,COLOR_BGR2HSV);

		Mat low, high;
		inRange(hsv_im, Scalar(0,50,50),Scalar(30,255,255),low);
		inRange(hsv_im, Scalar(130,50,50),Scalar(179,255,255),high);

		Mat comb;
		addWeighted(low, 1., high, 1., 0., comb);

		// imshow("converted",comb);
		// waitKey(0);

		SimpleBlobDetector::Params params;
		params.filterByColor = 1;
		params.blobColor = 255;
		// params.filterByInertia = 1;
		// params.minInertiaRatio = .5;
		// params.maxInertiaRatio = 1;
		params.filterByArea = 1;
		params.minArea = 10;
		params.maxArea = 324;
		
		SimpleBlobDetector detector(params);
		vector<KeyPoint> keypoints;
		detector.detect(comb, keypoints);
		printf("%s: count %lu\n", argv[i], keypoints.size());

		// Mat im_with_keypoints;
		// drawKeypoints(im, keypoints, im_with_keypoints, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		// imshow("keypoints",im_with_keypoints);
		// waitKey(0);
	}
}