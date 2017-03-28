//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

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

		string imname = argv[i];
		imname = imname.substr(imname.rfind('/')+1);
		int startprediction = imname.find("_prediction");
		int lastDot = imname.rfind('.');
		imname.erase(startprediction,lastDot - startprediction);
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

		Mat redOnly;
		addWeighted(low, 1., high, 1., 0., redOnly);
		bitwise_not(redOnly,redOnly);

		Mat greenOnly;
		inRange(hsv_im, Scalar(45,50,50),Scalar(75,255,255),greenOnly);
		bitwise_not(greenOnly,greenOnly);

		imshow("redOnly",redOnly);
		waitKey(0);

		imshow("greenOnly",greenOnly);
		waitKey(0);

		SimpleBlobDetector::Params params;
		params.filterByColor = true;
		params.blobColor = 0;
		params.filterByInertia = false;
		// params.minInertiaRatio = 0;
		// params.maxInertiaRatio = 1;
		params.filterByArea = false;
		params.minArea = 10;
		params.maxArea = 300;

		params.filterByCircularity = false;
		params.filterByConvexity = false;
		
		SimpleBlobDetector detector(params);
		vector<KeyPoint> redkeypoints, greenkeypoints;
		detector.detect(redOnly, redkeypoints);
		detector.detect(greenOnly, greenkeypoints);

		//image name, whiteCount, blueCount
		printf("%s,%lu,%lu\n", imname.c_str(), redkeypoints.size(),greenkeypoints.size());

		Mat im_with_keypoints;
		drawKeypoints(im, redkeypoints, im_with_keypoints, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("redkeypoints",im_with_keypoints);
		waitKey(0);

		// Mat im_with_keypoints2;
		// drawKeypoints(im, greenkeypoints, im_with_keypoints2, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		// imshow("greenkeypoints",im_with_keypoints2);
		// waitKey(0);
	}
}