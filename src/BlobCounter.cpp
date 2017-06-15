//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int getMSI(string filename)
{
	int startMSIIndex = filename.find("msi");
	int nextUnderscore = filename.find("_",startMSIIndex);
	if(nextUnderscore == string::npos)
		nextUnderscore = filename.find(".",startMSIIndex);
	// printf("%s\n", filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3).c_str());
	return stoi(filename.substr(startMSIIndex+3,nextUnderscore - startMSIIndex + 3));
}

int main(int argc, char** argv)
{

	bool show = false;
	if(argc == 1)
	{
		printf("Use: ./BlobCounter im1.jpg im2.png ...\n");
		printf("   Optional: must go before listing of images:\n");
		printf("   -show      Shows the images and which blobs were picked up.\n");
		return 0;
	}

	int i = 1;
	if(string(argv[1]).find("-show") != string::npos)
	{
		i++;
		show = true;
	}
	// printf("for\n");
	for(; i < argc; i++)
	{
		Mat im = imread(argv[i], 1);

		//making a border that is all blue will help the blob detector find blobs on the edge of an image
		int b = 5;
		copyMakeBorder(im,im,b,b,b,b,BORDER_CONSTANT,Scalar(255,0,0));

		string imname = argv[i];
		imname = imname.substr(imname.rfind('/')+1);
		int startprediction = imname.find("_prediction");
		int lastDot = imname.rfind('.');
		imname.erase(startprediction,lastDot - startprediction);

		cv::Size mysizeMatched(750,750 * im.rows / im.cols); //make the 750 smaller if it's too big for your screen
		
		if(show)
		{
			Mat im_show = im.clone();
			resize(im_show,im_show,mysizeMatched);
			imshow("orig",im_show);
			waitKey(0);
		}


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


		//copy-convert image to B/W where black is where the red was. This helps with the blob detector.
		Mat redOnly;
		addWeighted(low, 1., high, 1., 0., redOnly);
		bitwise_not(redOnly,redOnly);

		//copy-convert image to B/W where black is where the green was
		Mat greenOnly;
		inRange(hsv_im, Scalar(45,50,50),Scalar(75,255,255),greenOnly);
		bitwise_not(greenOnly,greenOnly);

		if(show)
		{
			Mat red_show = redOnly.clone();
			resize(red_show,red_show,mysizeMatched);
			imshow("redOnly",red_show);
			waitKey(0);

			Mat green_show = greenOnly.clone();
			resize(green_show,green_show,mysizeMatched);
			imshow("greenOnly",green_show);
			waitKey(0);
		}

		SimpleBlobDetector::Params params;
		params.filterByColor = true;
		params.blobColor = 0;
		params.filterByInertia = false;
		// params.minInertiaRatio = 0;
		// params.maxInertiaRatio = 1;
		params.filterByArea = false;
		params.minArea = 0;
		params.maxArea = 300;

		params.filterByCircularity = false;
		params.filterByConvexity = false;

		vector<KeyPoint> redkeypoints, greenkeypoints;
		#if CV_MAJOR_VERSION < 3
		SimpleBlobDetector detector(params);
		detector.detect(redOnly, redkeypoints);
		detector.detect(greenOnly, greenkeypoints);
		#else
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
		detector->detect(redOnly, redkeypoints);
		detector->detect(greenOnly, greenkeypoints);
		#endif

		
		

		//image name, whiteCount, blueCount
		printf("%d,%lu,%lu\n", getMSI(imname), redkeypoints.size(),greenkeypoints.size());

		if(show)
		{
			Mat im_with_keypoints;
			drawKeypoints(im, redkeypoints, im_with_keypoints, Scalar(0,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			resize(im_with_keypoints,im_with_keypoints,mysizeMatched);
			imshow("redkeypoints",im_with_keypoints);
			waitKey(0);

			Mat im_with_keypoints2;
			drawKeypoints(im, greenkeypoints, im_with_keypoints2, Scalar(0,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			resize(im_with_keypoints2,im_with_keypoints2,mysizeMatched);
			imshow("greenkeypoints",im_with_keypoints2);
			waitKey(0);
		}
	}
}