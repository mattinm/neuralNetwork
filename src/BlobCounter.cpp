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

	for(int i = 1; i < argc; i++)
	{
		SimpleBlobDetector::Params params;
		params.filterByColor = 1;
		params.blobColor = 255;



		Mat im = imread(argv[i], IMREAD_GRAYSCALE);
		SimpleBlobDetector detector(params);
		vector<KeyPoint> keypoints;
		detector.detect(im, keypoints);
		printf("%s: count %lu\n", argv[i], keypoints.size());

		// Mat im_with_keypoints;
		// drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		// imshow("keypoints",im_with_keypoints);
		// waitKey(0);
	}
}