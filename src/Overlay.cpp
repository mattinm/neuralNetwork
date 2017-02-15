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
		printf("Use: ./Overlay originalImage predictionImageInColor outLocation\n");
		return 0;
	}
	//argv[1] original image
	//argv[2] prediction image
	//argv[3] output location
	Mat orig, pred, dest;
	string origname(argv[1]);
	orig = imread(argv[1],1); //read color
	pred = imread(argv[2],1);
	string outloc = string(argv[3]);
	if(outloc.rfind('/') != outloc.length() - 1)
		outloc += "/";

	cvtColor(orig,orig, COLOR_BGR2BGRA);
	cvtColor(pred,pred, COLOR_BGR2BGRA);

	double alpha = 0.5;
	double beta = 1 - alpha;

	// imshow("orig",orig);
	// waitKey(0);

	// imshow("pred",pred);
	// waitKey(0);

	// printf("channels %d\n", pred.channels());
	for(int i = 0; i < pred.rows; i++)
		for(int j = 0; j < pred.cols; j++)
		{
			Vec4b& bgra = pred.at<Vec4b>(i,j);
			if(bgra[2] < 100 && bgra[1] < 100) //red < x && green < y
			{
				const Vec4b& o = orig.at<Vec4b>(i,j);
				bgra[0] = o[0];
				bgra[1] = o[1];
				bgra[2] = o[2];
				bgra[3] = o[3];
				// // bgra[0] = 0;
				// bgra[3] = 0; // transparent alpha
				// printf("transparenting\n");
			}
		}

	// imshow("new pred", pred);
	// waitKey(0);

	addWeighted(orig,alpha,pred,beta,0,dest);

	// imshow("overlayed",dest);
	// waitKey(0);

	string outName = outloc;
	outName += "overlay_";
	outName += origname.substr(origname.rfind('/')+1);

	// printf("out name: %s\n", outName.c_str());

	imwrite(outName.c_str(),dest);
}