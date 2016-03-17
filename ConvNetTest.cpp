/*********************************************
 *
 *	ConvNetTest
 *
 * 	Created by Connor Bowley on 3/16/15
 *
 *********************************************/

#include <iostream>
#include <vector>
#include <iomanip>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ConvNet.h"

 using namespace std;

int main(int argc, char** argv)
{
	cout << "Making Net" << endl;
	Net *testNet = new Net(32,32,3);
	testNet->debug();
	cout << "Adding MaxPoolLayer" << endl;
	testNet->addMaxPoolLayer(2,2);


	cout << "Done" << endl;
	//delete testNet;
	return 0;
}