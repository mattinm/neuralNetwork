//
//  NeuralNetworkFunctions.cpp
//  
//
//  Created by Connor Bowley on 3/4/16.
//
//

#include "NeuralNetworkFunctions.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <limits>
#include <vector>
#include <iomanip>

#define GETMAX(x,y) (x > y) ? x: y

using namespace cv;
using namespace std;

/******************************
 *
 * Classes
 *
 ******************************/

/*********************
 * Array3D is a class that functions as 3 dimensional array whose size can be determined at runtime but is 
 * unchangeable after initialization.
 *********************/
class Array3D
{
	private:
		int* array;
	public:
		Array3D();
		Array3D(int,int,int);
		Array3D(int,int,int,int);
		Array3D(int,int,int,int*);
		~Array3D(void);
		int rows, cols, depth;
		int width, height;
		int get(int,int,int);
		void set(int,int,int,int);
		void setArray(int,int,int,int*);
		void print(void);
};

Array3D::Array3D()
{
	array = new int[0];
	rows = 0;
	width = 0;
	cols = 0;
	height = 0;
	depth = 0;
}

Array3D::Array3D(int numRows, int numCols, int dep)
{
	array = new int[numRows*numCols*dep];
	rows = numRows;
	width = numRows;
	cols = numCols;
	height = numCols;
	depth = dep;
}

Array3D::Array3D(int numRows, int numCols, int dep, int init)
{
	array = new int[numRows*numCols*dep];
	rows = numRows;
	width = numRows;
	cols = numCols;
	height = numCols;
	depth = dep;
	int sizes = rows*cols*depth;
	for(int i=0;i<sizes;i++)
	{
		array[i] = init;
	}
}

Array3D::Array3D(int numRows, int numCols, int dep, int *init)
{
	array = new int[numRows*numCols*dep];
	rows = numRows;
	width = numRows;
	cols = numCols;
	height = numCols;
	depth = dep;
	int sizes = rows*cols*depth;
	for(int i=0;i<sizes;i++)
	{
		array[i] = init[i];
	}
}


Array3D::~Array3D(void)
{
	delete array;
}

int Array3D::get(int row, int col, int dep)
{
	return array[row*cols*depth+col*depth+dep];
}

void Array3D::set(int val, int row, int col, int dep)
{
	array[row*cols*depth+col*depth+dep] = val;
}

void Array3D::setArray(int numRows, int numCols, int dep, int* newArray)
{
	delete array;
	array = new int[numRows*numCols*dep];
	rows = numRows;
	width = numRows;
	cols = numCols;
	height = numCols;
	depth = dep;
	int sizes = rows*cols*depth;
	for(int i=0;i<sizes;i++)
	{
		array[i] = newArray[i];
	}
}

void Array3D::print(void)
{
	for(int i=0;i< rows;i++)
	{
		cout << "|";
		for(int j=0;j<cols;j++)
		{
			for(int k=0; k< depth; k++)
			{
				//printf("%4d",get(i,j,k));
				cout << setw(4) << get(i,j,k);
				if(k != depth-1) cout << ",";
			}
			cout << "|";
		}
		cout << endl;
	}
}
/*********************
 * end Array3D
 *********************/

Array3D* convolve(Array3D *source, Array3D *weights, int numFiltersAndBiases, int *bias, int stride)
{
	float fspatialSize = (source->rows - weights[0].rows)/(float)stride + 1;
	if(std::fmod(fspatialSize, 1.0) > .0001)
	{
		cout << "The stride appears to be the wrong size" << endl;
		return new Array3D(0,0,0);
	}
	int numFilters = numFiltersAndBiases;
	int subsetWidth = weights[0].rows;
	int subsetHeight = weights[0].height;
	// formula from http://cs231n.github.io/convolutional-networks/
	int depth2 = numFilters; // num of filters => depth 2
	int width2 =  (source->rows - weights[0].rows)/stride + 1;
	int height2 = (source->cols - weights[0].cols)/stride + 1;
	printf("(%d  - %d) / %d + 1\n",source->rows,weights[0].rows,stride);
	int oX, oY;
	Array3D *out = new Array3D(width2,height2,depth2);
	int sum;
	for(int f=0; f<numFilters; f++) // which filter we're on
	{
		oX = 0;
		
		for(int i=0; i < source->rows; i+= stride) // row in source
		{
			oY = 0;
			for(int j=0; j < source->cols;j+=stride) // col in source
			{
				//now we go into the stride subset
				sum = 0;
				for(int k=0; k < source->depth; k++) // depth in source
				{
					for(int s=0; s < subsetWidth; s++) // row in stride subset of source
					{
						for(int r=0; r < subsetHeight; r++) // col in stride subset of source
						{
							sum += source->get(i+s,j+r,k) * weights[f].get(s,r,k);
						}
					}
				}
				// add bias
				sum += bias[f];
				// add into out[i%stride, j%stride, f]
				out->set(sum, oX, oY, f);
				oY++;
			}
			oX++;
		}
	}
	return out;
}

int getMax(double x, double y)
{
	if(x > y) return x;
	return y;
}

Array3D* maxPool(Array3D *source, int squarePoolSize)
{
	if(source->rows % squarePoolSize != 0 || source->cols % squarePoolSize !=0)
	{
		printf("The number of rows and cols need to be evenly divisible by the pool size\n");
		return new Array3D(0,0,0);
	}
	if(squarePoolSize == 1)
		return source;
	Array3D *out = new Array3D(source->rows/squarePoolSize,source->cols/squarePoolSize,source->depth);
	//printf("%d%d\n",out->rows,out->cols);
	int oX=0, oY=0;
	for(int k=0; k<source->depth; k++)
	{
		oX = 0;
		for(int i=0;i<source->rows;i+=squarePoolSize)
		{
			oY = 0;
			for(int j=0;j<source->cols;j+=squarePoolSize)
			{
				int maxVal = INT_MIN;
				for(int s=0;s<squarePoolSize;s++)
				{
					for(int r=0; r< squarePoolSize; r++)
					{
						maxVal = getMax(source->get(i+s,j+r,k),maxVal);
					}
				}
				out->set(maxVal,oX,oY++,k);
			}
			oX++;
		}
	}
	
	return out;
}

/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
Array3D* padZeros(Array3D *source, int numZeros)
{
	if(numZeros <= 0)
	{
		return source;
	}
	
	Array3D *output = new Array3D(source->rows + 2*numZeros, source->cols + 2*numZeros, source->depth);
	
	for(int i=0; i< output->rows; i++)
	{
		for(int j=0; j< output->cols; j++)
		{
			for(int k=0; k< output->depth; k++)
			{
				if(i < numZeros || i >= output->rows - numZeros || j < numZeros || j >= output->cols - numZeros)
				{
					output->set(0, i,j,k);
				}
				else
				{
					//printf("%d %d %d\n",i,j,k);
					output->set(source->get(i-numZeros,j-numZeros,k),i,j,k);
				}
			}
		}
	}
	
	return output;
}

Array3D* convertArrayToArray3D(int array[5][5][3], int width, int height, int depth)
{
	Array3D *out = new Array3D(width, height, depth);
	for(int i=0; i<width;i++)
	{
		for(int j=0;j<height;j++)
		{
			for(int k=0;k<depth;k++)
			{
				out->set(array[i][j][k],i,j,k);
			}
		}
	}
	return out;
}
Array3D* convertColorMatToArray3D(Mat m)
{
	if(m.type() != CV_8UC3)
	{
		return new Array3D(1,1,1);
	}
	Array3D *out = new Array3D(m.rows,m.cols,3);
	
	for(int i=0; i< m.rows; i++)
	{
		for(int j=0; j< m.cols; j++)
		{
			Vec3b curPixel = m.at<Vec3b>(i,j);
			out->set(curPixel[0],i,j,0);
			out->set(curPixel[1],i,j,1);
			out->set(curPixel[2],i,j,2);
		}
	}
	return out;
}



/******************************
 *
 * Functions
 *
 ******************************/


void printVector(const vector<vector<vector<double> > > &dest)
{
	for(int i=0;i < dest.size();i++) // rows
	{
		cout << "|";
		for(int j=0;j < dest[0].size();j++)
		{
			for(int k=0; k < dest[0][0].size(); k++)
			{
				//printf("%4d",get(i,j,k));
				cout << setw(4) << dest[i][j][k];
				if(k != dest[0][0].size()-1) cout << ",";
			}
			cout << "|";
		}
		cout << endl;
	}
}

void convert1DArrayTo3DVector(const double *array, int width, int height, int depth, vector<vector<vector<double> > > &dest)
{
	//resize dest vector
	dest.resize(width);
	for(int i=0; i < width; i++)
	{
		dest[i].resize(height);
		for(int j=0; j < height; j++)
		{
			dest[i][j].resize(depth);
		}
	}
	
	for(int i=0; i < width;i++)
	{
		for(int j=0; j < height; j++)
		{
			for(int k=0; k < depth; k++)
			{
				dest[i][j][k] = *array++;
			}
		}
	}
}


void convertColorMatToVector(Mat m, vector<vector<vector<double> > > &dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}
	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	
	for(int i=0; i< m.rows; i++)
	{
		for(int j=0; j< m.cols; j++)
		{
			Vec3b curPixel = m.at<Vec3b>(i,j);
			dest[i][j][0] = curPixel[0];
			dest[i][j][1] = curPixel[1];
			dest[i][j][2] = curPixel[2];
		}
	}
}


/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
void padZeros(const vector<vector<vector<double> > > &source, int numZeros, vector<vector<vector<double> > > &dest)
{
	int width2 = source.size() + 2*numZeros;
	int height2 = source.size() + 2*numZeros;
	int depth2 = source[0][0].size();
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	
	for(int i=0; i< dest.size(); i++) // rows
	{
		for(int j=0; j< dest[0].size(); j++) // cols
		{
			for(int k=0; k< dest[0][0].size(); k++) // depth
			{
				if(i < numZeros || i >= dest.size() - numZeros || j < numZeros || j >= dest[0].size() - numZeros)
				{
					dest[i][j][k] = 0;
				}
				else
				{
					//printf("%d %d %d\n",i,j,k);
					dest[i][j][k] = source[i-numZeros][j-numZeros][k];
				}
			}
		}
	}
}


void maxPool(const vector<vector<vector<double> > > &source, int squarePoolSize, vector<vector<vector<double> > > &dest)
{
	if(source.size() % squarePoolSize != 0 || source[0].size() % squarePoolSize != 0)
	{
		printf("The number of rows and cols need to be evenly divisible by the pool size\n");
		throw "Invalid pool size";
	}
	int width2 = source.size()/squarePoolSize;
	int height2 = source[0].size()/squarePoolSize;
	int depth2 = source[0][0].size();
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	//printf("%d%d\n",out->rows,out->cols);
	int oX=0, oY=0;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size();i+=squarePoolSize) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size();j+=squarePoolSize) // cols
			{
				double maxVal = numeric_limits<double>::min();
				for(int s=0;s<squarePoolSize;s++)
				{
					for(int r=0; r< squarePoolSize; r++)
					{
						maxVal = GETMAX(source[i+s][j+r][k],maxVal);
					}
				}
				dest[oX][oY++][k] = maxVal;
			}
			oX++;
		}
	}
}


void convolve(const vector<vector<vector<double> > > &source, const vector<vector<vector<vector<double> > > > &weights, int numFiltersAndBiases, double *bias, int stride, vector<vector<vector<double> > > &dest)
{
	float fspatialSize = (source.size() - weights[0].size())/(float)stride + 1;
	if(std::fmod(fspatialSize, 1.0) > .0001)
	{
		cout << "The stride appears to be the wrong size" << endl;
		throw "Wrong stride size";
	}
	int numFilters = numFiltersAndBiases;
	int subsetWidth = weights[0].size();
	int subsetHeight = weights[0][0].size();
	// formula from http://cs231n.github.io/convolutional-networks/
	int depth2 = numFilters; // num of filters => depth 2
	int width2 =  (source.size() - weights[0].size())/stride + 1;
	int height2 = (source[0].size() - weights[0][0].size())/stride + 1;
	printf("(%lu  - %lu) / %d + 1\n",source.size(),weights[0].size(),stride);
	int oX, oY;
	//resize dest vector
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	double sum;
	for(int f=0; f<numFilters; f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i < source.size()-1; i+= stride) // row in source
		{
			oY = 0;
			for(int j=0; j < source[i].size()-1;j+=stride) // col in source
			{
				//now we go into the stride subset
				sum = 0;
				for(int k=0; k < source[i][j].size(); k++) // depth in source
				{
					for(int s=0; s < subsetWidth; s++) // row in stride subset of source
					{
						for(int r=0; r < subsetHeight; r++) // col in stride subset of source
						{
							cout << i << "+" << s << ", " << j << "+" << r << ", " << k << endl;
							//sum += source[i+s][j+r][k] * weights[f][s][r][k];
							sum += source.at(i+s).at(j+r).at(k) * weights.at(f).at(s).at(r).at(k);
						}
					}
				}
				// add bias
				sum += bias[f];
				// add into out[i%stride, j%stride, f]
				cout << sum << endl;
				dest[oX][oY][f] = sum;
				oY++;
			}
			oX++;
		}
	}
}

int main(int argc, char** argv)
{
	vector<vector<vector<double> > > testVector;//(1,vector <vector <double> >(1, vector<double>(1, 0.0)));
	Mat M(4,4,CV_8UC3, Scalar(100, 150, 200));
	M += 5*Mat::eye(M.rows,M.cols, M.type());
	cout << "M = " << endl << M << "\n\n";
	
	
	double testArray[] = {
		0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
		2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
		1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
		1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
		2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};
	
	convert1DArrayTo3DVector(testArray,5,5,3,testVector);
	
	vector<vector<vector<double> > > padVector;
	padZeros(testVector,1,padVector);
	
	vector<vector<vector<double> > > convVector;
	
	vector<vector<vector<vector<double> > > > filters(2);
	
	double filter0[] = {
		 1, 0, 1,	-1, 0, 0,	-1, 1, 0,
		 0, 1, 0,	 1, 0, 0,	 1,-1,-1,
		-1, 0, 0,	 1, 1,-1,	-1, 1, 0
	};
	double filter1[] = {
		 0, 1,-1,	-1,-1,-1,	-1, 0,-1,
		 1, 1,-1,	 0, 0,-1,	 0, 1,-1,
		 1,-1, 1,	 1, 1,-1,	-1, 0, 1
	};
	double bias[] = {1,0};

	
	convert1DArrayTo3DVector(filter0,3,3,3,filters[0]);
	convert1DArrayTo3DVector(filter1,3,3,3,filters[1]);
	
	convolve(padVector,filters,2,bias,2,convVector);
	printVector(convVector);
	
	
	
	return 0;
}

/*solution to conv
 | 1, 1| 2, 3| 6, 4|
 | 3,-4| 3,-4| 4,-1|
 | 4,-5| 4,-3| 5,-1|
*/













