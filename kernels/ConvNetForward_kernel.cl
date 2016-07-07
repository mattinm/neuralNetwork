#define RELU_CAP 5000.0
#define LEAKY_RELU_CONST .01

// DO NOT CHANGE THESE DEFAULTS VALUES. IF YOU MUST, MAKE THEM THE SAME LENGTH IN CHARACTERS

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void relu(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	// neurons[i] = clamp(prevNeurons[i], -RELU_CAP, RELU_CAP);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= RELU_CAP)
		neurons[i] = prevNeurons[i];
	else if(prevNeurons < 0)
		neurons[i] = 0;
	else
		neurons[i] = RELU_CAP;
}

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyRelu(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	//double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	double newVal;
	if(prevNeurons[i] >= 0) 
		newVal = prevNeurons[i];
	else 
		newVal = prevNeurons[i] * LEAKY_RELU_CONST;

	// neurons[i] = clamp(newVal, -RELU_CAP, RELU_CAP);

	if(-RELU_CAP <= newVal && newVal <= RELU_CAP)
		neurons[i] = newVal;
	else if(newVal < -RELU_CAP)
		neurons[i] = -RELU_CAP;
	else
		neurons[i] = RELU_CAP;
}


//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride)
{
	// int width = prevwidth;
	// int depth = prevdepth;
	
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	int i = x;
	int strxdep = stride * prevdepth;
	int i_div_dep = i / prevdepth;
	int numBlocksPerRow = (prevwidth - poolsize)/stride + 1; 
	//int ourHeight = i/numBlocksPerRow/depth;
	//int ourRowStartIndex = ourHeight * width * stride * depth + i%depth;
	//int ourRowShift = ((i/depth)%numBlocksPerRow) * stride * depth;
	//int ourStartIndex = ourRowStartIndex + ourRowShift;
	//i = ourRowStartIndex + ourRowShift;

	i = (i_div_dep/numBlocksPerRow * prevwidth * strxdep + i%prevdepth) + (((i_div_dep)%numBlocksPerRow) * strxdep);
	
	int amountToNextLayer = (prevwidth - poolsize) * prevdepth;
	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] > maxVal)
				maxVal = prevNeurons[i];
			i += prevdepth;
		}
		i += amountToNextLayer;
	}
	neurons[x] = maxVal;
}

__kernel void convolve(__global double* prevNeurons, __global double* neurons,
	__global double* weights, __global double* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
{
	//int myHeight = myBlock/numBlocksPerRow;
	//int myRowStartIndex = (myBlock/numBlocksPerRow) * width * strxdep;
	//int myRowShift = (myBlock%numBlocksPerRow) * strxdep;

	//int width = prevwidth;
	//int depth = prevdepth;

	int i = get_global_id(0);
	int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	
	//int myFilter = i/(numBlocksPerRow * numBlocksPerRow);
	int myFilter = i%numFilters;
	int filterLayerSize = filterSize * prevdepth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int strxdep = stride * prevdepth;
	//int myStartIndex = ((myBlock/numBlocksPerRow) * width * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);
	//int h = myStartIndex;
	int h = ((myBlock/numBlocksPerRow) * prevwidth * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);

	int amountToNextLayer = (prevwidth - filterSize) * prevdepth;

	//can I do the pointer arithmetic better?

	double result = 0;
	__global double* curWeight = &(weights[j]);
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			//result += weights[j++] * prevNeurons[h++];
			result += *(curWeight++) * prevNeurons[h++];
			// result = mad(*(curWeight++),prevNeurons[h++],result);
		}
		h += amountToNextLayer;
	}
	//printf("numFil: %d id: %d myBlock: %d\n",numFilters,get_global_id(0), myBlock);
	//printf("In convolve. Global id = %d\n\tmyFilter = %d\n\tresult = %f\n",i,myFilter,result);
	neurons[i] = result + biases[myFilter];
}

__kernel void convolveConstant(__global double* prevNeurons, __global double* neurons,
	__constant double* weights, __constant double* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
// __kernel void convolveConstant(__global double* prevNeurons, __global double* neurons,
// 	__constant double* weights, __constant double* biases, int numFilters, int filterSize, int strxdep, 
// 	int prevwidth, int amountToNextLayer, int filterLayerSize, int numBlocksPerRow)
{
	//int myHeight = myBlock/numBlocksPerRow;
	//int myRowStartIndex = (myBlock/numBlocksPerRow) * width * strxdep;
	//int myRowShift = (myBlock%numBlocksPerRow) * strxdep;

	//int width = prevwidth;
	//int depth = prevdepth;

	int i = get_global_id(0);
	int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	
	int myFilter = i%numFilters;
	int filterLayerSize = filterSize * prevdepth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int strxdep = stride * prevdepth;
	//int myStartIndex = ((myBlock/numBlocksPerRow) * width * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);
	//int h = myStartIndex;
	int h = ((myBlock/numBlocksPerRow) * prevwidth * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);

	int amountToNextLayer = (prevwidth - filterSize) * prevdepth;

	//can I do the pointer arithmetic better?

	double result = 0;
	__constant double* curWeight = &(weights[j]);
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			//result += weights[j++] * prevNeurons[h++];
			result += *(curWeight++) * prevNeurons[h++];
			// result = mad(*(curWeight++),prevNeurons[h++],result);
		}
		h += amountToNextLayer;
	}
	//printf("numFil: %d id: %d myBlock: %d\n",numFilters,get_global_id(0), myBlock);
	//printf("In convolve. Global id = %d\n\tmyFilter = %d\n\tresult = %f\n",i,myFilter,result);
	neurons[i] = result + biases[myFilter];
}

__kernel void softmax(__global double *prevNeurons, __global double *neurons,
	double denominator)
{
	int i = get_global_id(0);
	neurons[i] = exp(prevNeurons[i])/denominator;
}

__kernel void softmax_allCL(__global double *prevNeurons, __global double *neurons,
	__global double* denominator)
{
	int i = get_global_id(0);
	neurons[i] = exp(prevNeurons[i])/(*denominator);
}

__kernel void zeroPad(__global double *prevNeurons, __global double *neurons, int pad, int prevwidth,
	int prevheight, int depth)
{
	int x = get_global_id(0);

	//turn x into i, j, k
	const int nw = prevwidth + 2*pad;
	const int nh = prevheight + 2*pad;

	int ourDepth = x%depth;
	int ourCol = ((x-ourDepth)/depth) % nw;
	int ourRow = ((x-ourDepth)/depth) / nw;

	if(ourRow < pad || ourRow >= nh-pad || ourCol < pad || ourCol >= nw-pad)
		neurons[x] = 0;
	else
	{
		int i = ourRow - pad;
		int j = ourCol - pad;
		int k = ourDepth;
		int oldIndex = (i * prevwidth * depth) + (j * depth) + k;

		neurons[x] = prevNeurons[oldIndex];
	}
}

/*************************************************
*
*	Helper kernels for softmax_allCL
*
*************************************************/

//make this 2 kernels? one to get max, one to subtract? Prob not worth it without a lot of classes
__kernel void maxSubtraction(__global double* source, int size)
{
	if(size <= 0)
		return;
	double max = source[0];
	double cur;
	for(int i = 1; i < size; i ++)
	{
		cur = source[i];
		if(cur > max)
			max = cur;
	}
	for(int i=0; i < size; i++)
		source[i] -= max;
}

__kernel void vectorESum(__global double* source, int size, __global double* denom)
{
	if(size <= 0)
		return;
	double sum = 0;
	for(int i=0; i < size; i++)
		sum += exp(source[i]);
	*denom = sum;
}



















