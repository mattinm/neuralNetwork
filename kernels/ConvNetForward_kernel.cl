//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void relu(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= 5)
		neurons[i] = prevNeurons[i];
	else if(prevNeurons < 0)
		neurons[i] = 0;
	else
		neurons[i] = 5;
}

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyRelu(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	//double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	double newVal;
	if(prevNeurons[i] > 0) newVal = prevNeurons[i];
	else newVal = prevNeurons[i] * 0.01;
	if(newVal >= -5 && newVal <= 5)
		neurons[i] = newVal;
	else if(newVal < -5)
		neurons[i] = -5;
	else
		neurons[i] = 5;
}


//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride)
{
	int width = prevwidth;
	int depth = prevdepth;
	/*
	//getting the start index of a flattened 3d array for maxPool
	int i = get_global_id(0);
	int numBlocksPerRow = (width - poolsize)/stride + 1); 
	int ourHeight = i/numBlocksPerRow;
	int ourRowStartIndex = ourHeight * width * stride * depth;
	int ourRowShift = (i%numBlocksPerRow) * stride * depth;
	int ourStartIndex = ourRowStartIndex + ourRowShift;
	*/
	int x = get_global_id(0);
	int numBlocksPerRow = (width - poolsize)/stride + 1; // maybe calc once and pass in as param
	int i = ((x/numBlocksPerRow) * width * stride * depth) + ((x%numBlocksPerRow) * stride * depth); //see large comment above

	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] > maxVal)
				maxVal = prevNeurons[i];
			i += depth;
		}
		i += depth*width;
	}
	neurons[x] = maxVal;
}

__kernel void convolve(__global double* prevNeurons, __global double* neurons,
	__constant double* weights, __constant double* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
{
	//int myHeight = myBlock/numBlocksPerRow;
	//int myRowStartIndex = (myBlock/numBlocksPerRow) * width * strxdep;
	//int myRowShift = (myBlock%numBlocksPerRow) * strxdep;

	int width = prevwidth;
	int depth = prevdepth;

	int i = get_global_id(0);
	int numBlocksPerRow = (width - filterSize)/stride + 1;
	int myFilter = i/numFilters;
	int filterLayerSize = filterSize * depth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = i%numFilters;
	int strxdep = stride * depth;
	int myStartIndex = ((myBlock/numBlocksPerRow) * width * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);
	int h = myStartIndex;

	int amountToNextLayer = (width - filterSize) * depth;

	//can I do the pointer arithmetic better?

	double result = 0;
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			result += weights[j++] * prevNeurons[h++];
		}
		h += amountToNextLayer;
	}
	neurons[i] = result + biases[myFilter];
}

__kernel void softmax(__global double *prevNeurons, __global double *neurons,
	double denominator)
{
	int i = get_global_id(0);
	neurons[i] = exp(prevNeurons[i])/denominator;
}


















