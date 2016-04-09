//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void relu(__global float* prevNeurons, __global float* neurons)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= 5000)
		neurons[i] = prevNeurons[i];
	else if(prevNeurons < 0)
		neurons[i] = 0;
	else
		neurons[i] = 5;
}

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyRelu(__global float* prevNeurons, __global float* neurons)
{
	const int i = get_global_id(0);
	//float newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	float newVal;
	if(prevNeurons[i] >= 0) 
		newVal = prevNeurons[i];
	else 
		newVal = prevNeurons[i] * 0.01;

	if(-5000 <= newVal && newVal <= 5000)
		neurons[i] = newVal;
	else if(newVal < -5000)
		neurons[i] = -5000;
	else
		neurons[i] = 5000;
}


//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global float* prevNeurons, __global float* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride)
{
	int width = prevwidth;
	int depth = prevdepth;



	
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	int i = x;
	int strxdep = stride * depth;
	int i_div_dep = i / depth;
	int numBlocksPerRow = (width - poolsize)/stride + 1; 
	//int ourHeight = i/numBlocksPerRow/depth;
	//int ourRowStartIndex = ourHeight * width * stride * depth + i%depth;
	//int ourRowShift = ((i/depth)%numBlocksPerRow) * stride * depth;
	//int ourStartIndex = ourRowStartIndex + ourRowShift;
	//i = ourRowStartIndex + ourRowShift;

	i = (i_div_dep/numBlocksPerRow * width * strxdep + i%depth) + (((i_div_dep)%numBlocksPerRow) * strxdep);
	
	int amountToNextLayer = (width - poolsize) * depth;
	float maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] > maxVal)
				maxVal = prevNeurons[i];
			//if(x == 0)
			//	printf("%f %d\n", prevNeurons[i],i);
			i += depth;
		}
		i += amountToNextLayer;
	}
	neurons[x] = maxVal;
}

__kernel void convolve(__global float* prevNeurons, __global float* neurons,
	__constant float* weights, __constant float* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
{
	//int myHeight = myBlock/numBlocksPerRow;
	//int myRowStartIndex = (myBlock/numBlocksPerRow) * width * strxdep;
	//int myRowShift = (myBlock%numBlocksPerRow) * strxdep;

	int width = prevwidth;
	int depth = prevdepth;

	int i = get_global_id(0);
	int numBlocksPerRow = (width - filterSize)/stride + 1;
	
	//int myFilter = i/(numBlocksPerRow * numBlocksPerRow);
	int myFilter = i%numFilters;
	int filterLayerSize = filterSize * depth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int strxdep = stride * depth;
	int myStartIndex = ((myBlock/numBlocksPerRow) * width * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);
	int h = myStartIndex;

	int amountToNextLayer = (width - filterSize) * depth;

	//can I do the pointer arithmetic better?

	float result = 0;
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			result += weights[j++] * prevNeurons[h++];
		}
		h += amountToNextLayer;
	}
	//printf("numFil: %d id: %d myBlock: %d\n",numFilters,get_global_id(0), myBlock);
	//printf("In convolve. Global id = %d\n\tmyFilter = %d\n\tresult = %f\n",i,myFilter,result);
	neurons[i] = result + biases[myFilter];
}

__kernel void softmax(__global float *prevNeurons, __global float *neurons,
	float denominator)
{
	int i = get_global_id(0);
	neurons[i] = exp(prevNeurons[i])/denominator;
}


__kernel void zeroPad(__global float *prevNeurons, __global float *neurons, int pad, int prevwidth,
	int prevheight, int depth)
{
	int x = get_global_id(0);

	//turn x into i, j, k
	int nw = prevwidth + 2*pad;
	int nh = prevheight + 2*pad;

	int ourRow = x/(nw * depth);
	int ourCol = (x/depth) % nw;
	int ourDepth = x%depth;

	if(ourRow == 0 || ourRow == nh-1 || ourCol == 0 || ourCol == nw-1)
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



















