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
	double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	if(newVal >= -5 && newVal <= 5)
		neurons[i] = newVal;
	else if(newVal < -5)
		neurons[i] = -5;
	else
		neurons[i] = 5;
}


//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global double* prevNeurons, __global double* neurons,
	int width, int depth, int poolsize, int stride)
{
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

__kernel void convolve()
{

}

__kernel void softmax()
{

}