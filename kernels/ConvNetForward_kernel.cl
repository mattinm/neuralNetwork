__kernel void relu(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= 5)
		neurons[i] = prevNeurons;
	else if(prevNeurons < 0)
		neurons[i] = 0;
	else
		neurons[i] = 5;
}

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
/*
__kernel void maxPool(__global double* prevNeurons, __global double* neurons, 
	int width, int depth, int poolsize, int stride)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
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

*/
}