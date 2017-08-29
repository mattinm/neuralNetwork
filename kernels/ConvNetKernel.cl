// backprops that need the neuron vals.
//		Conv - keep the padded vals

// backprops that need info. 
// 		Activ  	1 means write the dneuron val into it.
//				0 means write 0 into it (it fell outside the range)
//	   		    .01 means write leakyReluConst * dneuron val into it

//		maxPool write the index of the maxval for the new neuron into a parallel array for the new neurons.
//			in backprop search that array for your index and if they match, += the dneurons value into it

// DO NOT CHANGE THESE DEFINES. IF YOU MUST, MAKE THEM THE SAME LENGTH IN CHARACTERS

#define RELU_CAP 5000.0 	 //max value that can pass through relu or leakyRelu
#define LEAKY_RELU_CONST .01 //multiplication constant for negative values in leakyRelu
#define l2Lambda 0.05		 //multiplication constant for L2 Regularization
#define MOMENT_CONST .9 	 //multiplication constant for momentum
#define MAX_NORM_CAP 6.0 	 //max absolute value a weight can have

#define EPSILON 1e-8         //small constant so no divide by 0

#define iterations 50        //iterations for exact_exp

#define MAX_NEURON 429494400

double exact_exp(double z) { //from the EXACT CNN builder from Travis Desell
    bool is_negative = z < 0;
    if (is_negative) z = -z;

    // exp(x) = sum (k = 0 to inf) z^k/k!
    double result = 1.0 + z;

    double prev = z;
    for (short k = 2; k < iterations; k++) {
        prev *= (z / k);
        result += prev;
    }

    if (is_negative) {
        return 1.0 / result;
    } else {
        return result;
    }
}

//All kernels ending in F are forward only kernels, which are now requiring the max neuron size

/*************************************************
*
*	ActivLayer kernels
*
*************************************************/
//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void reluF(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	if(i >= MAX_NEURON)
		printf("broke max in relu");
	// neurons[i] = clamp(prevNeurons[i], -RELU_CAP, RELU_CAP);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= RELU_CAP)
		neurons[i] = prevNeurons[i];
	else if(prevNeurons[i] < 0)
		neurons[i] = 0;
	else
		neurons[i] = RELU_CAP;
}


__kernel void relu(__global double* prevNeurons, __global double* neurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= RELU_CAP)
	{
		neurons[i] = prevNeurons[i];
		dneuronInfo[i] = 1;
	}
	else
	{
		dneuronInfo[i] = 0;
		if(prevNeurons[i] < 0)
			neurons[i] = 0;
		else
			neurons[i] = RELU_CAP;
	}
}

//prevNeurons is the set of neurons in the prev layer, the set you are writing into
__kernel void relu_back(__global double* prevdNeurons, __global double* dneurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	prevdNeurons[i] = dneuronInfo[i] * dneurons[i];	
}

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyReluF(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	if(i >= MAX_NEURON)
		printf("broke max in leaky relu");
	//double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	double newVal;
	if(prevNeurons[i] >= 0) 
		newVal = prevNeurons[i];
	else 
		newVal = prevNeurons[i] * LEAKY_RELU_CONST;

	neurons[i] = clamp(newVal, -RELU_CAP, RELU_CAP);

	if(-RELU_CAP <= newVal && newVal <= RELU_CAP)
		neurons[i] = newVal;
	else if(newVal < -RELU_CAP)
		neurons[i] = -RELU_CAP;
	else
		neurons[i] = RELU_CAP;
}


//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyRelu(__global double* prevNeurons, __global double* neurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	double newVal;
	double dneur = LEAKY_RELU_CONST;
	if(prevNeurons[i] >= 0) 
	{
		newVal = prevNeurons[i];
		dneur = 1;
	}
	else 
		newVal = prevNeurons[i] * LEAKY_RELU_CONST;

	if(-RELU_CAP <= newVal && newVal <= RELU_CAP)
		neurons[i] = newVal;
	else
	{
		dneur = 0;
		if(newVal < -RELU_CAP)
			neurons[i] = -RELU_CAP;
		else
			neurons[i] = RELU_CAP;
	}
	dneuronInfo[i] = dneur;
}

__kernel void leakyRelu_back(__global double* prevdNeurons, __global double* dneurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	prevdNeurons[i] = dneuronInfo[i] * dneurons[i];
}

/*************************************************
*
*	MaxPool kernels
*
*************************************************/

//num threads should be the size of the neurons after the maxPool
__kernel void maxPoolF(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride)
{
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	if(x >= MAX_NEURON)
		printf("broke max in max pool with x");
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
	if(i >= MAX_NEURON)
		printf("broke max in max pool with first i");
	int amountToNextLayer = (prevwidth - poolsize) * prevdepth;
	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(i >= MAX_NEURON)
				printf("broke max in max pool with i of %d\n", i);
			if(prevNeurons[i] > maxVal)
				maxVal = prevNeurons[i];
			i += prevdepth;
		}
		i += amountToNextLayer;
	}
	neurons[x] = maxVal;
}

//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride, __global int* maxIndexes)
{
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	int i = x;
	int strxdep = stride * prevdepth;
	int i_div_dep = i / prevdepth;
	int numBlocksPerRow = (prevwidth - poolsize)/stride + 1; 

	i = (i_div_dep/numBlocksPerRow * prevwidth * strxdep + i%prevdepth) + (((i_div_dep)%numBlocksPerRow) * strxdep);
	
	int amountToNextLayer = (prevwidth - poolsize) * prevdepth;
	int maxIndex = i;
	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] >= maxVal)
			{
				maxVal = prevNeurons[i];
				maxIndex = i;
			}
			i += prevdepth;
		}
		i += amountToNextLayer;
	}
	neurons[x] = maxVal;
	maxIndexes[x] = maxIndex;
}

//run for each neuron in prevdNeurons
__kernel void maxPool_back(__global double* prevdNeurons, __global double* dneurons, __global int* maxIndexes, int numIndexes, int depth)
{
	const int i = get_global_id(0);
	double result = 0;
	
	//good, not best
	for(int j= i % depth; j < numIndexes; j += depth)
		if(maxIndexes[j] == i)
			result += dneurons[j];

	prevdNeurons[i] = result;
}

//num threads should be the size of the neurons after the maxPool
__kernel void avgPool(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride)
{
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	int i = x;
	int strxdep = stride * prevdepth;
	int i_div_dep = i / prevdepth;
	int numBlocksPerRow = (prevwidth - poolsize)/stride + 1; 

	i = (i_div_dep/numBlocksPerRow * prevwidth * strxdep + i%prevdepth) + (((i_div_dep)%numBlocksPerRow) * strxdep);
	
	int amountToNextLayer = (prevwidth - poolsize) * prevdepth;
	double sum = 0;
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			sum += prevNeurons[i];
			i += prevdepth;
		}
		i += amountToNextLayer;
	}
	// sum /= poolsize * poolsize;
	neurons[x] = sum / (poolsize * poolsize);
}

int getFlatIndex(int row, int col, int depth_z, int width, int depth_size)
{
	return row * width * depth_size + col * depth_size + depth_z;
}

//run for each neuron in prevdNeurons
__kernel void avgPool_back(__global double* prevdNeurons, __global double* dneurons, int prevwidth, int depth, int poolsize, int stride)
{
	const int i = get_global_id(0);
	// printf("\npw %d depth %d pool %d stride %d\n", prevwidth, depth, poolsize, stride);
	
	int numBlocksPerRow = (prevwidth - poolsize)/stride + 1; //equals width/height of dneurons
	int d = i % depth;

	double result = 0;
	// int hits = 0;
	for(int row = 0; row < numBlocksPerRow; row++)
	{
		// printf("start block row %d\n", row);
		int rstart = stride * row;
		for(int col = 0; col < numBlocksPerRow; col++) // we are in a block
		{
			// printf("start block col %d\n", col);
			int cstart = stride * col;
			for(int r = rstart; r < rstart + poolsize; r++)
			{
				for(int c = cstart; c < cstart + poolsize; c++)
				{
					int index = getFlatIndex(r,c,d,prevwidth,depth);
					// printf("index try %d (%d,%d,%d)\n", index,r,c,d);
					if(index == i)
					{
						result += dneurons[getFlatIndex(row,col,d,numBlocksPerRow,depth)];
						// hits++;
					}
				}
			}
		}
	}
	// printf("hits = %d\n", hits);

	prevdNeurons[i] = result / (poolsize * poolsize);
}

/*************************************************
*
*	Convolution kernels (and Zero Padding)
*
*************************************************/

__kernel void convolveF(__global double* prevNeurons, __global double* neurons,
	__global double* weights, __global double* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
{
	// printf("?");
	const int i = get_global_id(0);
	int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	
	int myFilter = i%numFilters;
	int filterLayerSize = filterSize * prevdepth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int strxdep = stride * prevdepth;
	int h = ((myBlock/numBlocksPerRow) * prevwidth * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);

	int amountToNextLayer = (prevwidth - filterSize) * prevdepth;

	//can I do the pointer arithmetic better?

	double result = 0;
	if(i == 0)
		printf("max %d\n", filterSize * filterLayerSize * numFilters);
	// __global double* curWeight = &(weights[j]);
	// printf("j = %d - %d\n", j, j + filterLayerSize * filterSize);
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			if(h >= MAX_NEURON)
				printf("broke max in conv with h");
			if(j >= filterSize * filterLayerSize * numFilters)
				printf("\n\n\nbroke weights with j of %d max %d\n", j, filterSize * filterLayerSize * numFilters);
			// j++;
			// result += 1.0; //works 1 time
			result += weights[j++] * prevNeurons[h++]; //breaks
			// result += weights[j++]; // broke

			// result += *(curWeight++) * prevNeurons[h++];
			// result += prevNeurons[h++]; //worked 2 times
		}
		h += amountToNextLayer;
	}
	if(i >= MAX_NEURON)
		printf("broke max in conv with i of %d",i);
	
	neurons[i] = result + biases[myFilter];
}

__kernel void convolveConstantF(__global double* prevNeurons, __global double* neurons,
	__constant double* weights, __constant double* biases, int numFilters, int filterSize, int stride, int prevwidth, int prevdepth)
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
		}
		h += amountToNextLayer;
	}
	//printf("numFil: %d id: %d myBlock: %d\n",numFilters,get_global_id(0), myBlock);
	//printf("In convolve. Global id = %d\n\tmyFilter = %d\n\tresult = %f\n",i,myFilter,result);
	neurons[i] = result + biases[myFilter];
}

//can keep the padded vals by using the calling code
__kernel void convolve(__global double* prevNeurons, __global double* neurons,
	__global double* weights, __global double* biases, int numFilters, int filterSize, int stride,
	 int prevwidth, int prevdepth)
{
	//calculated const variables. same for all threads
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	const int filterLayerSize = filterSize * prevdepth;
	const int strxdep = stride * prevdepth;
	const int amountToNextLayer = (prevwidth - filterSize) * prevdepth;


	int i = get_global_id(0);
	
	int myFilter = i%numFilters;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int h = ((myBlock/numBlocksPerRow) * prevwidth * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);//myStartIndex
	// int h = myStartIndex;

	double result = 0;
	__global double* curWeight = &(weights[j]);
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			//result += weights[j++] * prevNeurons[h++];
			result += *(curWeight++) * prevNeurons[h++];
		}
		h += amountToNextLayer;
	}
	neurons[i] = result + biases[myFilter];
}

__kernel void convolve_back_neurons(__global double* prevdNeurons, __global double* dneurons,
	__global double* weights, int numFilters, int filterSize, int stride, int prevwidth, int depth)
{
	
	//calculated const variables and calculations used a lot
	const int x = get_global_id(0);
	const int numWeightsPerFilter = filterSize * filterSize * depth;
	const int filterLayerSize = filterSize * depth;
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	const int widthxdepth = prevwidth * depth;
	const int toNextRow = filterLayerSize + widthxdepth * (stride-1);

	//variable declarations
	int start = 0; 
	int origStart;
	int d = 0;
	double result = 0;
	int endlayer;
	int placeInFilter;

	for(int a=0; a < numBlocksPerRow; a++) //change to numBlocksPerCol to all for non-square images
	{
		for(int b = 0; b < numBlocksPerRow; b++)
		{
			origStart = start;
			if(x < origStart)
			{
				prevdNeurons[x] = result;
				return;
			}
			for(int miniHeight = 0; miniHeight < filterSize; miniHeight++)
			{
				endlayer = start + filterLayerSize;
				if(start <= x && x < endlayer)
				{
					placeInFilter = (x - start) + miniHeight*filterLayerSize;
					for(int f=0; f < numFilters; f++)
					{
						result += weights[placeInFilter] * dneurons[d+f];//[d++];
						placeInFilter += numWeightsPerFilter; // gets us to same element in next filter
					}
					//if we found it in this minilayer, it wont be in any of the others
					break;
				}
				start += widthxdepth;
			}
			d += numFilters;
			start = origStart + depth;
		}
		start = origStart + toNextRow;
	}
	prevdNeurons[x] = result;
}


__kernel void convolve_back_weights(__global double* weights, __global double* prevNeurons, __global double* dneurons,
	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize)
{
	
	const int x = get_global_id(0);
	const int numWeightsPerFilter = filterSize * filterSize * depth;
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	const int depxstr = depth * stride;
	const int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);

	int d = x/numWeightsPerFilter; //myFilter
	int p = x % numWeightsPerFilter; // my place in the filter
	double myDerivative = 0;

	for(int a=0; a < numBlocksPerRow; a++)
	{
		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
		{
			myDerivative += prevNeurons[p] * dneurons[d];
			p += depxstr;
			d += numFilters;
		}
		p += toNextBlockDown;
	}

	//L2 Reg
	myDerivative += l2Lambda * weights[x];

	//max-norm
	double myWeight = weights[x];
	myWeight -= stepSize * myDerivative;
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;

	//w/out max-norm
	//weights[x] -= stepSize * myDerivative;
	
}
__kernel void convolve_back_weights_moment(__global double* weights, __global double* prevNeurons, __global double* dneurons,
	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize, __global double* velocity)
{
	
	const int x = get_global_id(0);
	const int numWeightsPerFilter = filterSize * filterSize * depth;
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	const int depxstr = depth * stride;
	const int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);

	int d = x/numWeightsPerFilter; //myFilter
	int p = x % numWeightsPerFilter; // my place in the filter
	double myDerivative = 0;

	for(int a=0; a < numBlocksPerRow; a++)
	{
		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
		{
			myDerivative += prevNeurons[p] * dneurons[d];
			p += depxstr;
			d += numFilters;
		}
		p += toNextBlockDown;
	}

	//L2 Reg?
	myDerivative += l2Lambda * weights[x];// * weights[x];

	//normal momentum
	//double myVel = velocity[x];
	//myVel = MOMENT_CONST * myVel - stepSize * myDerivative;

	//Nesterov Accelerated Momentum
	double myVel = velocity[x];
	double prevVel = myVel;
	myVel = MOMENT_CONST * myVel - stepSize * myDerivative;

	//max-norm
	double myWeight = weights[x];
	//myWeight += myVel; //normal momentum
	myWeight += -MOMENT_CONST * prevVel + (1+MOMENT_CONST) * myVel; // Nesterov Momentum
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;
}

//should have numBiases work units
__kernel void convolve_back_biases(__global double* biases, __global double* dneurons, int dneuronSize, 
	int dneuronDepth, double stepSize)
{
	
	const int i = get_global_id(0);
	const int dneurFaceSize = dneuronSize/dneuronDepth;

	int j = i;//%dneuronDepth //which filter we're in. dont need % because there is only one bias per filter
	double myDerivative = 0;
	for(int a = 0; a< dneurFaceSize; a++) // calc myDerivative
	{
		myDerivative += dneurons[j];
		j+=dneuronDepth;
	}
	//printf("stepSize %lf myDerivative %lf, result %.9lf\n", stepSize, myDerivative, stepSize * myDerivative);
	biases[i] -= stepSize * myDerivative; // update bias

}
__kernel void convolve_back_weights_no_update_accum(__global double* weights, __global double* prevNeurons, __global double* dneurons,
	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize, __global double* dweights)
{
	// printf("ori\n");
	const int x = get_global_id(0);
	const int numWeightsPerFilter = filterSize * filterSize * depth;
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	const int depxstr = depth * stride;
	const int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);
	double myDerivative = 0;

	// int d = x/numWeightsPerFilter; //myFilter
	// int p = x % numWeightsPerFilter; // my place in the filter
	// for(int a=0; a < numBlocksPerRow; a++)
	// {
	// 	for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
	// 	{
	// 		myDerivative += prevNeurons[p] * dneurons[d];
	// 		p += depxstr;
	// 		d += numFilters;
	// 	}
	// 	p += toNextBlockDown;
	// }

	__global double *pptr = prevNeurons + x % numWeightsPerFilter;
	__global double *dptr = dneurons + x/numWeightsPerFilter;
	for(int a=0; a < numBlocksPerRow; a++)
	{
		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
		{
			myDerivative += *pptr * *dptr;
			pptr += depxstr;
			dptr += numFilters;
		}
		pptr += toNextBlockDown;
	}

	//L2 Reg?
	myDerivative += l2Lambda * weights[x];
	// printf("%d old %lf my %lf new %lf\n", x, dweights[x],myDerivative,dweights[x]+myDerivative);

	dweights[x] += myDerivative;
}

//should have numBiases work units
__kernel void convolve_back_biases_no_update_accum(__global double* biases, __global double* dneurons, int dneuronSize, 
	int dneuronDepth, double stepSize, __global double* dbiases)
{
	
	const int i = get_global_id(0);
	const int dneurFaceSize = dneuronSize/dneuronDepth;

	int j = i;//%dneuronDepth //which filter we're in. dont need % because there is only one bias per filter
	double myDerivative = 0;
	for(int a = 0; a< dneurFaceSize; a++) // calc myDerivative
	{
		myDerivative += dneurons[j];
		j+=dneuronDepth;
	}
	//printf("stepSize %lf myDerivative %lf, result %.9lf\n", stepSize, myDerivative, stepSize * myDerivative);
	dbiases[i] += myDerivative;
	// biases[i] -= stepSize * myDerivative; // update bias

}

__kernel void zero_out(__global double* mem)
{
	mem[get_global_id(0)] = 0;
}

__kernel void update_weights(__global double* weights, __global double* dweights, double stepSize)
{
	const int x = get_global_id(0);

	double myWeight = weights[x];
	myWeight -= stepSize * dweights[x];
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;

	// printf("%d %lf %lf\n", x, dweights[x], weights[x]);
}

__kernel void update_weights_moment(__global double* weights, __global double* dweights, double stepSize, __global double* velocity)
{
	const int x = get_global_id(0);

	//Nesterov Accelerated Momentum
	double myVel = velocity[x];
	double prevVel = myVel;
	myVel = MOMENT_CONST * myVel - stepSize * dweights[x];

	//max-norm
	double myWeight = weights[x];
	//myWeight += myVel; //normal momentum
	myWeight += -MOMENT_CONST * prevVel + (1+MOMENT_CONST) * myVel; // Nesterov Momentum
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;
}

__kernel void update_biases(__global double* biases, __global double* dbiases, double stepSize)
{
	const int i = get_global_id(0);
	biases[i] -= stepSize * dbiases[i];
}

//////////////////////////
//END MINIBATCH
//////////////////////////

__kernel void zeroPad(__global double *prevNeurons, __global double *neurons, int pad, int prevwidth,
	int prevheight, int depth, int newSize)
{
	const int i = get_global_id(0);
	if(i >= MAX_NEURON)
		printf("broke max in zeropad with i\n");
	const int ourImage = i / newSize;
	int x = i - newSize * ourImage; // make x the number it would be if there was only one image in neurons
	const int nw = prevwidth + 2*pad;
	const int nh = prevheight + 2*pad;

	int ourDepth = x%depth;
	int ourCol = ((x-ourDepth)/depth) % nw;
	int ourRow = ((x-ourDepth)/depth) / nw;

	int ourRealDepth = i%depth;
	int ourRealCol = ((i-ourDepth)/depth) % nw;
	int ourRealRow = ((i-ourDepth)/depth) / nw;

	if(ourRow < pad || ourRow >= nh-pad || ourCol < pad || ourCol >= nw-pad)
		neurons[i] = 0;
	else
	{
		int i = ourRealRow - pad - ourImage * 2 * pad;
		int j = ourRealCol - pad;
		int k = ourRealDepth;
		int oldIndex = (i * prevwidth * depth) + (j * depth) + k;

		if(i >= MAX_NEURON)
			printf("broke max in zeropad with oldIndex of %d\n", oldIndex);
		neurons[i] = prevNeurons[oldIndex];
	}
}

// run on each of the padded dneurons. NOT prevdNeurons!
__kernel void zeroPad_back(__global double* prevdNeurons, __global double* dneurons, int pad, int prevwidth,
	int prevheight, int depth, int newSize)
{
	const int x = get_global_id(0);
	const int nw = prevwidth + 2*pad;
	const int nh = prevheight + 2*pad;

	int ourDepth = x%depth;
	int ourCol = ((x-ourDepth)/depth) % nw;
	int ourRow = ((x-ourDepth)/depth) / nw;

	if(!(ourRow < pad || ourRow >= nh-pad || ourCol < pad || ourCol >= nw-pad))
	{
		int i = ourRow - pad;
		int j = ourCol - pad;
		int k = ourDepth;
		int oldIndex = (i * prevwidth * depth) + (j * depth) + k;

		prevdNeurons[oldIndex] = dneurons[x];
	}
}

/*************************************************
*
*	Softmax and Softmax helper kernels
*
*************************************************/

__kernel void softmax_allCL(__global double *prevNeurons, __global double *neurons,
	__global double* denominator, int size)
{
	int i = get_global_id(0);
	if(i >= MAX_NEURON)
		printf("broke max in softmax with i");
	int d = i / size;
	neurons[i] = exp(prevNeurons[i])/(denominator[d]);
	// printf("CNN Output - Class %d: conf %lf\n", i, neurons[i],*denominator);
}

//pushes the derivatives into prevdNeurons
__kernel void softmax_back(__global double* dNeurons, __global double* neurons, int trueVal)
{
	int i = get_global_id(0);

	if(i == trueVal)
	{
		dNeurons[i] = neurons[i] - 1;
	}
	else
	{
		dNeurons[i] = neurons[i];
	}

	// printf("SoftBack - Class %d: gradient %lf, trueVal %d\n",i,dNeurons[i],trueVal);
}

__kernel void maxSubtraction(__global double* source, int size)
{
	const int x = get_global_id(0);
	int start = size * x;
	int end = start + size;
	if(end >= MAX_NEURON)
		printf("broke max in max sub with end");
	if(size <= 0)
		return;
	double max = source[start];
	double cur;
	for(int i = start+1; i < end; i ++)
	{
		cur = source[i];
		if(cur > max)
			max = cur;
	}
	for(int i=start; i < end; i++)
		source[i] -= max;
}

__kernel void vectorESum(__global double* source, int size, __global double* denom)
{
	const int x = get_global_id(0);
	if(x >= MAX_NEURON)
		printf("broke max in vec e sum with x");
	int start = size * x;
	int end = start + size;
	if(end >= MAX_NEURON)
		printf("broke max in vec e sum with end");
	if(size <= 0)
		return;
	double sum = 0;
	for(int i=start; i < end; i++)
		sum += exact_exp(source[i]);
	denom[x] = sum;
}

/*************************************************
*
*	Batch Normalization kernels
*
*************************************************/

__kernel void batch_norm_run(__global double* prevNeurons, __global double* neurons, const __global double* gamma, const __global double* beta, 
	const __global double* e, const __global double* var, int depth)
{
	int x = get_global_id(0);
	int k = x;
	if(depth > 0)
		k = x % depth;
	double rootVarPlusEps = pow(var[k] + EPSILON,0.5);
	double gam = gamma[k];
	double front = gam * prevNeurons[x] / rootVarPlusEps;
	double back = beta[k] - gam * e[k] / rootVarPlusEps;

	if(x >= MAX_NEURON)
		printf("broke max in batch norm with x");
	neurons[x] = front + back;
}

__kernel void batch_norm(__global double* prevNeurons, __global double* neurons, const __global double* gamma, const __global double* beta, 
	const __global double* mu, const __global double* sigma_squared, int depth)
{
	int x = get_global_id(0);
	int k = x;
	if(depth > 0)
		k = x % depth;
	double xhat = (prevNeurons[x] - mu[k])/pow(sigma_squared[k] + EPSILON, 0.5);

	neurons[x] = gamma[k] * xhat + beta[k];	
}

__kernel void batch_norm_back(__global double* prevdNeurons, __global double* dNeurons, int depth, __global double* gamma, __global double* mu, 
	__global double* sigma2, __global double* delta_mu, __global double* delta_sigma2, __global double* bn_x, int minibatch_size)
{
	int i = get_global_id(0);
	int k = i;
	if(depth > 0)
		k = i % depth;
	double delta_xhat = dNeurons[i] * gamma[k];
	// double delta_x = delta_xhat / pow(sigma2[k] + EPSILON, 0.5) + delta_sigma2[k] * 2 * (bn_x[i] - mu[k]) / minibatch_size
	// 	+ delta_mu[k] / minibatch_size;
	double delta_x = delta_xhat / pow(sigma2[k] + EPSILON, 0.5) + (delta_sigma2[k] * 2.0 * (bn_x[i] - mu[k])
		+ delta_mu[k]) / minibatch_size;
	prevdNeurons[i] = delta_x;

	//rough approximation
	// prevdNeurons[i] = 1.0 / pow(sigma2[k] + EPSILON, 0.5) * delta_xhat;
	// printf("index: %d sigma2: %lf gamma: %lf depth: %d -- dn %lf -> pdn %lf\n", i, sigma2[k], gamma[k], depth, dNeurons[i], prevdNeurons[i]);
	// prevdNeurons[i] = dNeurons[i] * gamma[k];
}

__kernel void update_gamma_and_beta(__global double* gamma, __global double* beta, __global double* delta_gamma, __global double* delta_beta, double stepSize)
{
	int i = get_global_id(0);
	gamma[i] -= delta_gamma[i] * stepSize;
	beta[i] -= delta_beta[i] * stepSize;
}


/*************************************************
*
*	Helper kernels
*
*************************************************/

__kernel void copyArray(__global double* source, __global double* dest)
{
	int x = get_global_id(0);
	dest[x] = source[x];
}

__kernel void plusEquals(__global double* dest, __global double* src)
{
	int x = get_global_id(0);
	dest[x] += src[x];
}

__kernel void divideEquals(__global double* dest, int num)
{
	int x = get_global_id(0);
	dest[x] /= num;
	printf("Avg gradient - Class %d: gradient %lf\n", x, dest[x]);
}