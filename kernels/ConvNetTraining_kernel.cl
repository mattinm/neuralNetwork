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
#define MAX_NEURON_SIZE 50000

// END DEFINES

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

/*************************************************
*
*	ActivLayer kernels
*
*************************************************/

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
	
	//good
	for(int j= i % depth; j < numIndexes; j += depth)
	{
		if(maxIndexes[j] == i)
		{
			result += dneurons[j];
		}
	}

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
	sum /= poolsize * poolsize;
	neurons[x] = sum;
}

//run for each neuron in prevdNeurons
__kernel void avgPool_back(__global double* prevdNeurons, __global double* dneurons, __global int* maxIndexes, int numIndexes, int depth)
{
	const int i = get_global_id(0);
	double result = 0;
	
	//good
	for(int j= i % depth; j < numIndexes; j += depth)
	{
		if(maxIndexes[j] == i)
		{
			result += dneurons[j];
		}
	}

	prevdNeurons[i] = result;
}

/*************************************************
*
*	Convolution kernels (and Zero Padding)
*
*************************************************/

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

//////////////////////////
//MINIBATCH
//////////////////////////
// __kernel void convolve_back_weights_no_update_accum(__global double* weights, __global double* prevNeurons, __global double* dneurons,
// 	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize, __global double* dweights)
// {
// 	// printf("new\n");
// 	// printf("convolve_back_weights_no_update_accum: array size %d\n",MAX_NEURON_SIZE);
// 	const int x = get_global_id(0);
// 	const int numWeightsPerFilter = filterSize * filterSize * depth;
// 	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
// 	const int depxstr = depth * stride;
// 	const int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);
// 	int d = x/numWeightsPerFilter; //myFilter
// 	int p = x % numWeightsPerFilter; // my place in the filter

// 	int fs2 = filterSize * filterSize;
// 	int num_weights = fs2 * numFilters;
// 	int nw = (prevwidth - filterSize) / stride + 1;
// 	int nw2 = nw * nw;
// 	int nmove = nw2 / num_weights;
// 	int remainder = nw2 % num_weights;
// 	int rstart = nw2 - remainder;


// 	__local double testArray[MAX_NEURON_SIZE]; //fill in with dneurons but with depth as the outermost dim instead of the innermost
	

// 	int nend = x * nmove + nmove;
// 	// printf("fs2 %d nw %d nw2 %d nmove %d remainder %d rstart %d nstart %d nend %d numFilters %d\n", 
// 		// fs2,nw,nw2,nmove,remainder,rstart, x * nmove, nend, numFilters);
// 	for(int n = x * nmove; n < nend; n++)
// 	{
// 		int dn_depth = n % numFilters; //numFilters is the depth of dneurons
// 		// int dn_face_location = (n - dn_depth) / dn_depth;
// 		int newIndex = dn_depth * fs2 + (n - dn_depth) / dn_depth;// dn_depth + dn_face_location
// 		// if(newIndex > MAX_NEURON_SIZE)
// 			// printf("BADDDDDDD\n\n\n\n\n");
// 		testArray[newIndex] = dneurons[n];
// 	}
// 	if(x < remainder)
// 	{
// 		int n = x + rstart;
// 		int dn_depth = n % numFilters; //numFilters is the depth of dneurons
// 		int newIndex = dn_depth * fs2 + (n - dn_depth) / dn_depth;
// 		// if(newIndex > MAX_NEURON_SIZE)
// 			// printf("BADDDDDDD remainder\n\n\n\n\n");
// 		testArray[newIndex] = dneurons[n];
// 	}

// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	// printf("post barrier\n");

// 	// printf("t[0] %lf\n", testArray[0]);

// 	double myDerivative = 0;
// 	int t = d * fs2;
// 	for(int a=0; a < numBlocksPerRow; a++)
// 	{
// 		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
// 		{
// 			// printf("t: %d\n", t);
// 			// if(t > MAX_NEURON_SIZE)
// 				// printf("BADDDDDDD t\n\n\n\n\n");
// 			myDerivative += prevNeurons[p] * testArray[t]; //dneurons[d];
// 			p += depxstr;
// 			// d += numFilters;
// 			t++;
// 		}
// 		p += toNextBlockDown;
// 	}

// 	//L2 Reg?
// 	myDerivative += l2Lambda * weights[x];
// 	// printf("%d old %lf my %lf new %lf\n", x, dweights[x],myDerivative,dweights[x]+myDerivative);

// 	dweights[x] += myDerivative;
// 	// printf("end\n");
// }

//original
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
	int prevheight, int depth)
{
	const int x = get_global_id(0);
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

// run on each of the padded dneurons. NOT prevdNeurons!
__kernel void zeroPad_back(__global double* prevdNeurons, __global double* dneurons, int pad, int prevwidth,
	int prevheight, int depth)
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
*	Softmax kernels
*
*************************************************/

__kernel void softmax(__global double *prevNeurons, __global double *neurons, double denominator)
{
	int i = get_global_id(0);
	neurons[i] = exact_exp(prevNeurons[i])/denominator;
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
		sum += exact_exp(source[i]);
	*denom = sum;
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

/*************************************************
*
*	Forward-only kernels
*
*************************************************/

__kernel void reluF(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= RELU_CAP)
		neurons[i] = prevNeurons[i];
	else if(prevNeurons < 0)
		neurons[i] = 0;
	else
		neurons[i] = RELU_CAP;
}

//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyReluF(__global double* prevNeurons, __global double* neurons)
{
	const int i = get_global_id(0);
	//double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	double newVal;
	if(prevNeurons[i] >= 0) 
		newVal = prevNeurons[i];
	else 
		newVal = prevNeurons[i] * LEAKY_RELU_CONST;

	if(-RELU_CAP <= newVal && newVal <= RELU_CAP)
		neurons[i] = newVal;
	else if(newVal < -RELU_CAP)
		neurons[i] = -RELU_CAP;
	else
		neurons[i] = RELU_CAP;
}

__kernel void maxPoolF(__global double* prevNeurons, __global double* neurons,
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
	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] > maxVal)
				maxVal = prevNeurons[i];
			i += depth;
		}
		i += amountToNextLayer;
	}
	neurons[x] = maxVal;
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
	// printf("bnr e %lf, var %lf\n", e[k], var[k]);
	double rootVarPlusEps = pow(var[k] + EPSILON,0.5);
	double gam = gamma[k];
	// if(var[k] < 0) printf("why?\n");
	double front = gam * prevNeurons[x] / rootVarPlusEps;
	double back = beta[k] - gam * e[k] / rootVarPlusEps;
	// if(isnan(prevNeurons[x])) printf("nan px\n");

	neurons[x] = front + back;
	
	// if(isnan(neurons[x])) printf("nan y\n");
}

__kernel void batch_norm(__global double* prevNeurons, __global double* neurons, const __global double* gamma, const __global double* beta, 
	const __global double* mu, const __global double* sigma_squared, int depth)
{
	int x = get_global_id(0);
	int k = x;
	if(depth > 0)
		k = x % depth;
	double xhat = (prevNeurons[x] - mu[k])/pow(sigma_squared[k] + EPSILON, 0.5);
	// printf("index: %d x: %lf gamma: %lf beta: %lf mu: %lf sigma2: %lf depth: %d xhat: %lf y: %lf\n", x,prevNeurons[x],gamma[k],beta[k],mu[k],sigma_squared[k],depth,xhat,gamma[k] * xhat + beta[k]);
	// if(x == 0)
	// 	printf("xhat[0] gpu: %lf\n",xhat);
	neurons[x] = gamma[k] * xhat + beta[k];	
	// neurons[x] = gamma[k] * prevNeurons[x] + beta[k];
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

	// prevdNeurons[i] = 1.0 / pow(sigma2[k] + EPSILON, 0.5) * delta_xhat;
	// printf("index: %d sigma2: %lf gamma: %lf depth: %d -- dn %lf -> pdn %lf\n", i, sigma2[k], gamma[k], depth, dNeurons[i], prevdNeurons[i]);
	// prevdNeurons[i] = dNeurons[i] * gamma[k];
}

__kernel void update_gamma_and_beta(__global double* gamma, __global double* beta, __global double* delta_gamma, __global double* delta_beta, double stepSize)
{
	int i = get_global_id(0);
	// double og = gamma[i], ob = beta[i];
	gamma[i] -= delta_gamma[i] * stepSize;
	beta[i] -= delta_beta[i] * stepSize;

	// printf("index: %d og: %lf ng: %lf ob: %lf nb: %lf --- dg: %lf db: %lf\n", i, og, gamma[i], ob, beta[i], delta_gamma[i], delta_beta[i]);
}
