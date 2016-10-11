static int GPU_POSITION(int x, int y, int window_width)
{
	return ((y * window_width) + x);
}

static int GPU_POSITION3(int x, int y, int z, int window_width)
{
	return ((y * window_width) + x) * 3 + z;
}


//unique x
__kernel void vcolumnCalcCosts(__global float* costs, __global char* dirs, __global float* vals, unsigned int window_width, unsigned int window_height, unsigned int count)
{ 
	unsigned int x = get_global_id(0);
	if(x >= window_width - count)
		return;

	costs[GPU_POSITION(x,window_height-1,window_width)] = vals[GPU_POSITION(x,window_height-1,window_width)];
	dirs[GPU_POSITION(x,window_height-1,window_width)] = 0;

	unsigned int xmax = window_width - count - 1;
	float cost_left, cost_up, cost_right;
	int mypos;
	for(int y = window_height - 2; y >= 0; y--)
	{
		barrier(CLK_GLOBAL_MEM_FENCE);
		//do left side
		if(x == 0)
		{
			if(costs[GPU_POSITION(0,y+1,window_width)] < costs[GPU_POSITION(1,y+1,window_width)])
			{
				costs[GPU_POSITION(0,y,window_width)] = vals[GPU_POSITION(0,y,window_width)] + costs[GPU_POSITION(0,y+1,window_width)];
				dirs[GPU_POSITION(0,y,window_width)] = 0;
			}
			else
			{
				costs[GPU_POSITION(0,y,window_width)] = vals[GPU_POSITION(0,y,window_width)] + costs[GPU_POSITION(1,y+1,window_width)];
				dirs[GPU_POSITION(0,y,window_width)] = 1;
			}
		}
		else if(x < xmax)// x < window_width - count - 1;
		{
			cost_left  = costs[GPU_POSITION(x-1, y+1, window_width)];
			cost_up    = costs[GPU_POSITION(x  , y+1, window_width)];
			cost_right = costs[GPU_POSITION(x+1, y+1, window_width)];
			mypos = GPU_POSITION(x,y,window_width);

			if(cost_left < cost_up && cost_left < cost_right)
			{
				costs[mypos] = vals[mypos] + cost_left;
				dirs[mypos] = -1;
			}
			else if(cost_right< cost_up && cost_right < cost_left)
			{
				costs[mypos] = vals[mypos] + cost_right;
				dirs[mypos] =1;
			}
			else 
			{
				costs[mypos] = vals[mypos] + cost_up;
				dirs[mypos] = 0;
 			}
		}
		else
		{
			//do the right side
			if (costs[GPU_POSITION(x, y+1, window_width)] < costs[GPU_POSITION(x-1, y+1, window_width)]) 
			{
				costs[GPU_POSITION(x, y, window_width)] = vals[GPU_POSITION(x, y, window_width)] + costs[GPU_POSITION(x, y+1, window_width)];
				dirs[GPU_POSITION(x, y, window_width)] = 0;
			}
			else
			{
				costs[GPU_POSITION(x, y, window_width)] = vals[GPU_POSITION(x, y, window_width)] + costs[GPU_POSITION(x-1, y+1, window_width)];
				dirs[GPU_POSITION(x, y, window_width)] = -1;
			}
		}
	}
}

//need 1 thread
__kernel void vcalcSeamToRemove(__global float* costs, __global char *dirs, __global int* seam, unsigned int window_width, unsigned int window_height, unsigned int count, __global float* vmin)
{
	float min_val = 200000;
	for(unsigned int x = 0; x < window_width - count; x++)
	{
		if(costs[GPU_POSITION(x,0,window_width)] < min_val)
		{
			min_val = costs[GPU_POSITION(x,0,window_width)];
			seam[0] = x;
		}
	}
	*vmin = min_val;
	// printf("min_val %f, seam[0] %d\n", min_val, seam[0]);
	for(unsigned int y = 1; y < window_height; y++)
	{
		seam[y] = seam[y-1] + dirs[GPU_POSITION(seam[y-1], y-1, window_width)];
		// printf("dirs[%d][%d] = %d\n", seam[y-1], y-1, dirs[GPU_POSITION(seam[y-1], y-1, window_width)]);
	}
}

__kernel void vseamremove(__global int *image, __global float* vals, __global int* seam, unsigned int window_width, unsigned int window_height, unsigned int count)
{
	unsigned int y =get_global_id(0);
	if(y>=window_height)
		return;
	unsigned int x;
	for (x = seam[y]; x <window_width - count -1; x++) 
	{
		image[GPU_POSITION3(x, y, 0, window_width)] = image[GPU_POSITION3(x+1, y, 0, window_width)];
		image[GPU_POSITION3(x, y, 1, window_width)] = image[GPU_POSITION3(x+1, y, 1, window_width)];
		image[GPU_POSITION3(x, y, 2, window_width)] = image[GPU_POSITION3(x+1, y, 2, window_width)];

		vals[GPU_POSITION(x, y, window_width)] = vals[GPU_POSITION(x+1, y, window_width)];
	}

	image[GPU_POSITION3(x, y, 0, window_width)] = 0;
	image[GPU_POSITION3(x, y, 1, window_width)] = 0;
	image[GPU_POSITION3(x, y, 2, window_width)] = 0;

	vals[GPU_POSITION(x, y, window_width)] = 0;
}

__kernel void hrowCalcCosts(__global float* costs, __global char* dirs, __global float* vals, unsigned int window_width, unsigned int window_height, unsigned int count)
{ 
	unsigned int y = get_global_id(0);
	if(y >= window_height - count)
		return;

	// printf("%d\n", GPU_POSITION(window_width - 1, y, window_width));
	costs[GPU_POSITION(window_width - 1, y, window_width)] = vals[GPU_POSITION(window_width - 1, y, window_width)];
	dirs[GPU_POSITION(window_width - 1, y, window_width)] = 0;

	unsigned int ymax = window_height - count - 1;
	float cost_left, cost_up, cost_down;
	int mypos;
	for(int x = window_width - 2; x >= 0; x--)
	{
		barrier(CLK_GLOBAL_MEM_FENCE);
		//do left side
		if(y == 0)
		{
			if(costs[GPU_POSITION(x+1,0,window_width)] < costs[GPU_POSITION(x+1,1,window_width)])
			{
				costs[GPU_POSITION(x,0,window_width)] = vals[GPU_POSITION(x,0,window_width)] + costs[GPU_POSITION(x+1,0,window_width)];
				dirs[GPU_POSITION(x,0,window_width)] = 0;
			}
			else
			{
				costs[GPU_POSITION(x,0,window_width)] = vals[GPU_POSITION(x,0,window_width)] + costs[GPU_POSITION(x+1,1,window_width)];
				dirs[GPU_POSITION(x,0,window_width)] = 1;
			}
		}
		else if(y < ymax)// x < window_width - count - 1;
		{
			cost_up    = costs[GPU_POSITION(x+1, y-1, window_width)];
			cost_left  = costs[GPU_POSITION(x+1, y  , window_width)];
			cost_down  = costs[GPU_POSITION(x+1, y+1, window_width)];
			mypos = GPU_POSITION(x,y,window_width);

			if(cost_up < cost_left && cost_up < cost_down)
			{
				costs[mypos] = vals[mypos] + cost_up;
				dirs[mypos] = -1;
			}
			else if(cost_down < cost_left)
			{
				// printf("%d\n", 1);
				costs[mypos] = vals[mypos] + cost_down;
				dirs[mypos] = 1;
			}
			else 
			{
				costs[mypos] = vals[mypos] + cost_left;
				dirs[mypos] = 0;
 			}
		}
		else
		{
			//do the right side
			mypos = GPU_POSITION(x,y,window_width);
			if (costs[GPU_POSITION(x+1, y, window_width)] < costs[GPU_POSITION(x+1, y-1, window_width)]) 
			{
				costs[mypos] = vals[mypos] + costs[GPU_POSITION(x+1, y, window_width)];
				dirs[mypos] = 0;
			}
			else
			{
				costs[mypos] = vals[mypos] + costs[GPU_POSITION(x+1, y-1, window_width)];
				dirs[mypos] = -1;
			}
		}
	}
}

__kernel void hcalcSeamToRemove(__global float* costs, __global char *dirs, __global int* seam, unsigned int window_width, unsigned int window_height, unsigned int count, __global float* hmin)
{
	float min_val = 2000000;
	for(unsigned int y = 0; y < window_height - count; y++)
	{
		if(costs[GPU_POSITION(0,y,window_width)] < min_val)
		{
			min_val = costs[GPU_POSITION(0,y,window_width)];
			seam[0] = y;
		}
	}
	*hmin = min_val;
	// printf("min_val %f, seam[0] %d\n", min_val, seam[0]);
	for(unsigned int x = 1; x < window_width; x++)
	{
		seam[x] = seam[x-1] + dirs[GPU_POSITION(x-1, seam[x-1], window_width)];
		// printf("dirs[%d][%d] = %d\n", seam[y-1], y-1, dirs[GPU_POSITION(seam[y-1], y-1, window_width)]);
	}
}

__kernel void hseamremove(__global int *image, __global float* vals, __global int* seam, unsigned int window_width, unsigned int window_height, unsigned int count)
{
	unsigned int x =get_global_id(0);
	if(x >= window_width)
		return;
	unsigned int y;
	for (y = seam[x]; y < window_height - count -1; y++) 
	{
		image[GPU_POSITION3(x, y, 0, window_width)] = image[GPU_POSITION3(x, y+1, 0, window_width)];
		image[GPU_POSITION3(x, y, 1, window_width)] = image[GPU_POSITION3(x, y+1, 1, window_width)];
		image[GPU_POSITION3(x, y, 2, window_width)] = image[GPU_POSITION3(x, y+1, 2, window_width)];

		vals[GPU_POSITION(x, y, window_width)] = vals[GPU_POSITION(x, y+1, window_width)];
	}

	image[GPU_POSITION3(x, y, 0, window_width)] = 0;
	image[GPU_POSITION3(x, y, 1, window_width)] = 0;
	image[GPU_POSITION3(x, y, 2, window_width)] = 0;

	vals[GPU_POSITION(x, y, window_width)] = 0;
}