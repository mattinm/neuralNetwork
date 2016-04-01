__kernel void ADD(__global float* x, __global float* y, __global float* a)
{
	const int i = get_global_id(0);

	a[i] = x[i] + y[i];
}

__kernel void SUB(__global float* x, __global float* y, __global float* a)
{
	const int i = get_global_id(0);

	a[i] = x[i] - y[i];
}