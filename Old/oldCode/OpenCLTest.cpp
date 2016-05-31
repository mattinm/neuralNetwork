#ifdef __APPLE__
    #include "OpenCL/opencl.h"
	#include "OpenCL/cl.h"
#else
    #include "CL/cl.h"
#endif

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		//std::exit (1);
	}
}

cl_program CreateProgram (const std::string& source, cl_context& context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

int main(void)
{
	char name[128];

	//make test data to add
	static const size_t testDataSize = 10;
	vector<float> x(testDataSize),y(testDataSize),z(testDataSize);
	for(int i=0; i< testDataSize; i++)
	{
		x[i] = 1.0 * i;
		y[i] = 3.0 * i;
	}

	//OpenCL stuff
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);

	cout << "Platform id count: " <<  platformIdCount << endl;

	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
	cout << "Device id count: " << deviceIdCount << endl;

	vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

	for(int i=0; i< deviceIdCount; i++)
	{
		clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, 128, name, NULL);
		cout << name << endl;
	}

	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	cout << "contextProperties made" << endl;

	cl_int error;
	cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), 
		nullptr, nullptr, &error);
	cout << "context made" << endl;

	cl_program program = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), context);
	cout << "program made" << endl;

	//Note: clBuildProgram returns -11 if the 5th argument is null or nullptr
	cl_int buil = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
	cout << "Success = " << CL_SUCCESS <<  endl;
	if(buil != CL_SUCCESS)
	{
		cout << "program build failed " << buil << endl;
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, deviceIds[2], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout << "--- Build log ---\n" << buffer << endl;
		//exit(1);
	}
	cout << "program built" << endl;

	cl_kernel convKernel = clCreateKernel(program, "convolve", &error);
	CheckError(error);

	cl_kernel softmaxKernel = clCreateKernel(program, "softmax", &error);
	CheckError(error);
	cout << "kernel made" << endl;

	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[2], 0, &error);
	CheckError(error);
	cout << "queue made" << endl;

	cl_mem xBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(float) * testDataSize, x.data(), &error);
	CheckError(error);


	cl_mem yBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(float) * testDataSize, y.data(), &error);
	CheckError(error);
/*
	cl_mem zBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(float) * testDataSize, nullptr, &error);
	CheckError(error);
*/
	double denom = 1;
	clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), &xBuf);
	clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), &yBuf);
	clSetKernelArg(softmaxKernel, 2, sizeof(double), &denom);
/*
	clSetKernelArg(subkernel, 0, sizeof(cl_mem), &xBuf);
	clSetKernelArg(subkernel, 1, sizeof(cl_mem), &yBuf);
	clSetKernelArg(subkernel, 2, sizeof(cl_mem), &zBuf);
*/
	const size_t globalWorkSize[] = {testDataSize, 0, 0};
	CheckError(clEnqueueNDRangeKernel(queue, softmaxKernel, 1,
		nullptr,
		globalWorkSize,
		nullptr,
		0, nullptr, nullptr));

	clFinish(queue);

	CheckError(clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0,
		sizeof(float) * testDataSize,
		z.data(), 0, nullptr, nullptr));

	for(int i=0; i< testDataSize-1; i++)
	{
		cout << z[i] << ", ";
	}
	cout <<z[testDataSize-1] << endl;
/*
	CheckError(clEnqueueNDRangeKernel(queue, addkernel, 1,
		nullptr,
		globalWorkSize,
		nullptr,
		0, nullptr, nullptr));
	clFinish(queue);

	CheckError(clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0,
		sizeof(float) * testDataSize,
		z.data(), 0, nullptr, nullptr));
*/
	for(int i=0; i< testDataSize-1; i++)
	{
		cout << z[i] << ", ";
	}
	cout <<z[testDataSize-1] << endl;

	clReleaseCommandQueue(queue);
	clReleaseMemObject(xBuf);
	clReleaseMemObject(yBuf);

	clReleaseKernel(convKernel);
	clReleaseKernel(softmaxKernel);
	clReleaseProgram(program);

	clReleaseContext(context);
}















