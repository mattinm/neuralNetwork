//this file builds kernels to check for syntax errors

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
	#include "OpenCL/cl.h"
#else
    #include "CL/cl.h"
#endif

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

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
		std::exit (1);
	}
}

cl_program CreateProgram (const std::string& source, cl_context context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		cout << "Need one argument. \n./kernelBuilder pathToKernelToBuild" << endl;
		return -1;
	}
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);

	//cout << "Platform id count: " <<  platformIdCount << endl;

	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceIdCount);
	//cout << "Device id count: " << deviceIdCount << endl;

	vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_CPU, deviceIdCount, deviceIds.data(), nullptr);

	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	//cout << "contextProperties made" << endl;

	cl_int error;
	cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), 
		nullptr, nullptr, &error);
	//cout << "context made" << endl;

	cl_program program = CreateProgram(LoadKernel(argv[1]), context);
	//cout << "program made" << endl;

	cout << "Num devices: " << deviceIdCount << endl;
	
	error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
	if(error != CL_SUCCESS)
	{
		cout << "Error: " << error << endl;
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);

		cout << "--- Build log ---\n";
		//if(buffer != string("Error getting function data from server"))
			cout << buffer << endl;
		//exit(1);
	}
	else
	{
		cout << "Build succeeded" << endl;
	}

	clReleaseProgram(program);

	clReleaseContext(context);
}