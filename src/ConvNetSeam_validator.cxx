#include "config.h"
#include "util.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "validate_util.h"
#include "validate_util2.h"
#include "md5_file.h"
#include "error_numbers.h"
#include "stdint.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "ConvNetEvent.h"

//from UNDVC_COMMON
#include "parse_xml.hxx"
#include "file_io.hxx"

using namespace std;

struct cnn_output
{
	string cnn_config_id;
	string video_id; 
	Observations obs;
};

static inline std::string &ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}

static inline std::string &rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

static inline std::string &trim(std::string &s)
{
	return ltrim(rtrim(s));
}

int init_result(RESULT& result, void*& data)
{
	//vector<OUTPUT_FILE_INFO> files;
	OUTPUT_FILE_INFO fi;
	try
	{
		string eventString = parse_xml<string>(result.stderr_out, "error");
		stringstream ss(eventString);

		string temp;
		getline(ss, temp, '\n');
		while(getline(ss, temp, '\n'))
		{
			trim(temp);
			log_messages.printf(MSG_DEBUG, "Error: '%s'\n", temp.c_str());
		}
		exit(0);
		return 1;
	}
	catch(string error_message)
	{
		log_messages.printf(MSG_DEBUG,"ConvNetSeam_validator get_error_from_result([RESULT#%lu %s]) failed with error: %s\n",result.id, result.name, error_message.c_str());
	}
	catch(const exception &ex)
	{
		log_messages.printf(MSG_CRITICAL,"ConvNetSeam_validator get_error_from_result([RESULT#%lu %s]) failed with error: %s\n",result.id, result.name, ex.what());
		exit(0);
		return 1;
	}

	int retval = get_output_file_path(result, fi.path);
	//int retval = get_output_file_infos(result, files);
	//OUTPUT_FILE_INFO &fi = files[0];
	if(retval)
	{
		log_messages.printf(MSG_CRITICAL, "ConvNetSeam_validator: Failed to get output file path: %lu %s\n",result.id, result.name);
		exit(0);
		return retval;
	}

	log_messages.printf(MSG_DEBUG,"Result file path: '%s'\n", fi.path.c_str());

	ifstream infile(fi.path);

	cnn_output *res = new cnn_output();
	string line;
	getline(infile,line);
	res->cnn_config_id = line;
	getline(infile,line);
	res->video_id = line;
	stringstream ss;
	while(getline(infile,line))
	{
		ss << line << '\n';
	}
	res->obs.load(ss.str());


	
	data = (void*)res;

	log_messages.printf(MSG_DEBUG,"Successful init result.\n");
	return 0;
}

int compare_results(RESULT &r1, void *data1, RESULT const& r2, void *data2, bool& match)
{

	cnn_output *res1 = (cnn_output*)data1;
	cnn_output *res2 = (cnn_output*)data2;

	if(res1->obs.equals(res2->obs))
		match = true;
	else
		match = false;
	return 0;
}

int cleanup_result(RESULT const&, void *data)
{
	cnn_output* res = (cnn_output*)data;
	delete res;
	return 0;
}
