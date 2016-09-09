#include "config.h"
#include "util.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "validate_util.h"
#include "md5_file.h"
#include "error_numbers.h"
#include "stdint.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

//from UNDVC_COMMON
#include "parse_xml.hxx"
#include "file_io.hxx"

using namespace std;

struct cnn_output
{
	vector<string> names; // [0] should be time. classes starting at [1]
	vector<vector<double> > outputNums; //time (based on frame num)  will be [0]. classes starting at [1];
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
		log_messages.printf(MSG_DEBUG,"ConvNet_Video_validation get_error_from_result([RESULT#%d %s]) failed with error: %s\n",result.id, result.name, error_message.c_str();
	}
	catch(const exception &ex)
	{
		log_messages.printf(MSG_CRITICAL,"ConvNet_Video_validation get_error_from_result([RESULT#%d %s]) failed with error: %s\n",result.id, result.name, ex.what());
		exit(0);
		return retval;
	}

	int retval = get_output_file_path(result, fi.path);
	if(retval)
	{
		log_messages.printf(MSG_CRITICAL, "ConvNet_Video_validation: Failed to get output file path: %d %s\n",result.id, result.name);
		exit(0);
		return retval;
	}

	log_messages.printf(MSG_DEBUG,"Result file path: '%s'\n", fi.path.c_str());

	ifstream infile(fi.path);

	cnn_output *res = new cnn_output();
	string linetemp, itemtemp;

	//first get the names across the top (first will be the time)
	getline(infile,linetemp);
	stringstream ss;
	ss.str(linetemp);
	while(getline(ss, itemtemp, ','))
		res->names.push_back(itemtemp);
	
	//then get all the numbers from the rest of the rows.
	while(getline(infile,linetemp)
	{
		//now linetemp holds a comma separated line
		res->outputNums.resize(res->outputNums.size() + 1);
		ss.str(linetemp);
		while(getline(linetemp,itemtemp,','))
			res->outputNums.back().push_back(stod(itemtemp));
	}
	infile.close();

	data = (void*)res;

	log_messages.printf(MSG_DEBUG,"Successful init result.\n");
	return 0;
}

int compare_results(RESULT &r1, void *data1, RESULT &r2, void *data2, bool& match)
{
	float threshodl = 0.015;
	cnn_output *res1 = (cnn_output*)data1;
	cnn_output *res2 = (cnn_output*)data2;

	log_messages.printf(MSG_DEBUG,"Check number of classes.\n");
	if(res1->names.size() != res2->names.size())
	{
		match = false;
		log_messages.printf(MSG_CRITICAL,"ERROR, number of classes is different. %lu vs %lu\n",res1->names.size(),res2->names.size());
		return 0;
	}

	log_messages.printf(MSG_DEBUG, "Check number of frames ran.\n");
	
}
