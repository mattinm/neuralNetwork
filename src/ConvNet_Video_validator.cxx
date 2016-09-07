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
	}
}
