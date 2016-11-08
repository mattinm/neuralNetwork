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

#include "ConvNetSeam_validator.cxx"

#define mysql_query_check(conn,query) __mysql_check(conn, query, __FILE__, __LINE__)


using namespace std;

//structs and classes

// struct cnn_output
// {
// 	string cnn;
// 	string video_id;
// 	Observations obs;
// };

//Global variables
MYSQL *wildlife_db_conn = NULL;


//methods

void __mysql_check(MYSQL *conn, string query, const char* file, const int line)
{
	mysql_query(conn, query.c_str());
	if(mysql_errno(conn) != 0)
	{
		ostringstream ex_msg;
		ex_msg << "ERROR in MYSQL query: '" << query << "'. Error: " << mysql_errno(conn) << " -- '" << mysql_error(conn) << "'. Thrown on " << file << ":" << line;
		cerr << ex_msg.str() << endl;
		exit(1);
	}
}

void init_wildlife_database()
{
	wildlife_db_conn = mysql_init(NULL);

	//get database info from file
	string db_host, db_name, db_password, db_user;
	ifstream db_info_file("../wildlife_db_info");
	db_info_file >> db_host >> db_name >> db_user >> db_password;
	db_info_file.close();

	log_messages.printf(MSG_NORMAL, "parsed db info, host: '%s', name: '%s', user: '%s', pass: '%s'\n", db_host.c_str(), db_name.c_str(), db_user.c_str(), db_password.c_str());

	if(mysql_real_connect(wildlife_db_conn, db_host.c_str(),db_user.c_str(), db_password.c_str(), db_name.c_str(), 0, NULL, 0) == NULL)
	{
		log_messages.printf(MSG_CRITICAL,"Error conneting to database: %d, '%s'\n",mysql_errno(wildlife_db_conn), mysql_error(wildlife_db_conn));
		exit(1);
	}
}

int assimilate_handler(WORKUNIT& wu, vector<RESULT>& /*results*/, RESULT& canonical_result)
{
	if(wildlife_db_conn == NULL)
		init_wildlife_database();

	if(wu.error_mask > 0)
	{
		log_messages.printf(MSG_CRITICAL,"\n", "[RESULT#%d %s] assimilate_handler: WORKUNIT ERRORED OUT\n", canonical_result.id, canonical_result.name);
	}
	else if(wu.canonical_resultid == 0)
	{
		log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] assimilate_handler: error mask not set and canonical result id == 0, should never happen\n", canonical_result.id, canonical_result.name);
        exit(1);
	}
	else //this means we have a good canonical result
	{
		cnn_output *data;
		init_result(canonical_result,(void*)data);

		ostringstream query;
		
	}
}




