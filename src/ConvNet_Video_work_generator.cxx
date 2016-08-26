//MySQL
#include <mysql.h>

//boinc includes
#include "boinc_db.h"
#include "error_numbers.h"
#include "tools/backend_lib.h"
#include "parse.h"
#include "util.h"
#include "svn_version.h"
#include "sched_config.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "str_util.h"

//General
#include <fstream>
#include <string>

using namespace std;

const char* app_name = "ConvNet_Video";
const char* in_template_file = "ConvNet_in.xml";
const char* out_template_file = "ConvNet_out.xml";

MYSQL *wildlife_db_conn = NULL;


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

int make_job(string cnn_name, string video_name, int speciesID)
{

	DB_WORKUNIT work;
	char name[256], path[256];
	char command_line[1024];
	char additional_xml[512];
	const char* infiles[2]; //CNN, video	
	int retval;

	//make a unique name for the job using the speciesID, CNN name, and the video name
	sprintf(name, "S%d_%s_%s",speciesID,cnn_name.c_str(),video_name.c_str());

	//Create the input file
	retval = config.download_path(name, path);
	if(retval) return retval;


	return 1;
}

int main(int argc, char** argv)
{
	

}