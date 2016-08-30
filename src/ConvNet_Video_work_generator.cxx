//MySQL
#include <mysql.h>

//boinc includes
#include "db/boinc_db.h"
#include "lib/error_numbers.h"
#include "tools/backend_lib.h"
#include "lib/parse.h"
#include "lib/util.h"
#include "svn_version.h"
#include "sched/sched_config.h"
#include "sched/sched_util.h"
#include "lib/sched_msgs.h"
#include "lib/str_util.h"

//General
#include <fstream>
#include <string>

#define REPLICATION_FACTOR 1

using namespace std;

const char* app_name = "ConvNet_Video";
const char* in_template_file = "ConvNet_in.xml";
const char* out_template_file = "ConvNet_out.xml";

char* in_template;

DB_APP app;

MYSQL *wildlife_db_conn = NULL;


double estfpops(double video_length, double cnn_size)
{
	return 1e12;
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

int make_job(string cnn_name, string video_name, int speciesID)
{

	DB_WORKUNIT wu;
	char name[256], path[256];
	char command_line[1024];
	char additional_xml[512];
	const char* infiles[2]; //CNN, video	
	int retval;


	double fpops_est = estfpops(3600, 7);

	double credit = fpops_est / (2.5 * 10e10);

	//make a unique name for the job using the speciesID, CNN name, and the video name
	sprintf(name, "S%d_%s_%s",speciesID,cnn_name.c_str(),video_name.c_str());
/*
	//Create the input file
	retval = config.download_path(name, path);
	if(retval) return retval;
	
	//Fill in job parameters
	wu.clear();
	wu.appid = app.id;
	strcpy(wu.name, name);
	wu.rsc_fpops_est = fpops_est;
	wu.rsc_fpops_bound = fpops_est * 100;
	wu.rsc_memory_bound = 2e9;
	wu.rsc_disk_bound = 2e9;
	wu.min_quorum = REPLICATION_FACTOR;
	wu.target_nresults = REPLICATION_FACTOR;
	wu.max_error_results = REPLICATION_FACTOR * 4;
	wu.max_total_results = REPLICATION_FACTOR * 8;
	wu.max_success_results = REPLICATION_FACTOR * 4;

	//Register job with BOINC
	sprintf(path, "templates/%s", out_template_file);
	//sprintf(command_line, ""); //need preferably consistent parameters
				//cnn_path, video/image_path, stride, jump
	
	sprintf(additional_xml, "<credit>%.3lf</credit>", credit);
*/
	return create_work(
		wu,
		in_template,
		path,
		config.project_path(path),
		infiles,
		0,
		config,
		command_line,
		additional_xml
	);
}

int main(int argc, char** argv)
{
	

}
