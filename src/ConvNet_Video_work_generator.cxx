/*
This is the work generator for Running ConvNetVideoDriverCL_BOINC.

*/
//boinc includes
#include "boinc_db.h"
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
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#define REPLICATION_FACTOR 1
//CUSHION - maintain this many unsent results
#define CUSHION 2000 

#define mysql_query_check(conn,query) __mysql_check(conn, query, __FILE__, __LINE__)

using namespace std;
typedef unsigned int uint;

const char* app_name = "ConvNet_Video";
const char* in_template_file = "ConvNet_in.xml";
const char* out_template_file = "ConvNet_out.xml";

char* in_template;

DB_APP app;

MYSQL *wildlife_db_conn = NULL;

time_t start_time;
int seqno;

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

int make_job(vector<string>& cnns, string video_name, int speciesID, int stride, int jump)
{

	DB_WORKUNIT wu;
	char name[256], path[256];
	char command_line[1024];
	char additional_xml[512];
	vector<const char*>  infiles; //CNNs, video	
	int retval;
	string extraCNNs = "";

	infiles.push_back(video_name.c_str());
	for(uint i = 0; i < cnns.size(); i++)
		infiles.push_back(cnns[i].c_str());

	//Command line options
	//int stride = 1;
	//int jump = 1;


	double fpops_est = estfpops(3600, 7);

	double credit = fpops_est / (2.5 * 10e10);

	//make a unique name for the job using the speciesID, CNN name, and the video name
	sprintf(name, "S%d_%lu%s_%s",speciesID,cnns.size(),cnns[0].c_str(),video_name.c_str());

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

	for(uint i = 1; i < cnns.size(); i++)
	{
		extraCNNS += " cnn=";
		extraCNNS += cnns[i];
	}
	sprintf(command_line, " %s %s stride=%d jump=%d %s", cnns[0].c_str(), video_name.c_str(), stride, jump, extraCNNs.c_str());
	
	sprintf(additional_xml, "<credit>%.3lf</credit>", credit);

	return create_work(
		wu,
		in_template,
		path,
		config.project_path(path),
		infiles.data(),
		0,
		config,
		command_line,
		additional_xml
	);
}
/*
void make_jobs()
{
	int unsent_results;
	int retval;
	
	check_stop_daemons(); //checks if there is stop in place

	retval = count_unsent_results(unsent_results, 0);

	if(retval)
	{
		log_messages.printf(MSG_CRITICAL, "count_unsent_jobs() failed: %s\n", boincerror(retval));
		exit(retval);
	}
}
*/
void usage()
{
	printf("\nWork Generator for running CNNs over video using BOINC\n");
	printf("./ConvNet_Video_work_generator <flags>\n");
	printf("Work generation flags:\n");
	//printf("--app=<name>            Application name. Required.\n");
	printf("--speciesId=<int>       Will only generate jobs for the given species.\n");
	printf("--locationId=<int>      Will only generate jobs for the given location.\n");
	printf("--number_jobs=<int>     The number of jobs to generate. If <= 0 will gen for all found videos. Defaults to 100.\n");
	printf("--expertFinished=true   Will only generate jobs where expertFinished=true.\n");
	printf("-d=X                    Sets debug level to X.\n");
	printf("-v | --version          Shows version information.\n");
	printf("\nCNN flags:\n");
	printf("--cnn=<pathToCNN>       A CNN that should be used to run over video. Must be called one or more times.\n");
	printf("--stride=<int>          How large the stride over the video should be. Defaults to 1.\n");
	printf("--jump=<int>            A jump of 10 would run the CNN over every 10th frame. Defaults to 1.\n");
	printf("-h | --help             Display this usage statement.\n");
}

void main_loop(int argc, char** argv)
{	
	long unsent_results;
	int retval;
	long total_generated = 0;
	long total_errors = 0;

	int speciesId = -1;
	int locationId = -1;
	int number_jobs = 100; //jobs to generate when under the cushion

	vector<string> cnns;
	int stride = 1;
	int jump = 1;
	bool expertFinished = false;

	for(int i = 0; i < argc; i++)
	{
		string arg = string(argv[i]);
		//work generation args
		if(arg.find("--speciesId=") != string::npos)
			speciesId = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("--locationId=") != string::npos)
			locationId = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("--expertFinished=true") != string::npos)
			expertFinished = true;
		else if(arg.find("--number_jobs=") != string::npos)
			number_jobs = stoi(arg.substr(arg.find('=')+1));
		//else if(arg.find("--app=") != string::npos)
		else if(arg.find("-d=") != string::npos)
		{
			int dl = stoi(arg.substr(arg.find('=')+1));
			log_messages.set_debug_level(dl);
			if(dl == 4) g_print_queries = true;
		}
		else if(arg.find("-v") != string::npos || arg.find("--version") != string::npos)
		{
			printf("%s\n",SVN_VERSION);
		}
		//program args
		else if(arg.find("--cnn=") != string::npos)
			cnns.push_back(arg.substr(arg.find('=')+1));
		else if(arg.find("--stride=") != string::npos)
			stride = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("--jump=") != string::npos)
			jump = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("--help") != string::npos || arg.find("-h") != string::npos);
		{
			usage();
			exit(0);
		}
	}

	init_wildlife_database();	
	
	check_stop_daemons();

	retval = count_unsent_results(unsent_results, app.id);
	if(retval)
	{
		log_messages.printf(MSG_CRITICAL,"count_unsent_jobs() failed: %s\n", boincerror(retval));
		exit(retval);
	}
	log_messages.printf(MSG_DEBUG,"%ld results are available with a cushion of %d\n", unsent_results, CUSHION);

	ostringstream unclassified_video_query, finished_expert_query;
	finished_expert_query << "SELECT DISTINCT id, watermarked_filename, duration_s, species_id, location_id, size, md5_hash"
		<< " FROM video_2"
		<< " WHERE processing_status != 'UNWATERMARKED'"
		<< " AND processing_status != 'WATERMARKING'"
		<< " AND md5_hash IS NOT NULL"
		<< " AND size IS NOT NULL"
		<< " AND expert_finished = 'FINISHED'";
	
	unclassified_video_query << "SELECT id, watermarked_filename, duration_s, species_id, location_id, size, md5_hash"
		<< " FROM video_2"
		<< " WHERE processing_status != 'UNWATERMARKED'"
		<< " AND processing_status != 'WATERMARKING'"
		<< " AND md5_hash IS NOT NULL"
		<< " AND size IS NOT NULL";
	
	//If species specified, limit it to that species, else do all species
	if(speciesId > 0)
	{
		finished_expert_query << " AND species_id = " << speciesId;
		unclassified_video_query << " AND species_id = " << speciesId;
	}
	
	//if location specified, limit it to location, else do all locations
	if(locationId > 0)
	{
		finished_expert_query << " AND location_id = " << locationId;
		unclassified_video_query << " AND location_id = " << locationId;
	}

	if(number_jobs > 0)
	{
		finished_expert_query << " LIMIT " << number_jobs;
		unclassified_video_query << " LIMIT " << number_jobs;
	}

	if(expertFinished)
		mysql_query_check(wildlife_db_conn, finished_expert_query.str());
	else
		mysql_query_check(wildlife_db_conn, unclassified_video_query.str());

	MYSQL_RES *video_result = mysql_store_result(wildlife_db_conn);
	printf("query made...\n");

	MYSQL_ROW video_row;
	while((video_row = mysql_fetch_row(video_result)))
	{
		int video_id = atoi(video_row[0]);
		string video_address = video_row[1];
		video_address += ".mp4";
		double duration_s = atof(video_row[2]);
		int species_id = atoi(video_row[3]);
		int location_id = atoi(video_row[4]);
		int filesize = atoi(video_row[5]);
		string md5_hash = video_row[6];
		
		printf("generating work unit\n");
		int job_id = make_job(cnns, video_address, species_id, stride, jump);
		if(job_id == -1)
			total_errors++;
		else
			total_generated++;
		printf("Generated %ld workunits.\nGenerated %ld errored workunits.\n\n",total_generated, total_errors);
	}

	printf("Done getting results\n");
	mysql_free_result(video_result);

	log_messages.printf(MSG_DEBUG, "Workunits generated: %ld. Errors: %ld\n",total_generated, total_errors);
}

int main(int argc, char** argv)
{
	
	int retval;
	//processing projects config file
	retval = config.parse_file();
	if(retval)
	{
		log_messages.printf(MSG_CRITICAL, "Can't parse config.xml: %s\n", boincerror(retval));
		exit(1);
	}

	//opening connection to boinc db
	retval = boinc_db.open(config.db_name, config.db_host, config.db_user, config.db_passwd);
	if(retval)
	{
		log_messages.printf(MSG_CRITICAL,"Can't open BOINC DB\n");
		exit(0);
	}
	
	printf("App name: %s\n",app_name);
	char  buf[500];
	sprintf(buf, "where name='%s'",app_name);
	if(app.lookup(buf))
	{
		log_messages.printf(MSG_CRITICAL, "can't find app %s\n", app_name);
		exit(1);
	}
	
	start_time = time(0);
	seqno = 0;
	
	log_messages.printf(MSG_NORMAL,"Starting\n");

	main_loop(argc, argv);
}
