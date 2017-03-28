/***************************
*
* This program:
* 	1a. Brings in a video from database.
*	1b. Seamcarve frames.
*	2.  Trains a cnn over the images.
*	3.  Makes file saying where training images came from.
*
*
****************************/


//ConvNet
// #include <ConvNetCL.h>
// #include <ConvNetSeam.h>
#include <Seamcarver.h>
#include <ConvNetEvent.h>
#include <ConvNetCommon.h>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//MySQL
#include <mysql.h>

//Other
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <time.h>
#include <unordered_map>
#include <thread>

using namespace std;
using namespace cv;

#define mysql_query_check(conn,query) __mysql_check(conn, query, __FILE__, __LINE__)

// #define CLASSES_IN_OUT_FRAME 0
// #define CLASSES_DETAILED 1
// #define CLASSES_SUPER_DETAILED 2

// #define RANDOM_CROP -1
// #define DISTORT_DOWN 0
// #define SCALE_DOWN 1
// #define CARVE_DOWN_VTH 2
// #define CARVE_DOWN_HTV 3
// #define CARVE_DOWN_BOTH_RAW 4
// #define CARVE_DOWN_BOTH_SCALED 5

int detailLevel = CLASSES_ON_OFF_OUT; 

vector<string> class_names;
// vector<int> class_true_vals;
unordered_map<int,int> class_true_vals;
vector<int> class_true_vals_vector;

//Global variables
MYSQL *wildlife_db_conn = NULL;

//Helper functions
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

//Other Functions
void init_wildlife_database()
{
	wildlife_db_conn = mysql_init(NULL);

	//get database info from file
	string db_host, db_name, db_password, db_user;
	ifstream db_info_file("../wildlife_db_info");
	db_info_file >> db_host >> db_name >> db_user >> db_password;
	db_info_file.close();

	printf("parsed db info, host: '%s', name: '%s', user: '%s', pass: '%s'\n", db_host.c_str(), db_name.c_str(), db_user.c_str(), db_password.c_str());

	if(mysql_real_connect(wildlife_db_conn, db_host.c_str(),db_user.c_str(), db_password.c_str(), db_name.c_str(), 0, NULL, 0) == NULL)
	{
		printf("Error conneting to database: %d, '%s'\n",mysql_errno(wildlife_db_conn), mysql_error(wildlife_db_conn));
		exit(1);
	}
}

void setupDetailLevel(int detail)
{
	vector<int> classIds;
	getClasses(detail,classIds);
	char buf[50];
	for(int i = 0; i < classIds.size(); i++)
	{
		sprintf(buf,"%d",classIds[i]);
		class_names.push_back(string(buf));
		// class_true_vals.push_back(i);
		class_true_vals[classIds[i]] = i;
		class_true_vals_vector.push_back(i);
	}
}

int mapToTrueVal(vector<Event>& obs)
{
	//erase any non-relevant obs
	// printf("orig obs\n");
	// for(int i = 0; i < obs.size(); i++)
	// 	printf("%s\n", obs[i].type.c_str());
	for(int i = 0; i < obs.size(); i++)
	{
		bool found = false;
		for(int j = 0; j < class_names.size(); j++)
			if(obs[i].type == class_names[j])
			{
				found = true;
				break;
			}
		if(!found)
		{
			obs.erase(obs.begin()+i);
			i--;
		}
	}

	// printf("relevant obs\n");
	// for(int i = 0; i < obs.size(); i++)
	// 	printf("%s\n", obs[i].type.c_str());

	// //if more than 1 relevant observation is there, that's interesting
	// if(obs.size() > 1)
	// 	printf("Multiple relevant observations found\n");
	if(obs.size() == 0)
	{
		printf("No relevant observations found.\n");
		return -1;
	}

	return class_true_vals[stoi(obs[0].type)];
}

//do not use this function on multiple VideoCaptures at once. Run one throught to the end before
//going on to the next
bool getNextFrame(VideoCapture& video, Mat& frame, int& framenum, int jump = 1)
{
	// firstGot should regulate itself so it'll reset when a video runs out of frames
	static bool firstGot = false; 
	bool moreFrames = true;
	if(firstGot) // not first frame
	{
		bool val1 = true;
		for(int i = 0; i < jump; i++)
			if(!video.grab())
				val1 = false;
		bool val2 = video.retrieve(frame);
		framenum += jump;
		if(!val1 || !val2)
		{
			firstGot = false; // this means video ended
			moreFrames = false;
		}
	}
	else // first frame
	{
		bool val = video.grab();
		firstGot = true;
		if(!val)
		{
			printf("first frame val failed on grab\n");
			firstGot = false;
			moreFrames = false;
		}
		val = video.retrieve(frame);
		if(!val)
		{
			printf("first frame failed on retreive\n");
			firstGot = false;
			moreFrames = false;
		}
		framenum++;
	}
	return moreFrames;
}


/*
possible dataTypes
0x08 - unsigned byte
0x09 - signed byte
0x0B - short
0x0C - int
0x0D - float
0x0E - double

dims will be converted to big endian on output
dim[0] should be number of items to output, then x, y, z dimesions
*/
void initIDX(ofstream& out, string outName, uint8_t dataType, uint8_t numDims, vector<int32_t>& dims)
{
	if(out.is_open())
		out.close();
	out.open(outName.c_str(), ios_base::binary | ios::trunc);
	//put magic number of 0 0 dataType numdims
	out.put(0); out.put(0);
	out.put(dataType);
	out.put(numDims);


	//put all dims as Big Endian 32bit ints
	for(size_t i = 0; i < dims.size(); i++)
		for(int j = 24; j >= 0; j-=8)
			out.put(dims[i] >> j);
	// out.flush();
}

void writeMatToIDX(ofstream& out, const Mat& frame, uint8_t dataType)
{
	if(dataType != 0x08)
	{
		printf("Mats should be written to IDXs with a dataType of 0x08 (unsigned byte).\n");
		exit(0);
	}

	for(int i = 0; i < frame.rows; i++)
		for(int j = 0; j < frame.cols; j++)
		{
			const Vec3b& pix = frame.at<Vec3b>(i,j);
			out.put(pix[0]);
			out.put(pix[1]);
			out.put(pix[2]);
		}
	// out.flush();
}

bool preprocessFrame(Mat frame, Mat* dest, cv::Size cvSize, int inputSize, int frameWidth, int frameHeight, int scaleType, Seamcarver* carver)
{
	// imshow("orig",frame);
	// waitKey(0);

	// cout << frame << endl;
	// getchar();

	// printf("start preproscess\n");
	int vseams = frameWidth  - inputSize;
	int hseams = frameHeight - inputSize;
	int numSeams = frameWidth - frameHeight;
	if(scaleType == DISTORT_DOWN) // straight scale to size. Distort if necessary
	{
		resize(frame,*dest,cvSize);
	}
	else if(scaleType == RANDOM_CROP)
	{
		default_random_engine gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		uniform_int_distribution<int> disI(0,frameHeight - inputSize);
		uniform_int_distribution<int> disJ(0,frameWidth - inputSize);
		int si = disI(gen);
		int sj = disJ(gen);
		Mat temp(frame,Range(si,si+inputSize),Range(sj,sj+inputSize));
		*dest = temp;
	}
	else if(scaleType == SCALE_DOWN) // seamcarve to square. Scale to size
	{
		Mat temp;
		if(numSeams > 0) //width > height. landscape
		{
			//vertical seams, fast
			//seamcarve_v_cpu(numSeams,frame,temp);//bring us to square
			carver->carve_v(frame,numSeams,temp);
			// printf("before resize\n");
			if(temp.empty())
				printf("empty after carver\n"); 

			resize(temp, *dest,cvSize);

		}
		else // height > width. portrait
		{
			//horizontal seams, fast
			// seamcarve_hf_cpu(-numSeams, frame, temp);
			carver->carve_h(frame,numSeams,temp);
			resize(temp, *dest,cvSize);
		}
	}
	else if(scaleType == CARVE_DOWN_VTH || scaleType == CARVE_DOWN_HTV || scaleType == CARVE_DOWN_BOTH_SCALED) // seamcarve in both directions down to size. No normal scaling
	{
		// seamcarve_both_vth_cpu(vseams, hseams, frame, *dest);
		printf("The current scaleType is deprecated and no longer supported.");
		return false;
	}
	// else if(scaleType == CARVE_DOWN_HTV)
		// seamcarve_both_htv_cpu(hseams, vseams, frame, *dest);
	else if(scaleType == CARVE_DOWN_BOTH_RAW)
		// seamcarve_both_raw_cpu(vseams, hseams, frame, *dest);
		carver->carve_b(frame,vseams,hseams,*dest);
	// else if(scaleType == CARVE_DOWN_BOTH_SCALED)
		// seamcarve_both_scaled_cpu(vseams, hseams, frame, *dest);
	else
	{
		printf("Unknown scaleType %d\n", scaleType);
		return false;
	}
	// printf("end preproscess\n");
	return true;
}

int main(int argc, const char **argv)
{
	if(argc == 1)
	{
		printf("Usage: ./ConvNetSeamTrain \n");
		// printf(" -cnn=<cnn_config>          Sets CNN architecture. Required.\n");
		printf(" -size=<int>                The size the width and height of the seamcarved images should be. Required.\n");
		printf(" -outname=<name>            Sets base name of output, no extension needed. Required. Will end up outname_train_data.idx, outname_train_label.idx,\n");
		printf("                            outname_test_data.idx, outname_test_label.idx\n");
		printf(" -video=<video_id>          Picks a video to use for training. Must be in database. Can be used multiple times.\n");
		printf(" -species_id=<species_num>  Sets species to grab videos of\n");
		printf(" -max_videos=<int>          Max videos to bring in for training.\n");
		printf(" -max_time=<double>         Max number of hours of video to train on\n");
		printf(" -testPercent=<0-100>       Percent of data to use as test data.\n");
		// printf(" -carveDown_both_scaled     Frames are seamcarved in both directions at the same time based on scaled energy values.\n");
		printf(" -carveDown_both_raw        Frames are seamcarved in both directions at the same time based on raw energy values.\n");
		// printf(" -carveDown_htv             Frames are seamcarved horizontally then vertically down to size.\n");
		// printf(" -carveDown_vth             Frames are seamcarved vertically then horizontally down to size.\n");
		printf(" -scaleDown                 Frames are seamcarved to square and scaled down to size\n");
		printf(" -distortDown               Frames are scaled down to size. No seamcarving. Possible distortion.\n");
		printf(" -random_crop               A random subimage of needed size is extracted from frames\n");
		//printf(" -images=<path_to_images> Picks path_to_images for training. Can be used multiple times.\n");
		// printf(" -device=<device_num>       OpenCL device to run CNN on\n");
		printf(" -jump=<int>                How many frames to jump between using frames. If jump=10,\n");
		printf("                            it will calc on frames 0, 10, 20, etc. Defaults to 1.\n");
		//printf(" -non_expert              Sets it not to pull from expert observed\n");

		// printf(" -train_as_is.              Default\n");
		// printf(" -train_equal_prop\n");
		printf(" -detail=<int>              3 - On nest, Off nest, Out of Frame (Default). More coming soon.\n");
		// printf(" -horizontal                Adds a horizontal flipped version of every image to the training and test sets\n");
		return 0;
	}

	setbuf(stdout,NULL); //so we can see when videos start and end on one line 
	//variable declarations
	vector<int> video_ids;
	bool expert = true;
	int jump = 1;
	int species_id = -1;
	// string cnn_path = "";
	string outname = "";
	int max_videos = -1;
	int max_time = -1;
	int device = -1;
	bool train_as_is = true;
	int detail = CLASSES_ON_OFF_OUT;
	double testPercent = 0;
	int scaleType = SCALE_DOWN;
	int32_t inputSize = -1; //assumes square input
	cv::Size cvSize;
	bool horizontal = false;
	time_t starttime, totalStartTime = time(NULL), totalVideostarttime;

	//0a. Parse through command line args
	for(int i = 1; i < argc; i++)
	{
		string arg(argv[i]);
		// if(arg.find("-cnn=") != string::npos)
		// 	cnn_path = arg.substr(arg.find('=')+1);
		if(arg.find("-outname=") != string::npos)
			outname = arg.substr(arg.find('=')+1);
		else if(arg.find("-video=") != string::npos)
			video_ids.push_back(stoi(arg.substr(arg.find('=')+1)));
		else if(arg.find("-species_id=") != string::npos)
			species_id = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_videos=") != string::npos)
			max_videos = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_time=") != string::npos)
			max_time = stoi(arg.substr(arg.find('=')+1));
		// else if(arg.find("-device=") != string::npos)
		// 	device = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-jump=") != string::npos)
			jump = stoi(arg.substr(arg.find('=')+1));
		// else if(arg.find("-train_equal_prop") != string::npos)
		// 	train_as_is = false;
		// else if(arg.find("-train_as_is") != string::npos)
		// 	train_as_is = true;
		else if(arg.find("-detail=") != string::npos)
			detail = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-testPercent=") != string::npos)
			testPercent = stod(arg.substr(arg.find('=')+1));
		else if(arg.find("-random_crop") != string::npos)
			scaleType = RANDOM_CROP;
		else if(arg.find("-distortDown") != string::npos)
			scaleType = DISTORT_DOWN;
		else if(arg.find("-scaleDown") != string::npos)
			scaleType = SCALE_DOWN;
		// else if(arg.find("-carveDown_vth") != string::npos)
		// 	scaleType = CARVE_DOWN_VTH;
		// else if(arg.find("-carveDown_htv") != string::npos)
		// 	scaleType = CARVE_DOWN_HTV;
		else if(arg.find("-carveDown_both_raw") != string::npos)
			scaleType = CARVE_DOWN_BOTH_RAW;
		// else if(arg.find("-carveDown_both_scaled") != string::npos)
		// 	scaleType = CARVE_DOWN_BOTH_SCALED;
		// else if(arg.find("-horizontal") != string::npos)
		// 	horizontal = true;
		else if(arg.find("-size=") != string::npos)
			inputSize = stoi(arg.substr(arg.find('=')+1));
		else
		{
			printf("Unknown arg: '%s'\n", argv[i]);
			return 0;
		}
	}

	// if(cnn_path == "")
	// {
	// 	printf("You must supply a CNN config file.\n");
	// 	return 0;
	// }
	if(outname == "")
	{
		printf("You must supply a base output name for the output IDX files using \"-outname=somePath\".\n");
		return 0;
	}
	if(inputSize < 1)
	{
		printf("You must supply a positive size for the seamcarved image width and height using \"-size=someInt\".\n");
	}

	setupDetailLevel(detail);

	printf("Preprocessing technique = %d\n", scaleType);

	// Net net;
	// net.load(cnn_path.c_str());
	// inputSize = net.getInputWidth();
	printf("input size is %d\n", inputSize);
	cvSize = cv::Size(inputSize,inputSize);

	if(max_time != -1)//turn hours to seconds
		max_time *= 3600;

	//0b. Open Wildlife Database
	init_wildlife_database();

	printf("database inited\n");

	//1a. Find videos in Database
	//from video_2 need id, archive_filename, start_time, duration_s

	ostringstream query;
	query << "SELECT id, archive_filename, start_time, duration_s FROM video_2 WHERE";
	if(expert)
		query << " expert_finished = 'FINISHED' AND expert_obs_count > 0";
	else
		query << " crowd_status = 'VALIDATED'";
	if(species_id != -1)
		query << " AND species_id = " << species_id;
	if(video_ids.size() > 0)
	{
		query << " AND ( ";
		for(int i = 0; i < video_ids.size(); i++)
		{
			if(i > 0)
				query << " OR";
			query << " id = " << video_ids[i];
		}
		query << " )";
	}
	if(max_videos != -1)
		query << " LIMIT " << max_videos;

	query << ";";

	//make MySQL query and get results
	// printf("Running query\n'%s'\n", query.str().c_str());
	mysql_query_check(wildlife_db_conn, query.str());
	MYSQL_RES *video_2_result = mysql_store_result(wildlife_db_conn);
	printf("Query to video_2 made.\n");

	//set up stuff for multithreading
	vector<thread> thr(10);
	vector<Seamcarver> carvers(thr.size());

	MYSQL_ROW video_row;
	int current_time = 0;
	vector<string> archive_video_names;
	vector<vector<Mat> > trainingData; // should probably find some way of reserving mem for this
	vector<vector<double> > training_trueVals;
	unsigned long totalAmountData = 0;
	default_random_engine gen(time(NULL));
	totalVideostarttime = time(NULL);
	while((video_row = mysql_fetch_row(video_2_result)))
	{
		//for each video, get variables
		int video_id = atoi(video_row[0]);
		string video_path = video_row[1];
		string video_name = video_path.substr(video_path.rfind('/')+1);
		string startDateAndTime = video_row[2];
		int obs_starttime = convnet::getTime(startDateAndTime.substr(startDateAndTime.find(' ')+1)); // need to do the substring to get rid of the date form datetime
		int duration = atoi(video_row[3]);
		string toSaveString = "Id: ";
		toSaveString += video_row[0];
		toSaveString += " - ";
		toSaveString += video_name;
		archive_video_names.push_back(toSaveString);
		// cout << video_row[1] << endl << video_row[2] << endl;

		printf("Video #%d Dur: %d ", video_id, duration);
		// printf("Start Time:  %s\n", convnet::getTime(obs_starttime).c_str());

		//if we are going to go over max time, continue and see if there is a shorter video
		if(max_time != -1 && current_time + duration > max_time)
			continue;
		current_time += duration;

		//wget video (in separate thread?)
		string sys_cmd = "wget -q http://wildlife.und.edu" + video_path;
		system(sys_cmd.c_str());
		sys_cmd = "ffmpeg -loglevel \"quiet\" -i " + video_name + " -acodec libfaac -b:a 128k -vcodec mpeg4 -b:v 1200k -flags +aic+mv4 " + video_name + ".mp4";
		system(sys_cmd.c_str());
		video_name += ".mp4";

		//get observations from video and put in observations
		Observations observations;
		ostringstream expert_query;

		//old database tables
		// expert_query << "SELECT event_type, start_time, end_time FROM expert_observations"
		// 	<< " WHERE video_id = " << video_id << ";";

		//new database tables
		expert_query << "SELECT event_id, start_time, end_time FROM timed_observations WHERE video_id = " << video_id << ";";
		// printf("Query to expert_observations:\n'%s'\n", expert_query.str().c_str());
		mysql_query_check(wildlife_db_conn, expert_query.str());
		MYSQL_RES *obs_result = mysql_store_result(wildlife_db_conn);
		
		MYSQL_ROW obs_row;
		while((obs_row = mysql_fetch_row(obs_result)))
		{
			string type = obs_row[0];
			string start = obs_row[1];
			string end = obs_row[2];

			//remove date from start and end
			start = start.substr(start.find(' ')+1);
			end = end.substr(end.find(' ')+1);
			// printf("%s %s %s\n", type.c_str(),start.c_str(),end.c_str());
			observations.addEvent(type, start, end);
		}
		mysql_free_result(obs_result); // free result because now it is stored in observations

		//open video
		VideoCapture video(video_name.c_str());
		if(!video.isOpened()) // if not opening, try next video
		{
			printf("Could not open video '%s' from path '%s'\n", video_name.c_str(), video_path.c_str());
			continue;
		}

		// cout << "fourcc " << video.get(CV_CAP_PROP_FOURCC) << endl;

		int frameWidth = video.get(CV_CAP_PROP_FRAME_WIDTH);
		int frameHeight = video.get(CV_CAP_PROP_FRAME_HEIGHT);

		//used for random crop
		uniform_int_distribution<int> disI(0,frameHeight - inputSize);
		uniform_int_distribution<int> disJ(0,frameWidth - inputSize);

		//numSeams is for SCALE_DOWN and is number of seams to square (NOT input size).
		//positive numSeams means width  > height - landscape
		//negative numSeams means height > width  - portrait
		int numSeams = frameWidth - frameHeight;

		//vseams is number of vertical seams (removing cols) until we get to input size
		int vseams = video.get(CV_CAP_PROP_FRAME_WIDTH)  - inputSize;

		//hseams is number of horizontal seams (removing rows) until we get to input size
		int hseams = video.get(CV_CAP_PROP_FRAME_HEIGHT) - inputSize;

		// printf("width %d, height %d, numFrames %lf\n", frameWidth, frameHeight, video.get(CV_CAP_PROP_FRAME_COUNT));


		

		vector<Mat> frame(thr.size());
		vector<Mat> dests(thr.size());
		int framenum = 0;
		vector<Event> curEvents;
		//go through video frame by frame.
		trainingData.resize(trainingData.size() + 1);
		training_trueVals.resize(training_trueVals.size() + 1);
		time_t videostarttime = time(NULL);
		//while(getNextFrame(video, frame, framenum, jump))
		bool done_with_video = false;
		while(!done_with_video)
		{
			// starttime = time(NULL);
			int misses = 0;
			//seamcarve/scale frame and put in trainingData
			int t;
			for(t = 0; t < thr.size(); t++)
			{
				if(!getNextFrame(video,frame[t],framenum, jump))
				{
					done_with_video = true;
					break;
				}

				observations.getEvents(obs_starttime + framenum * .1, curEvents); //assuming 10 frames per second.
				// cout << "Obs frame " << framenum << endl;
				// for(auto ev : curEvents)
				// 	cout << ev.toString();
				// cout << endl;
				int trueVal = mapToTrueVal(curEvents);
				if(trueVal != -1)
				{
					// Mat tempMat;
					training_trueVals.back().push_back(trueVal);
					// trainingData.back().push_back(tempMat.clone());

					// Mat* dptr = &(trainingData.back().back()); //destination pointer
					Mat* dptr = &(dests[t]);
					// printf("%lu %lu\n", &tempMat, dptr);
					// cout << &tempMat << " " << dptr << endl;
					Seamcarver* sptr = &(carvers[t]); // seamcarver pointer
					thr[t] = thread([=] {preprocessFrame(frame[t],dptr,cvSize,inputSize,frameWidth,frameHeight,scaleType,sptr);});
					// if(scaleType == DISTORT_DOWN) // straight scale to size. Distort if necessary
					// {
					// 	resize(frame,tempMat,cvSize);
					// }
					// else if(scaleType == RANDOM_CROP)
					// {
					// 	int si = disI(gen);
					// 	int sj = disJ(gen);
					// 	Mat temp(frame,Range(si,si+inputSize),Range(sj,sj+inputSize));
					// 	tempMat = temp;
					// }
					// else if(scaleType == SCALE_DOWN) // seamcarve to square. Scale to size
					// {
					// 	Mat temp;
					// 	if(numSeams > 0) //width > height. landscape
					// 	{
					// 		//vertical seams, fast
					// 		seamcarve_vf(numSeams,frame,temp);//bring us to square
					// 		resize(temp, tempMat,cvSize);
					// 	}
					// 	else // height > width. portrait
					// 	{
					// 		//horizontal seams, fast
					// 		seamcarve_hf(-numSeams, frame, temp);
					// 		resize(temp, tempMat,cvSize);
					// 	}
					// }
					// else if(scaleType == CARVE_DOWN_VTH) // seamcarve in both directions down to size. No normal scaling
					// 	seamcarve_both_vth(vseams, hseams, frame, tempMat);
					// else if(scaleType == CARVE_DOWN_HTV)
					// 	seamcarve_both_htv(hseams, vseams, frame, tempMat);
					// else if(scaleType == CARVE_DOWN_BOTH_RAW)
					// 	seamcarve_both_raw(vseams, hseams, frame, tempMat);
					// else if(scaleType == CARVE_DOWN_BOTH_SCALED)
					// 	seamcarve_both_scaled(vseams, hseams, frame, tempMat);
					// else
					// {
					// 	printf("Unknown scaleType %d\n", scaleType);
					// 	return 0;
					// }
					// printf("TrueVal %d Frame %d\n",trueVal, framenum);
					// training_trueVals.back().push_back(trueVal);
					// trainingData.back().push_back(tempMat);
				}
				else
				{
					misses++;
					printf("miss\n");
					if(misses == 30)
					{
						printf("No observation found for at least 30 consecutive frames of video %d\n", video_id);
						done_with_video = true;
						break;
					}
					t--; //we just skip this frame with no obs and save the thread for the next frame :)
				}
			}
			for(int u = 0; u < t; u++)
			{
				thr[u].join();
				trainingData.back().push_back(dests[u].clone());
			}

			// printf("Video %s: frame %d\n\tTime Video so far: %s\n\tTime Frame: %s\n", video_name.c_str(), framenum, secondsToString(time(NULL)-videostarttime).c_str(),secondsToString(time(NULL)-starttime).c_str());
		}
		printf("- Images pulled: %lu - Time to process video: %s\n", trainingData.back().size(), convnet::secondsToString(time(NULL)-videostarttime).c_str());
		video.release();
			
		totalAmountData += trainingData.back().size();
		//rm video
		sys_cmd = "rm " + video_name;
		system(sys_cmd.c_str());
		sys_cmd = "rm " + video_name.substr(0,video_name.rfind('.'));
		system(sys_cmd.c_str());
		// printf("%s\n", sys_cmd.c_str());
	}

	printf("Total video time: %s\n", convnet::secondsToString(time(NULL) - totalVideostarttime).c_str());

	//free video_2_result
	mysql_free_result(video_2_result);

	//open up mem used from seamcarvers if you want
	for(int i = 0; i < carvers.size(); i++)
		carvers[i].destroyMem();

	if(totalAmountData == 0)
	{
		printf("No data was obtained. Exiting\n");
		return 0;
	}


	//add training and test data. get percentages as close as possible 
	//while using separate vidoes for each
	//if percent diff > 5 tell the user?
	unsigned long minSize = -1; // should be max ulong
	int minIndex = -1;
	vector<bool> isTraining(trainingData.size(),true); // true means isTraining, false means isTest

	ofstream of_train_data, of_train_label, of_test_data, of_test_label;

	if((int32_t)totalAmountData != totalAmountData)
	{
		printf("Too much data for int32_t to hold. Exiting\n");
		return -1;
	}
	


	//
	//If all of it is training
	//
	if(testPercent == 0 || trainingData.size() < 2)
	{
		// printf("putting all to training idxs\n");
		vector<int32_t> train_data_dims(4);
		train_data_dims[0] = (int32_t)totalAmountData;
		train_data_dims[1] = inputSize;
		train_data_dims[2] = inputSize;
		train_data_dims[3] = 3;
		vector<int32_t> train_label_dims(1);
		train_label_dims[0] = (int32_t)totalAmountData;
		initIDX(of_train_data, outname + string("_train_data.idx"), 0x08, 4, train_data_dims); //unsigned chars
		initIDX(of_train_label, outname + string("_train_label.idx"), 0x0B, 1, train_label_dims); //shorts
		// printf("Files should be made\n");
		//put it all for training
		for(int i = 0; i < trainingData.size(); i++)
		{
			// net.addTrainingData(trainingData[i],training_trueVals[i]);
			// printf("writing video %d\n", i);
			for(int j = 0; j < trainingData[i].size(); j++)
			{
				// printf("writing mat %d:%d\n", i,j);
				writeMatToIDX(of_train_data, trainingData[i][j], 0x08);
				short label = (short)(training_trueVals[i][j]);
				of_train_label.write(reinterpret_cast<const char *>(&label),2);
			}
			trainingData[i].resize(0); trainingData[i].shrink_to_fit();
			training_trueVals[i].resize(0); training_trueVals[i].shrink_to_fit();
		}
	}
	//
	//Else some of it is testing
	//
	else
	{
		printf("Testing data will be used.\n");
		printf("Total amount data: %lu\n", totalAmountData);
		//for accountability reasons, try to keep training and test videos from separate videos
		//if we can't get percents right, use smallest video for test
		
		testPercent *= .01; //make this a decimal again.
		double totalTestPercent = 0;
		int32_t totalTestCount = 0;
		bool noTestFound = true;
		for(int i = 0; i < trainingData.size(); i++)
		{
			double curPercent = trainingData[i].size() / (double)totalAmountData;
			if(curPercent + totalTestPercent < testPercent)
			{
				isTraining[i] = false;
				totalTestPercent += curPercent;
				totalTestCount += trainingData[i].size();
				noTestFound = false;
			}
			// if we can make this and only go over by 3% lets do it.
			else if(curPercent + totalTestPercent < testPercent + .03)
			{
				isTraining[i] = false;
				totalTestPercent += curPercent;
				totalTestCount += trainingData[i].size();
				noTestFound = false;
				break;
			}
			else
			{
				if(noTestFound && trainingData[i].size() < minSize)
				{
					minSize = trainingData[i].size();
					minIndex = i;
				}
			}
		}

		if(noTestFound)
		{
			isTraining[minIndex] = false;
			totalTestPercent += trainingData[minIndex].size() / (double)totalAmountData;
			totalTestCount += trainingData[minIndex].size();
		}

		printf("Test percent: desired: %lf, actual %lf\n", testPercent, totalTestPercent);

		//we know how much test data and how much training data, so set up IDXs
		vector<int32_t> train_data_dims(4), test_data_dims(4);

		train_data_dims[0] = (int32_t)totalAmountData - totalTestCount;
		train_data_dims[1] = inputSize;
		train_data_dims[2] = inputSize;
		train_data_dims[3] = 3;

		test_data_dims[0] = totalTestCount;
		test_data_dims[1] = inputSize;
		test_data_dims[2] = inputSize;
		test_data_dims[3] = 3;

		vector<int32_t> train_label_dims(1), test_label_dims(1);

		train_label_dims[0] = (int32_t)totalAmountData - totalTestCount;

		test_label_dims[0] = totalTestCount;

		initIDX(of_train_data, outname + string("_train_data.idx"), 0x08, 4, train_data_dims); //unsigned chars
		initIDX(of_train_label, outname + string("_train_label.idx"), 0x0B, 1, train_label_dims); //shorts
		initIDX(of_test_data, outname + string("_test_data.idx"), 0x08, 4, test_data_dims);
		initIDX(of_test_label, outname + string("_test_label.idx"), 0x0B, 1, test_label_dims);

		//now we know who is training and who is test

		//print who is who for documentation
		printf("Training videos:\n");
		for(int i = 0; i < trainingData.size(); i++)
			if(isTraining[i])
				printf("%s\n",archive_video_names[i].c_str());
		printf("Testing videos:\n");
		for(int i = 0; i < trainingData.size(); i++)
			if(!isTraining[i])
				printf("%s\n",archive_video_names[i].c_str());
		//put to mats
		for(int i = 0; i < trainingData.size(); i++)
		{
			if(isTraining[i])
			{
				// net.addTrainingData(trainingData[i],training_trueVals[i]);
				for(int j = 0; j < trainingData[i].size(); j++)
				{
					writeMatToIDX(of_train_data, trainingData[i][j], 0x08);
					short label = (short)(training_trueVals[i][j]);
					of_train_label.write(reinterpret_cast<const char *>(&label),2);
				}
			}
			else
			{
				// net.addTestData(trainingData[i],training_trueVals[i]);
				for(int j = 0; j < trainingData[i].size(); j++)
				{
					writeMatToIDX(of_test_data, trainingData[i][j], 0x08);
					short label = (short)(training_trueVals[i][j]);
					of_test_label.write(reinterpret_cast<const char *>(&label),2);
				}
			}

			trainingData[i].resize(0); trainingData[i].shrink_to_fit();
			training_trueVals[i].resize(0); training_trueVals[i].shrink_to_fit();
		}
	}
}