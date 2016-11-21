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
#include <ConvNetCL.h>
#include <ConvNetSeam.h>
#include <ConvNetEvent.h>

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

	//if more than 1 relevant observation is there, that's interesting
	if(obs.size() > 1)
		printf("Multiple relevant observations found\n");
	else if(obs.size() == 0)
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

int main(int argc, const char **argv)
{
	if(argc == 1)
	{
		printf("Usage: ./ConvNetSeamTrain \n");
		printf(" -cnn=<cnn_config>          Sets CNN architecture. Required.\n");
		printf(" -outname=<name>            Sets name of output cnn. Required.\n");
		printf(" -video=<video_id>          Picks a video to use for training. Must be in database. Can be used multiple times.\n");
		printf(" -species_id=<species_num>  Sets species to grab videos of\n");
		printf(" -max_videos=<int>          Max videos to bring in for training.\n");
		printf(" -max_time=<double>         Max number of hours of video to train on\n");
		printf(" -testPercent=<0-100>       Percent of data to use as test data.");
		printf(" -carveDown_both_scaled     Frames are seamcarved in both directions at the same time based on scaled energy values.\n");
		printf(" -carveDown_both_raw        Frames are seamcarved in both directions at the same time based on raw energy values.\n");
		printf(" -carveDown_htv             Frames are seamcarved horizontally then vertically down to size.\n");
		printf(" -carveDown_vth             Frames are seamcarved vertically then horizontally down to size.\n");
		printf(" -scaleDown                 Frames are seamcarved to square and scaled down to size\n");
		printf(" -distortDown               Frames are scaled down to size. No seamcarving. Possible distortion.\n");
		printf(" -random_crop               A random subimage of needed size is extracted from frames\n");
		//printf(" -images=<path_to_images> Picks path_to_images for training. Can be used multiple times.\n");
		printf(" -device=<device_num>       OpenCL device to run CNN on\n");
		printf(" -jump=<int>                How many frames to jump between using frames. If jump=10,\n");
		printf("                            it will calc on frames 0, 10, 20, etc. Defaults to 1.\n");
		//printf(" -non_expert              Sets it not to pull from expert observed\n");

		printf(" -train_as_is.              Default\n");
		printf(" -train_equal_prop\n");
		printf(" -detail=<int>              0 is in or out of frame (default). 1 is out of frame, on nest, flying\n");
		// printf(" -horizontal                Adds a horizontal flipped version of every image to the training and test sets\n");
		return 0;
	}

	setbuf(stdout,NULL); //so we can see when videos start and end on one line 
	//variable declarations
	vector<int> video_ids;
	bool expert = true;
	int jump = 1;
	int species_id = -1;
	string cnn_path = "";
	string outname = "";
	int max_videos = -1;
	int max_time = -1;
	int device = -1;
	bool train_as_is = true;
	int detail = 0;
	double testPercent = 0;
	int scaleType = SCALE_DOWN;
	int inputSize; //assumes square input
	cv::Size cvSize;
	bool horizontal = false;
	time_t starttime, totalStartTime = time(NULL), totalVideostarttime;

	//0a. Parse through command line args
	for(int i = 1; i < argc; i++)
	{
		string arg(argv[i]);
		if(arg.find("-cnn=") != string::npos)
			cnn_path = arg.substr(arg.find('=')+1);
		else if(arg.find("-outname=") != string::npos)
			outname = arg.substr(arg.find('=')+1);
		else if(arg.find("-video=") != string::npos)
			video_ids.push_back(stoi(arg.substr(arg.find('=')+1)));
		else if(arg.find("-species_id=") != string::npos)
			species_id = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_videos=") != string::npos)
			max_videos = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_time=") != string::npos)
			max_time = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-jump=") != string::npos)
			jump = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-train_equal_prop") != string::npos)
			train_as_is = false;
		else if(arg.find("-train_as_is") != string::npos)
			train_as_is = true;
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
		else if(arg.find("-carveDown_vth") != string::npos)
			scaleType = CARVE_DOWN_VTH;
		else if(arg.find("-carveDown_htv") != string::npos)
			scaleType = CARVE_DOWN_HTV;
		else if(arg.find("-carveDown_both_raw") != string::npos)
			scaleType = CARVE_DOWN_BOTH_RAW;
		else if(arg.find("-carveDown_both_scaled") != string::npos)
			scaleType = CARVE_DOWN_BOTH_SCALED;
		else if(arg.find("-horizontal") != string::npos)
			horizontal = true;
		else
		{
			printf("Unknown arg: '%s'\n", argv[i]);
			return 0;
		}
	}

	if(cnn_path == "")
	{
		printf("You must supply a CNN config file.\n");
		return 0;
	}
	if(outname == "")
	{
		printf("You must supply a name for the output CNN.\n");
		return 0;
	}

	setupDetailLevel(detail);

	printf("Preprocessing technique = %d\n", scaleType);

	Net net;
	net.load(cnn_path.c_str());
	inputSize = net.getInputWidth();
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
	printf("Running query\n'%s'\n", query.str().c_str());
	mysql_query_check(wildlife_db_conn, query.str());
	MYSQL_RES *video_2_result = mysql_store_result(wildlife_db_conn);
	printf("Query to video_2 made.\n");

	seamcarve_setDevice(0);

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
		int obs_starttime = getTime(startDateAndTime.substr(startDateAndTime.find(' ')+1));
		int duration = atoi(video_row[3]);
		string toSaveString = "Id: ";
		toSaveString += video_row[0];
		toSaveString += " - ";
		toSaveString += video_name;
		archive_video_names.push_back(toSaveString);
		// cout << video_row[1] << endl << video_row[2] << endl;

		printf("Video #%d Dur: %d ", video_id, duration);

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



		Mat frame;
		int framenum = 0;
		vector<Event> curEvents;
		//go through video frame by frame.
		trainingData.resize(trainingData.size() + 1);
		training_trueVals.resize(training_trueVals.size() + 1);
		time_t videostarttime = time(NULL);
		while(getNextFrame(video, frame, framenum, jump))
		{
			starttime = time(NULL);
			int misses = 0;
			//seamcarve/scale frame and put in trainingData
	

			observations.getEvents(obs_starttime + framenum * .1, curEvents); //assuming 10 frames per second.
			int trueVal = mapToTrueVal(curEvents);
			if(trueVal != -1)
			{
				Mat tempMat;
				if(scaleType == DISTORT_DOWN) // straight scale to size. Distort if necessary
				{
					resize(frame,tempMat,cvSize);
				}
				else if(scaleType == RANDOM_CROP)
				{
					int si = disI(gen);
					int sj = disJ(gen);
					Mat temp(frame,Range(si,si+inputSize),Range(sj,sj+inputSize));
					tempMat = temp;
				}
				else if(scaleType == SCALE_DOWN) // seamcarve to square. Scale to size
				{
					Mat temp;
					if(numSeams > 0) //width > height. landscape
					{
						//vertical seams, fast
						seamcarve_vf(numSeams,frame,temp);//bring us to square
						resize(temp, tempMat,cvSize);
					}
					else // height > width. portrait
					{
						//horizontal seams, fast
						seamcarve_hf(-numSeams, frame, temp);
						resize(temp, tempMat,cvSize);
					}
				}
				else if(scaleType == CARVE_DOWN_VTH) // seamcarve in both directions down to size. No normal scaling
					seamcarve_both_vth(vseams, hseams, frame, tempMat);
				else if(scaleType == CARVE_DOWN_HTV)
					seamcarve_both_htv(hseams, vseams, frame, tempMat);
				else if(scaleType == CARVE_DOWN_BOTH_RAW)
					seamcarve_both_raw(vseams, hseams, frame, tempMat);
				else if(scaleType == CARVE_DOWN_BOTH_SCALED)
					seamcarve_both_scaled(vseams, hseams, frame, tempMat);
				else
				{
					printf("Unknown scaleType %d\n", scaleType);
					return 0;
				}
				// printf("TrueVal %d Frame %d\n",trueVal, framenum);
				training_trueVals.back().push_back(trueVal);
				trainingData.back().push_back(tempMat);
			}
			else
			{
				printf("No observation found for frame %d, video %d\n", framenum, video_id);
				misses++;
				if(misses == 30)
					break;
			}
			// printf("Video %s: frame %d\n\tTime Video so far: %s\n\tTime Frame: %s\n", video_name.c_str(), framenum, secondsToString(time(NULL)-videostarttime).c_str(),secondsToString(time(NULL)-starttime).c_str());
		}
		printf("- Images pulled: %lu - Time to process video: %s\n", trainingData.back().size(), secondsToString(time(NULL)-videostarttime).c_str());
		video.release();
			
		totalAmountData += trainingData.back().size();
		//rm video
		sys_cmd = "rm " + video_name;
		system(sys_cmd.c_str());
		sys_cmd = "rm " + video_name.substr(0,video_name.rfind('.'));
		system(sys_cmd.c_str());
		// printf("%s\n", sys_cmd.c_str());
	}

	printf("Total video time: %s\n", secondsToString(time(NULL) - totalVideostarttime).c_str());

	//free video_2_result
	mysql_free_result(video_2_result);

	if(scaleType >= SCALE_DOWN && scaleType <= CARVE_DOWN_BOTH_SCALED)
		seamcarve_cleanup();


	if(totalAmountData == 0)
	{
		printf("No data was obtained. Exiting\n");
		return 0;
	}

	//2. Train CNN over seamcarved images
	//initialized up top
	net.setSaveName(outname);
	net.setClassNames(class_names,class_true_vals_vector);
	if(train_as_is)
		net.setTrainingType(TRAIN_AS_IS);
	else
		net.setTrainingType(TRAIN_EQUAL_PROP);
	if(device != -1)
		net.setDevice(device);
	if(!net.finalize())
	{
		printf("Error finalizing Net: \n%s\n",net.getErrorLog().c_str());
		return 0;
	}

	printf("CNN Layer Sizes\n");
	net.printLayerDims();

	//add training and test data. get percentages as close as possible 
	//while using separate vidoes for each
	//if percent diff > 5 tell the user?
	unsigned long minSize = -1; // should be max ulong
	int minIndex = -1;
	vector<bool> isTraining(trainingData.size(),true);
	if(testPercent == 0 || trainingData.size() < 2)
	{
		//put it all for training
		for(int i = 0; i < trainingData.size(); i++)
		{
			net.addTrainingData(trainingData[i],training_trueVals[i]);
			trainingData[i].resize(0); trainingData[i].shrink_to_fit();
			training_trueVals[i].resize(0); training_trueVals[i].shrink_to_fit();
		}
	}
	else
	{
		printf("Testing data will be used.\n");
		printf("Total amount data: %lu\n", totalAmountData);
		//for accountability reasons, try to keep training and test videos from separate videos
		//if we can't get percents right, use smallest video for test
		
		testPercent *= .01; //make this a decimal again.
		double totalTest = 0;
		bool noTestFound = true;
		for(int i = 0; i < trainingData.size(); i++)
		{
			double curPercent = trainingData[i].size() / (double)totalAmountData;
			if(curPercent + totalTest < testPercent)
			{
				isTraining[i] = false;
				totalTest += curPercent;
				noTestFound = false;
			}
			// if we can make this and only go over by 3% lets do it.
			else if(curPercent + totalTest < testPercent + .03)
			{
				isTraining[i] = false;
				totalTest += curPercent;
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
			totalTest += trainingData[minIndex].size() / totalAmountData;
		}

		//now we know who is training and who is test
		for(int i = 0; i < trainingData.size(); i++)
		{
			if(isTraining[i])
				net.addTrainingData(trainingData[i],training_trueVals[i]);
			else
				net.addTestData(trainingData[i],training_trueVals[i]);

			trainingData[i].resize(0); trainingData[i].shrink_to_fit();
			training_trueVals[i].resize(0); training_trueVals[i].shrink_to_fit();
		}
	}

	printf("Training Distribution\n");
	net.printTrainingDistribution();
	if(testPercent > 0 && trainingData.size() >= 2)
		net.printTestDistribution();

	starttime = time(NULL);
	net.train(); //this will save using the save name because of net.setSaveName called above
	printf("Total training time: %s\n", secondsToString(time(NULL)-starttime).c_str());
	string short_name = outname.substr(outname.rfind('/')+1);
	string prefix = outname.substr(0,outname.rfind('/')+1);
	string info_outname = prefix + "info_"+outname;
	ofstream infoFile(info_outname);
	infoFile << "Videos used:\n";
	for(int i = 0; i < archive_video_names.size(); i++)
	{
		infoFile << archive_video_names[i];
		if(!isTraining[i])
			infoFile << " TEST_SET";
		infoFile << endl;
	}
	infoFile.close();
	printf("Total run time: %s\n", secondsToString(time(NULL) - totalStartTime).c_str());

}