
#ifndef _____ConvNetEvent__
#define _____ConvNetEvent__

#include <string>
#include <vector>
#include <sstream>

#define CLASSES_IN_OUT_FRAME 0
#define CLASSES_DETAILED 1
#define CLASSES_SUPER_DETAILED 2
#define CLASSES_ON_OFF_OUT 3

#define DISTORT_DOWN 0
#define SCALE_DOWN 1
#define CARVE_DOWN_VTH 2
#define CARVE_DOWN_HTV 3
#define CARVE_DOWN_BOTH_RAW 4
#define CARVE_DOWN_BOTH_SCALED 5
#define RANDOM_CROP 6

#define PARENT_BEHAVIOR__NOT_IN_VIDEO 4
#define PARENT_BEHAVIOR__ON_NEST 41
#define PARENT_BEHAVIOR__OFF_NEST 42
#define PARENT_BEHAVIOR__FLYING 6
#define PARENT_BEHAVIOR__WALKING 7
#define PARENT_BEHAVIOR__STANDING 5
#define PARENT_BEHAVIOR__SITTING 8


//Vectors for each classes define
int getTime(std::string tim); // must be in format hh::mm::ss. Military time


void getClasses(int classLevel, std::vector<int>& dest);


std::string getTime(int tim); //in seconds. Formats to hh::mm::ss. Military time


//class definitions
struct Event
{
	std::string type; // this should be the id from observation_types table
	int starttime;
	int endtime;
	std::string starttime_string;
	std::string endtime_string;
	bool isOvernight; //this means the starttime is before midnight and endtime is after

	Event(std::string type, int starttime, int endtime);
	Event(std::string event);
	Event(const Event& other);
	Event();
	std::string toString() const;
	bool equals(const Event& other);
};

class Observations
{
	std::vector<Event> events;

public:
	Observations();
	Observations(std::string obs);
	void addEvent(std::string type, std::string starttime, std::string endtime);
	void addEvent(std::string type, int starttime, int endtime);
	void getEvents(std::string tim, std::vector<Event>& dest);
	void getEvents(int tim, std::vector<Event>& dest);
	void getAllEvents(std::vector<Event>& dest);
	std::string toString() const;
	void load(std::string obs);
	bool equals(const Observations& other);
};

std::string secondsToString(time_t seconds);

bool containsEvent(std::vector<Event> events, std::string type);


#endif /* defined _____ConvNetEvent__ */