#include "ConvNetEvent.h"

//Vectors for each classes define
int getTime(std::string tim) // must be in format hh::mm::ss. Military time
{
	int t = 0;
	t += stoi(tim.substr(tim.rfind(':')+1)); // seconds
	t += 60 * stoi(tim.substr(tim.find(':')+1, 2)); //minutes
	t += 3600 * stoi(tim.substr(0,2)); //hours
	return t;
}

void getClasses(int classLevel, std::vector<int>& dest)
{
	dest.clear();
	if(classLevel == CLASSES_ON_OFF_OUT)
	{
		dest.push_back(PARENT_BEHAVIOR__ON_NEST);
		dest.push_back(PARENT_BEHAVIOR__OFF_NEST);
		dest.push_back(PARENT_BEHAVIOR__NOT_IN_VIDEO);
	}
	else if(classLevel == CLASSES_DETAILED)
	{
		dest.push_back(PARENT_BEHAVIOR__NOT_IN_VIDEO);
		dest.push_back(PARENT_BEHAVIOR__ON_NEST);
		dest.push_back(PARENT_BEHAVIOR__FLYING);
		dest.push_back(PARENT_BEHAVIOR__WALKING);
	}
	else 
	{
		printf("Class level not currently supported\n");
		exit(0);
	}
}

std::string getTime(int tim) //in seconds. Formats to hh::mm::ss. Military time
{
	int seconds = tim % 60;
	tim /= 60; //now tim is in minutes
	int minutes = tim % 60;
	int hours = tim / 60;

	// printf("%2d:%2d:%2d\n", hours,minutes,seconds);

	char s[3], m[3], h[3];
	if(seconds < 10)
		sprintf(s, "0%d",seconds);
	else
		sprintf(s, "%d", seconds);
	if(minutes < 10)
		sprintf(m, "0%d",minutes);
	else
		sprintf(m, "%d", minutes);
	if(hours < 10)
		sprintf(h, "0%d",hours);
	else
		sprintf(h, "%d", hours);
	s[2] = '\0'; m[2] = '\0'; h[2] = '\0';
	// printf("%s:%s:%s\n", h,m,s);
	std::string out = h;
	out += ":";
	out += m;
	out += ":";
	out += s;

	return out;
}


Event::Event()
{
	starttime = -1;
	endtime = -1;
}

bool Event::equals(const Event &other)
{
	if(type == other.type && starttime == other.starttime && endtime == other.endtime)
		return true;
	return false;
}

Event::Event(std::string type, int starttime, int endtime)
{
	this->type = type;
	this->starttime = starttime;
	this->endtime = endtime;
	if(endtime < starttime)
		this->isOvernight = true;
	else
		this->isOvernight = false;
	this->starttime_string = getTime(starttime);
	this->endtime_string = getTime(endtime);
}

Event::Event(std::string event)
{
	std::string line;
	std::stringstream ss(event);
	getline(ss,line); //just says EVENT
	getline(ss,line); 
	std::string type = line;
	getline(ss,line);
	std::string starttime = line;
	getline(ss, line);
	std::string endtime = line;
	Event(type,getTime(starttime),getTime(endtime));
}

Event::Event(const Event& other)
{
	type = other.type;
	starttime = other.starttime;
	endtime = other.starttime;
	isOvernight = other.isOvernight;
	starttime_string = other.starttime_string;
	endtime_string = other.endtime_string;
}

std::string Event::toString() const
{
	std::string out = "EVENT\n";
	out += type; out += "\n";
	out += starttime_string; out += "\n";
	out += endtime_string; out += "\n";

	return out;
}


std::string secondsToString(time_t seconds)
{
	time_t secs = seconds%60;
	time_t mins = (seconds%3600)/60;
	time_t hours = seconds/3600;
	char out[100];
	if(hours > 0)
		sprintf(out,"%ld hours, %ld mins, %ld secs",hours,mins,secs);
	else if(mins > 0)
		sprintf(out,"%ld mins, %ld secs",mins,secs);
	else
		sprintf(out,"%ld secs",secs);
	std::string outstring = out;
	return outstring;
}

bool containsEvent(std::vector<Event> events, std::string type)
{
	for(unsigned int i = 0; i < events.size(); i++)
	{
		if(events[i].type == type)
			return true;
	}
	return false;
}

Observations::Observations()
{}

Observations::Observations(std::string obs)
{
	this->load(obs);
}

bool Observations::equals(const Observations& other)
{
	if(events.size() != other.events.size())
		return false;

	std::vector<Event> e1 = events;
	std::vector<Event> e2 = other.events;

	while(e1.size() > 0)
	{
		bool found = false;
		for(unsigned int i = 0; i < e2.size(); i++)
		{
			if(e1[0].equals(e2[i]))
			{
				e1.erase(e1.begin());
				e2.erase(e2.begin()+i);
				found = true;
				break;
			}
		}
		if(!found)
			return false;
	}
	return true;
}

void Observations::load(std::string obs)
{
	std::stringstream ss(obs);
	std::string line;
	getline(ss,line);
	std::stringstream event; //should say EVENT
	event << line << '\n';
	while(getline(ss,line))
	{
		if(line == "EVENT")
		{
			events.push_back(Event(event.str()));
			event.clear();
			event << line << '\n';
		}
		else
			event << line << '\n';
	}
}

//Class Level functions (and getTime)
void Observations::addEvent(std::string type, int starttime, int endtime)
{
	events.push_back(Event(type,starttime,endtime));
}

void Observations::addEvent(std::string type, std::string starttime, std::string endtime)
{
	Event event(type, getTime(starttime), getTime(endtime));
	events.push_back(event);
	// printf("event: %s, start %s|%d, end %s|%d\n", event.type.c_str(), starttime.c_str(), event.starttime, endtime.c_str(), event.endtime);
}

void Observations::getEvents(int tim, std::vector<Event>& dest)
{
	dest.clear();
	//seconds in a day = 3600 * 24 = 86400
	tim %= 86400; //make sure we are within a valid time for a day
	for(unsigned int i = 0; i < events.size(); i++)
	{
		//check if time is within event time. if so add to dest
		if(events[i].isOvernight) 
		{
			if(events[i].starttime <= tim || tim  <= events[i].endtime)
				dest.push_back(events[i]);
		}
		else
		{
			if(events[i].starttime <= tim && tim  <= events[i].endtime)
				dest.push_back(events[i]);
		}
	}
}

void Observations::getEvents(std::string tim, std::vector<Event>& dest)
{
	getEvents(getTime(tim),dest);
}

void Observations::getAllEvents(std::vector<Event>& dest)
{
	dest.resize(0);
	for(unsigned int i = 0; i < events.size(); i++)
		dest.push_back(events[i]);
}

std::string Observations::toString() const
{
	std::stringstream ss;
	for(size_t i = 0; i < events.size(); i++)
	{
		ss << events[i].toString();
	}
	return ss.str();
}
