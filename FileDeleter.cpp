#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		printf("Need 1 argument. Folder to delete all items with numbers in the names from.");
		return 0;
	}

	string folder = argv[1];
	string outStart = "rm ";
	char buf[5];
	if(folder.rfind("/") != folder.length()-1)
	{
		folder.append(1,'/');
	}
	outStart.append(folder);
	for(int i=0; i < 10000; i++)
	{
		string out = outStart;
		out.append(1,'*');

		//get three digit num
		if(i < 10)
			out.append(3,'0');
		else if(i < 100)
			out.append(2,'0');
		else if(i < 1000)
			out.append(1,'0');
		sprintf(buf,"%d",i);
		out.append(buf);
		out.append(1,'*');
		system(out.c_str());

	}
	for(int i=0; i < 1000; i++)
	{
		string out = outStart;
		out.append(1,'*');

		//get three digit num
		if(i < 10)
			out.append(2,'0');
		else if(i < 100)
			out.append(1,'0');
		sprintf(buf,"%d",i);
		out.append(buf);
		out.append(1,'*');
		system(out.c_str());

	}
	for(int i=0; i < 100; i++)
	{
		string out = outStart;
		out.append(1,'*');

		//get two digit num
		if(i < 10)
			out.append(1,'0');

		sprintf(buf,"%d",i);
		out.append(buf);
		out.append(1,'*');
		system(out.c_str());

	}
	for(int i=0; i < 10; i++)
	{
		string out = outStart;
		out.append(1,'*');

		sprintf(buf,"%d",i);
		out.append(buf);
		out.append(1,'*');
		system(out.c_str());

	}


}