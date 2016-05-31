#include <iostream>
#include <thread>

using namespace std;

void test1()
{
	cout << "test1" << endl;
}

void test2(int i)
{
	cout << "test2 " << i << endl;
}

int main(void)
{
	thread t(test1);
	thread t2(test2,1);

	t.join();
	t2.join();
}