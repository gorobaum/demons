#ifndef DEMONS_PROFILER_H_
#define DEMONS_PROFILER_H_

#include <ctime>
#include <iostream>
#include <string>


class Profiler {
public:
	Profiler(const std::string& msg) :
		msg_(msg) {
		t = clock();	
	}
	~Profiler() {
		t = clock()-t;
		std::cout << "Time on func " << msg_ << " is " << (((float)t)/CLOCKS_PER_SEC) << " secs\n";
	}
private:
	std::string msg_;
	clock_t t;
};

#endif