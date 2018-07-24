#pragma once


#include <string>
#include <vector>
#include <cv.h>
using namespace std;

#define PART_TYPE_DEPLOY     0
#define PART_TYPE_CAFFEMODEL 1
#define PART_TYPE_LABELS     2
#define PART_TYPE_NAMES     3

class CCAPI Package{
public:
	Package() :f(0){}

	bool open(const char* savefile);
	void push(const void* data, int len, int type);
	void close();

	virtual ~Package();

private:
	FILE* f;
};

class CCAPI PackageDecode{
public:
	PackageDecode();
	virtual ~PackageDecode();
	bool decode(const char* dat);
	size_t size(int index);
	uchar* data(int index);
	int type(int index);
	int count();
	int find(int type);

private:
	vector<size_t> sizes;
	vector<uchar*> deploy_data;
	vector<int> types;
};

CCAPI bool CCCALL crypt_model(const char* dir = ".");