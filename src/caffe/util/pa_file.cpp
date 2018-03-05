#include "caffe/util/pa_file.h"
#include <stdio.h>

#ifdef WIN32
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#else
#include <sys/io.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#endif

#include <stack>

#ifdef WIN32
#include <string>
#else
#include <string.h>
#endif

using namespace std;

int paGetFileSize(const char* file)
{
	FILE *fp = fopen(file, "r");
	if (!fp) return -1;
	fseek(fp, 0L, SEEK_END);
	int size = ftell(fp);
	fclose(fp);
	return size;
}

bool caseEqual(char a, char b, bool igrnoe_case){
	if (igrnoe_case){
		a = a > 'a' && a < 'z' ? a - 'a' + 'A' : a;
		b = b > 'a' && b < 'z' ? b - 'a' + 'A' : b;
	}
	return a == b;
}

bool genMatchBody(const char* str, const char* matcher, bool igrnoe_case = true){
	//   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true

	if (!matcher || !*matcher || !str || !*str) return false;

	const char* ptr_matcher = matcher;
	while (*str){
		if (*ptr_matcher == '?'){
			ptr_matcher++;
		}
		else if (*ptr_matcher == '*'){
			if (*(ptr_matcher + 1)){
				if (genMatchBody(str, ptr_matcher + 1, igrnoe_case))
					return true;
			}
			else{
				return true;
			}
		}
		else if (!caseEqual(*ptr_matcher, *str, igrnoe_case)){
			return false;
		}
		else{
			if (*ptr_matcher)
				ptr_matcher++;
			else
				return false;
		}
		str++;
	}

	while (*ptr_matcher){
		if (*ptr_matcher != '*')
			return false;
		ptr_matcher++;
	}
	return true;
}

bool genMatch(const char* str, const char* matcher, bool igrnoe_case = true){
	//   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true

	if (!matcher || !*matcher || !str || !*str) return false;

	char filter[500];
	strcpy(filter, matcher);

	vector<const char*> arr;
	char* ptr_str = filter;
	char* ptr_prev_str = ptr_str;
	while (*ptr_str){
		if (*ptr_str == ';'){
			*ptr_str = 0;
			arr.push_back(ptr_prev_str);
			ptr_prev_str = ptr_str + 1;
		}
		ptr_str++;
	}

	if (*ptr_prev_str)
		arr.push_back(ptr_prev_str);

	for (int i = 0; i < arr.size(); ++i){
		if (genMatchBody(str, arr[i], igrnoe_case))
			return true;
	}
	return false;
}

#ifdef WIN32

#if 0
int paFindFilesShort(const std::string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	struct _finddata_t fileinfo;
	long handle;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		handle = _findfirst((search_path + "*").c_str(), &fileinfo);
		if (handle != -1)
		{
			do
			{
				if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
					continue;

				if (type == PaFindFileType_File && (fileinfo.attrib & _A_SUBDIR) != _A_SUBDIR ||
					type == PaFindFileType_Directory && (fileinfo.attrib & _A_SUBDIR) == _A_SUBDIR)
				{
					if (genMatch(fileinfo.name, filter))
						out.push_back(fileinfo.name);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && (fileinfo.attrib & _A_SUBDIR) == _A_SUBDIR)
					ps.push(search_path + fileinfo.name + "\\");

			} while (!_findnext(handle, &fileinfo));
			_findclose(handle);
		}
	}

	return out.size() - nOldCount;
}

int paFindFiles(const std::string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	struct _finddata_t fileinfo;
	long handle;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		handle = _findfirst((search_path + "*").c_str(), &fileinfo);
		if (handle != -1)
		{
			do
			{
				if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
					continue;

				if (type == PaFindFileType_File && (fileinfo.attrib & _A_SUBDIR) != _A_SUBDIR ||
					type == PaFindFileType_Directory && (fileinfo.attrib & _A_SUBDIR) == _A_SUBDIR)
				{
					if (genMatch(fileinfo.name, filter))
						out.push_back(search_path + fileinfo.name);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && (fileinfo.attrib & _A_SUBDIR) == _A_SUBDIR)
					ps.push(search_path + fileinfo.name + "\\");

			} while (!_findnext(handle, &fileinfo));
			_findclose(handle);
		}
	}

	return out.size() - nOldCount;
}
#endif

int paFindFilesShort(const string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	int real_path_len = strlen(real_path);
	WIN32_FIND_DATAA find_data;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
		if (hFind != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0)
					continue;

				if (type == PaFindFileType_File && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY ||
					type == PaFindFileType_Directory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
				{
					if (PathMatchSpecA(find_data.cFileName, filter))
						out.push_back(find_data.cFileName);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
					ps.push(search_path + find_data.cFileName + "\\");

			} while (FindNextFileA(hFind, &find_data));
			FindClose(hFind);
		}
	}

	return out.size() - nOldCount;
}

int paFindFiles(const string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	WIN32_FIND_DATAA find_data;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
		if (hFind != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0)
					continue;

				if (type == PaFindFileType_File && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY ||
					type == PaFindFileType_Directory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
				{
					if (PathMatchSpecA(find_data.cFileName, filter))
						out.push_back(search_path + find_data.cFileName);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
					ps.push(search_path + find_data.cFileName + "\\");

			} while (FindNextFileA(hFind, &find_data));
			FindClose(hFind);
		}
	}

	return out.size() - nOldCount;
}


#else

int paFindFilesShort(const std::string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "/");

	struct dirent* fileinfo;
	DIR* handle;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		handle = opendir(search_path.c_str());
		if (handle != 0)
		{
			while (fileinfo = readdir(handle))
			{
				struct stat file_stat;
				if (strcmp(fileinfo->d_name, ".") == 0 || strcmp(fileinfo->d_name, "..") == 0)
					continue;

				if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
					continue;

				if (type == PaFindFileType_File && !S_ISDIR(file_stat.st_mode) ||
					type == PaFindFileType_Directory && S_ISDIR(file_stat.st_mode))
				{
					if (genMatch(fileinfo->d_name, filter))
						out.push_back(fileinfo->d_name);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && S_ISDIR(file_stat.st_mode))
					ps.push(search_path + fileinfo->d_name + "/");
			}
			closedir(handle);
		}
	}

	return out.size() - nOldCount;
}

int paFindFiles(const std::string& path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/,
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = path.length();

	if (clear_out)
		out.clear();

	strcpy(real_path, path.c_str());
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "/");

	struct dirent* fileinfo;
	DIR* handle;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while (!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		handle = opendir(search_path.c_str());
		if (handle != 0)
		{
			while (fileinfo = readdir(handle))
			{
				struct stat file_stat;
				if (strcmp(fileinfo->d_name, ".") == 0 || strcmp(fileinfo->d_name, "..") == 0)
					continue;

				if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
					continue;

				if (type == PaFindFileType_File && !S_ISDIR(file_stat.st_mode) ||
					type == PaFindFileType_Directory && S_ISDIR(file_stat.st_mode))
				{
					if (genMatch(fileinfo->d_name, filter))
						out.push_back(search_path + fileinfo->d_name);

					if (nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if (inc_sub_dirs && S_ISDIR(file_stat.st_mode)){
					if(genMatch(fileinfo->d_name, "del-*")){
						printf("ignore directory: %s\n", fileinfo->d_name);
					}else{
						ps.push(search_path + fileinfo->d_name + "/");
					}
				}
			}
			closedir(handle);
		}
	}

	return out.size() - nOldCount;
}


#endif

bool paFileExists(const char* file)
{
	FILE* f = fopen(file, "rb");
	if (!f) return false;
	fclose(f);
	return true;
}

void paFileName(const char* full_path, char* name_suffix /*= 0*/, char* name_buffer /*= 0*/, char* suffix_buffer /*= 0*/, char* dir_buffer /*= 0*/)
{
	int pos = 0;
	int pathlen = 0;

	if (name_buffer != 0 || name_suffix != 0 || name_suffix != 0 || dir_buffer != 0)
	{
		pathlen = strlen(full_path);
		if (pathlen > 0)
		{
			for (int i = pathlen - 1; i >= 0; --i)
			{
				if (full_path[i] == '\\' || full_path[i] == '/')
				{
					pos = i + 1;
					break;
				}
			}
		}
	}

	if (dir_buffer != 0)
	{
		int n = max(0, pos - 1);
		memcpy(dir_buffer, full_path, n);
		dir_buffer[n] = 0;
	}

	if (name_suffix != 0)
		strcpy(name_suffix, &full_path[pos]);

	if (name_buffer != 0)
	{
		const char* ptpos = strrchr(&full_path[pos], '.');
		if (ptpos == 0)
			strcpy(name_buffer, &full_path[pos]);
		else
		{
			memcpy(name_buffer, &full_path[pos], ptpos - &full_path[pos]);
			name_buffer[ptpos - &full_path[pos]] = 0;
		}
	}

	if (suffix_buffer != 0)
	{
		const char* ptpos = strrchr(&full_path[pos], '.');
		if (ptpos == 0)
			*suffix_buffer = 0;
		else
			strcpy(suffix_buffer, ptpos + 1);
	}
}