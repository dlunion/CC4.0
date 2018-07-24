#include "caffe/util/pa_file.h"
#include <stdio.h>
#include <windows.h>
#include <stack>
#include <string>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

using namespace std;

bool paWriteToFile(const char* file, const void* data, const size_t len)
{
	FILE* fn = fopen(file, "wb");
	bool ret_value = false;

	if (fn != 0)
	{
		fwrite(data, 1, len, fn);
		fclose(fn);
		ret_value = true;
	}

	return ret_value;
}

int paGetFileSize(const char* file)
{
	WIN32_FIND_DATAA find_data;
	HANDLE hFind = FindFirstFileA(file, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	FindClose(hFind);
	return find_data.nFileSizeLow;
}

ui64 paGetFileSize64(const char* file)
{
	WIN32_FIND_DATAA find_data;
	HANDLE hFind = FindFirstFileA(file, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	FindClose(hFind);
	return (ui64)find_data.nFileSizeLow | ((ui64)find_data.nFileSizeHigh << 32);
}

bool paReadAt(const char* file, void* buffer, size_t len_of_buffer, size_t* len_of_read /*= 0*/)
{
	if(len_of_read != 0)
		*len_of_read = 0;

	if(buffer == 0 || file == 0 || len_of_buffer == 0)
		return false;

	size_t sizeOfFile = paGetFileSize(file);
	size_t sizeOfRead = min(sizeOfFile, len_of_buffer);

	if(sizeOfRead == 0)
		return false;

	FILE* f = fopen(file, "rb");
	if(f == 0)
		return false;

	size_t alreadyRead = fread(buffer, 1, sizeOfRead, f);
	if(sizeOfRead != alreadyRead)
	{
		fclose(f);
		return false;
	}

	if(len_of_read != 0)
		*len_of_read = alreadyRead;

	fclose(f);
	return true;
}

void freeReadFile(unsigned char** pptr){
	if (pptr){
		unsigned char* ptr = *pptr;
		if (ptr)
			delete[] ptr;
		*pptr = 0;
	}
}

unsigned char* paReadFile(const char* file, size_t* out_of_file_size /*= 0*/)
{
	if (out_of_file_size != 0)
		*out_of_file_size = 0;

	size_t size = paGetFileSize(file);
	if (size == 0)
		return 0;

	if (out_of_file_size != 0)
		*out_of_file_size = size;

	FILE* fn = fopen(file, "rb");
	unsigned char* data = 0;

	if (fn != 0)
	{
		data = new unsigned char[size];
		if(data == 0)
		{
			fclose(fn);
			return 0;
		}

		size = fread(data, 1, size, fn);
		fclose(fn);
	}

	return data;
}

int paFindFilesShort(const char* path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/, 
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = strlen(path);

	if(clear_out)
		out.clear();

	strcpy(real_path, path);
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	int real_path_len = strlen(real_path);
	WIN32_FIND_DATAA find_data;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while(!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
		if(hFind != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0)
					continue;

				if (type == PaFindFileType_File && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY ||
					type == PaFindFileType_Directory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
				{
					if (PathMatchSpecA(find_data.cFileName, filter))
					{
						if(search_path.size() == real_path_len)
							out.push_back(find_data.cFileName);
						else
							out.push_back(string(&search_path[real_path_len]) + find_data.cFileName);
					}

					if(nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if(inc_sub_dirs && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
					ps.push(search_path + find_data.cFileName + "\\");

			}while(FindNextFileA(hFind, &find_data));
			FindClose(hFind);
		}
	}

	return out.size() - nOldCount;
}

int paFindFiles(const char* path, PaVfiles& out, const char* filter /*= "*"*/, bool inc_sub_dirs /*= true*/, bool clear_out /*= true*/, 
	PaFindFileType type /*= HpFindFileType_File*/, unsigned int nFilePerDir /*= 0*/)
{
	char real_path[260];
	size_t length = strlen(path);
	
	if(clear_out)
		out.clear();
	
	strcpy(real_path, path);
	if (real_path[length - 1] != '\\' && real_path[length - 1] != '/')
		strcat(real_path, "\\");

	WIN32_FIND_DATAA find_data;
	stack<string> ps;
	size_t nOldCount = out.size();
	ps.push(real_path);

	while(!ps.empty())
	{
		unsigned int nAlreadyCount = 0;
		string search_path = ps.top();
		ps.pop();

		HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
		if(hFind != INVALID_HANDLE_VALUE)
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

					if(nFilePerDir > 0 && ++nAlreadyCount == nFilePerDir) break;
				}

				if(inc_sub_dirs && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
					ps.push(search_path + find_data.cFileName + "\\");

			}while(FindNextFileA(hFind, &find_data));
			FindClose(hFind);
		}
	}

	return out.size() - nOldCount;
}

bool paCompareFile(const char* file1, const char* file2)
{
	size_t size_f1 = paGetFileSize(file1);
	size_t size_f2 = paGetFileSize(file2);

	if (size_f1 != size_f2)
		return false;

	if (size_f1 == 0)
		return true;

	unsigned char* fd1 = paReadFile(file1);
	unsigned char* fd2 = paReadFile(file2);

	if (fd1 == fd2)
		return false;

	if (fd1 == 0 || fd2 == 0)
	{
		if(fd1 != 0) delete fd1;
		if(fd2 != 0) delete fd2;
		return false;
	}

	bool ret_value = (memcmp(fd1, fd2, size_f1) == 0);
	delete fd1;
	delete fd2;
	return ret_value;
}

bool paCompareFileBig(const char* file1, const char* file2, size_t cacheSize /*= PAFileBlockPart*/)
{
	ui64 size_f1 = paGetFileSize64(file1);
	ui64 size_f2 = paGetFileSize64(file2);

	if (size_f1 != size_f2)
		return false;

	if (size_f1 == 0)
		return true;

	if(size_f1 < cacheSize)	//对小于12MB的文件用内存比较
		return paCompareFile(file1, file2);

	FILE* f1 = fopen(file1, "rb");
	FILE* f2 = fopen(file2, "rb");

	if(f1 == 0 || f2 == 0)
	{
		if(f1 == 0) fclose(f1);
		if(f2 == 0) fclose(f2);
		return false;
	}

	unsigned char* buf1 = (unsigned char*)malloc(cacheSize);
	unsigned char* buf2 = (unsigned char*)malloc(cacheSize);

	bool ret = false;
	size_t cbRead1 = fread(buf1, 1, cacheSize, f1);
	size_t cbRead2 = fread(buf2, 1, cacheSize, f2);

	while(!feof(f1))
	{
		if(cbRead1 != cbRead2) goto notMatched;
		if(cbRead1 == 0) break;
		if(memcmp(buf1, buf2, cbRead1) != 0) goto notMatched;

		cbRead1 = fread(buf1, 1, cacheSize, f1);
		cbRead2 = fread(buf2, 1, cacheSize, f2);
	};

	if(cbRead1 > 0 && (cbRead1 != cbRead2 || memcmp(buf1, buf2, cbRead1) != 0)) goto notMatched;
	ret = true;

notMatched:
	if(f1 == 0) fclose(f1);
	if(f2 == 0) fclose(f2);
	if(buf1 != 0) free(buf1);
	if(buf2 != 0) free(buf2);
	return ret;
}

bool paFileExists(const char* file)
{
	return PathFileExistsA(file);
}

void paFileName(const char* full_path, char* name_suffix /*= 0*/, char* name_buffer /*= 0*/, char* suffix_buffer /*= 0*/, char* dir_buffer /*= 0*/)
{
	int pos = 0;
	int pathlen = 0;

	if(name_buffer != 0 || name_suffix != 0 || name_suffix != 0 || dir_buffer != 0)
	{
		pathlen = strlen(full_path);
		if(pathlen > 0)
		{
			for (int i = pathlen - 1; i >= 0; --i)
			{
				if(full_path[i] == '\\' || full_path[i] == '/')
				{
					pos = i + 1;
					break;
				}
			}
		}
	}

	if(dir_buffer != 0)
	{
		int n = max(0, pos - 1);
		memcpy(dir_buffer, full_path, n);
		dir_buffer[n] = 0;
	}

	if (name_suffix != 0)
		strcpy(name_suffix, &full_path[pos]);
	
	if(name_buffer != 0)
	{
		const char* ptpos = strrchr(&full_path[pos], '.');
		if(ptpos == 0)
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
		if(ptpos == 0)
			*suffix_buffer = 0;
		else
			strcpy(suffix_buffer, ptpos + 1);
	}
}

void paGetModulePath(char* path /*= 0*/, char* dir /*= 0*/, char* name_suffix /*= 0*/, char* name /*= 0*/)
{
	char p[260] = {0};
	GetModuleFileNameA(0, p, sizeof(p));
	paFileName(p, name_suffix, name, 0, dir);
	if(path != 0) strcpy(path, p);
}

char* paChangePathName(char* full_path, const char* dir /*= 0*/, const char* name /*= 0*/, const char* suffix /*= 0*/)
{
	if(dir == 0 && name == 0 && suffix == 0)
		return full_path;

	char ffdir[260] = {0};
	char ffname[260] = {0};
	char ffsufix[260] = {0};
	paFileName(full_path, 0, name == 0 ? ffname : 0, suffix == 0 ? ffsufix : 0, dir == 0 ? ffdir : 0);
	
	const char* dddir	= dir		== 0 ? ffdir	: dir;
	const char* ddname	= name		== 0 ? ffname	: name;
	const char* ddsufix = suffix	== 0 ? ffsufix	: suffix;

	const char* adaPlus = 0;
	int dirLen = strlen(dddir);
	if(dirLen == 0 || dirLen > 0 && (dddir[dirLen - 1] == '\\' || dddir[dirLen - 1] == '/'))
		adaPlus = "%s%s";
	else
		adaPlus = "%s\\%s";
	
	if(strcmp(ddsufix, "") == 0)
	{
		if(dirLen == 0)
			strcpy(full_path, ddname);
		else
		{
			if(dddir[dirLen - 1] == '\\' || dddir[dirLen - 1] == '/')
				sprintf(full_path, "%s%s", dddir, ddname);
			else
				sprintf(full_path, "%s\\%s", dddir, ddname);
		}
	}
	else
	{
		if(dirLen == 0)
			sprintf(full_path, "%s.%s", ddname, ddsufix);
		else
		{
			if(dddir[dirLen - 1] == '\\' || dddir[dirLen - 1] == '/')
				sprintf(full_path, "%s%s.%s", dddir, ddname, ddsufix);
			else
				sprintf(full_path, "%s\\%s.%s", dddir, ddname, ddsufix);
		}
	}
	return full_path;
}

//////////////////////////////////////////////////////////////////////////
//写ini文件
//如果没有指定完整路径名，则windows会在windows目录查找文件。如果文件没有找到，则函数会创建它
bool paWriteIni(const char* app, const char* key, const char* fileName, const char* fmtValue, ...)
{
	char buffer[1 << 16];
	va_list vl;
	va_start(vl, fmtValue);
	vsprintf(buffer, fmtValue, vl);
	return WritePrivateProfileStringA(app, key, buffer, fileName);
}


//////////////////////////////////////////////////////////////////////////
//读ini文件
bool paReadIni(const char* app, const char* key, const char* fileName, std::string& value)
{
	value.reserve(1024);
	int len = GetPrivateProfileStringA(app, key, 0, (char*)value.c_str(), 1024, fileName);
	value = value.data();
	return len > 0;
}

//////////////////////////////////////////////////////////////////////////
//创建多级目录
bool paCreateDirectoryx(const char* dir)
{
	if(paFileExists(dir)) return true;

#define IsSplitChar(c)     (c == '/' || c == '\\')

	int i = 0;
	int prev = 0;
	if(IsSplitChar(dir[0]))
	{
		prev++;
		i++;
	}

	char itDir[260] = {0};
	for (; dir[i] != 0; ++i)
	{
		if (IsSplitChar(dir[i]))
		{
			int len = i - prev;
			if(len > 0)
			{
				strncat(itDir, &dir[prev], len);
				if(!paFileExists(itDir) && !CreateDirectoryA(itDir, 0))
					return false;

				strcat(itDir, "/");
			}
			prev = i + 1;
		}
	}

	int len = i - prev;
	if(len > 0)
	{
		strncat(itDir, &dir[prev], len);
		if(!paFileExists(itDir) && !CreateDirectoryA(itDir, 0))
			return false;
	}
	return true;
}