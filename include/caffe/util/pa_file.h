#pragma once

#include <crtdefs.h>
#include <vector>
#include <string>

typedef std::vector<std::string> PaVfiles;
typedef unsigned __int64 ui64;

#define PAFileBlockPart			(12 * 1024 * 1024)

enum PaFindFileType
{
	PaFindFileType_File,
	PaFindFileType_Directory
};

/*
    function:     paFileExists
	              文件是否存在

        file:     文件路径
      return:     存在返回true，不存在返回false
*/
bool paFileExists(const char* file);


/*
    function:     paWriteToFile
	              写出数据到文件

        file:     需要写出的位置
        data:     需要写出的数据
         len:     需要写出的数据长度
      return:     写出成功返回true，失败为false
*/
bool paWriteToFile(const char* file, const void* data, const size_t len);


/*
    function:     paGetFileSize
	              获取文件尺寸

        file:     文件路径
      return:     返回文件尺寸
*/
int paGetFileSize(const char* file);


/*
    function:     paGetFileSize64
	              获取文件尺寸，64位表示

        file:     文件路径
      return:     返回文件尺寸
*/
ui64 paGetFileSize64(const char* file);

void freeReadFile(unsigned char** pptr);

/*
    function:     paReadFile
	              读取文件数据，不使用的时候请使用delete释放返回值。

        file:     文件路径
out_of_file_size: 读取的数据长度，不需要这个返回值请置0
      return:     返回读取到的数据指针，在内部使用new分配的。
*/
unsigned char* paReadFile(const char* file, size_t* out_of_file_size = 0);


/*
    function:     paReadAt
	              读取文件数据到指定的缓冲区内存

        file:     文件路径
      buffer:     数据存储缓冲区
len_of_buffer:    缓冲区的大小
 len_of_read:     有多少数据被读取并写入到buffer中
      return:     如果读取成功则返回true，失败返回false
*/
bool paReadAt(const char* file, void* buffer, size_t len_of_buffer, size_t* len_of_read = 0);


/*
    function:     paFindFiles
	              寻找文件

        path:     文件路径，路径后面允许是\或者是/,或者没有
         out:     找到的文件全路径输出参数，执行这个函数的时候会清楚out中的数据
      filter:     过滤器，比如搜索jpg，则是*.jpg
inc_sub_dirs:     是否包含子目录，指示是否需要搜索子目录下的所有文件。
   clear_out:     执行前是否先清除out中已有的数据，如果不需要则寻找得到的东西将继续加入到out
      return:     返回找到的文件数量
*/
int paFindFiles(const char* path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true, 
	PaFindFileType type = PaFindFileType_File, unsigned int nFilePerDir = 0);


/*
    function:     paFindFiles
	              寻找文件

        path:     文件路径，路径后面允许是\或者是/,或者没有
         out:     找到的文件全路径输出参数，执行这个函数的时候会清楚out中的数据
      filter:     过滤器，比如搜索jpg，则是*.jpg
inc_sub_dirs:     是否包含子目录，指示是否需要搜索子目录下的所有文件。
   clear_out:     执行前是否先清除out中已有的数据，如果不需要则寻找得到的东西将继续加入到out
      return:     返回找到的文件数量
*/
int paFindFilesShort(const char* path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true, 
	PaFindFileType type = PaFindFileType_File, unsigned int nFilePerDir = 0);


/*
    function:     paCompareFile
	              比较两个文件是否一样。

       file1:     文件1
       file2:     文件2
      return:     数据完全一致的时候返回true，否则返回false
*/
bool paCompareFile(const char* file1, const char* file2);


/*
    function:     paCompareFileBig
	              比较两个文件是否一样。用于大文件比较

       file1:     文件1
       file2:     文件2
      return:     数据完全一致的时候返回true，否则返回false
*/
bool paCompareFileBig(const char* file1, const char* file2, size_t cacheSize = PAFileBlockPart);


/*
    function:     paFileName
	              获取路径的文件名，后缀，文件名+后缀，后面3个参数为0表示不取。

   full_path:     文件的全路径，比如是：“c:/123.abc.txt”
 name_suffix:     文件名和后缀，结果会保存到这里面去，只是需要提供足够大的缓冲区即可, 这里将填入：“123.abc.txt”
 name_buffer:     文件名缓冲区，这里将填入：“123.abc”
suffix_buffer:    后缀缓冲区，这里将填入：“txt”
  dir_buffer:     文件目录缓冲区，这里将填入：“c:”
*/
void paFileName(const char* full_path, char* name_suffix = 0, char* name_buffer = 0, char* suffix_buffer = 0, char* dir_buffer = 0);


//dir后面没有带/
void paGetModulePath(char* path = 0, char* dir = 0, char* name_suffix = 0, char* name = 0);


/*
    function:     paChangePathName
	              修改路径，把路径看作3块组成，目录+文件名+后缀名，任意更换其中的部分。返回full_path指针

   full_path:     文件的全路径
	     dir:     需要修改的目录，目录后面可以带"/"或者"\"或者不带.为0时保留full_path中的目录
        name:     文件名，为0时保留full_path中的文件名
      suffix:     后缀，为0时保留full_path中的后缀，为空时，文件名和后缀之间没有"."
*/
char* paChangePathName(char* full_path, const char* dir = 0, const char* name = 0, const char* suffix = 0);


//////////////////////////////////////////////////////////////////////////
//写ini文件
bool paWriteIni(const char* app, const char* key, const char* fileName, const char* fmtValue, ...);


//////////////////////////////////////////////////////////////////////////
//读ini文件
bool paReadIni(const char* app, const char* key, const char* fileName, std::string& value);


//////////////////////////////////////////////////////////////////////////
//创建多级目录
bool paCreateDirectoryx(const char* dir);