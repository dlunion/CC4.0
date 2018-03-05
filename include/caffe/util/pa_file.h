#pragma once

#include <vector>
#include <string>

typedef std::vector<std::string> PaVfiles;

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
    function:     paGetFileSize
	              获取文件尺寸

        file:     文件路径
      return:     返回文件尺寸
*/
int paGetFileSize(const char* file);


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
int paFindFiles(const std::string& path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true, 
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
int paFindFilesShort(const std::string& path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true,
	PaFindFileType type = PaFindFileType_File, unsigned int nFilePerDir = 0);


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
