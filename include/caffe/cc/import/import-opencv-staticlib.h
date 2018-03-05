#pragma once

#include <opencv2/core/version.hpp>
//定义宏，保证在debug模式下，导入opencv_xxxd.lib，release模式下导入opencv_xxx.lib
#ifdef _DEBUG
#   define CC_CVLIB(name) "opencv_" name CC_CVVERSION_ID "d.lib"
#   define CC_CVLIB_2(name) "opencv_" name CC_CVVERSION_ID2 "d.lib"
#   define CC_LIB(name) name "d.lib"
#else
#   define CC_CVLIB(name) "opencv_" name CC_CVVERSION_ID ".lib"
#   define CC_CVLIB_2(name) "opencv_" name CC_CVVERSION_ID2 ".lib"
#   define CC_LIB(name)   name ".lib"
#endif

#ifndef OPENCV_DLLIB
//对于静态库，必须导入如下这些库
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "winspool.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "uuid.lib")
#pragma comment(lib, "odbc32.lib")
#pragma comment(lib, "odbccp32.lib")
#pragma comment(lib, "Comctl32.lib")
#endif

//如果为3版，导入方式不同
#if CV_MAJOR_VERSION==3
//定义cv的库名称，2410版的，如果是其他版本，则修改为指定的就好了
#define CC_CVVERSION_ID       "2410"
#define CC_CVVERSION_ID2       "300"

#ifndef OPENCV_DLLIB
//导入静态库依赖
#pragma comment(lib, CC_LIB("IlmImf"))
#pragma comment(lib, CC_LIB("libjasper"))
#pragma comment(lib, CC_LIB("libjpeg"))
#pragma comment(lib, CC_LIB("libpng"))
#pragma comment(lib, CC_LIB("libtiff"))
#pragma comment(lib, CC_LIB("libwebp"))
#pragma comment(lib, "ippicvmt.lib")
#endif

#pragma comment( lib, CC_CVLIB("calib3d") )
#pragma comment( lib, CC_CVLIB_2("calib3d") )
#pragma comment( lib, CC_CVLIB("contrib") )
//#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB_2("core") )
//#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB_2("features2d") )
#pragma comment( lib, CC_CVLIB("features2d") )
#pragma comment( lib, CC_CVLIB_2("flann") )
#pragma comment( lib, CC_CVLIB("flann") )
#pragma comment( lib, CC_CVLIB("gpu") )
#pragma comment( lib, CC_CVLIB_2("highgui") )
#pragma comment( lib, CC_CVLIB("highgui") )
#pragma comment( lib, CC_CVLIB_2("imgcodecs") )
#pragma comment( lib, CC_CVLIB_2("imgproc") )
#pragma comment( lib, CC_CVLIB("imgproc") )
#pragma comment( lib, CC_CVLIB("legacy") )
#pragma comment( lib, CC_CVLIB_2("ml") )
#pragma comment( lib, CC_CVLIB("ml") )
#pragma comment( lib, CC_CVLIB("ocl") )
#pragma comment( lib, CC_CVLIB("nonfree") )
#pragma comment( lib, CC_CVLIB_2("objdetect") )
#pragma comment( lib, CC_CVLIB("objdetect") )
#pragma comment( lib, CC_CVLIB_2("photo") )
#pragma comment( lib, CC_CVLIB("photo") )
#pragma comment( lib, CC_CVLIB_2("shape") )
#pragma comment( lib, CC_CVLIB_2("stitching") )
#pragma comment( lib, CC_CVLIB("stitching") )
#pragma comment( lib, CC_CVLIB_2("superres") )
#pragma comment( lib, CC_CVLIB("superres") )
#pragma comment( lib, CC_CVLIB_2("ts") )
#pragma comment( lib, CC_CVLIB("ts") )
#pragma comment( lib, CC_CVLIB_2("video") )
#pragma comment( lib, CC_CVLIB("video") )
#pragma comment( lib, CC_CVLIB_2("videoio") )
#pragma comment( lib, CC_CVLIB_2("videostab") )
#pragma comment( lib, CC_CVLIB("videostab") )

#ifndef OPENCV_DLLIB
#pragma comment(lib, CC_LIB("zlib"))
#endif
#else
#define CC_CVVERSION_ID CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)

#ifndef OPENCV_DLLIB
//导入静态库依赖
#pragma comment(lib, CC_LIB("IlmImf"))
#pragma comment(lib, CC_LIB("libjasper"))
#pragma comment(lib, CC_LIB("libjpeg"))
#pragma comment(lib, CC_LIB("libpng"))
#pragma comment(lib, CC_LIB("libtiff"))
#endif

#pragma comment( lib, CC_CVLIB("calib3d") )
#pragma comment( lib, CC_CVLIB("contrib") )
#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB("features2d") )
#pragma comment( lib, CC_CVLIB("flann") )
#pragma comment( lib, CC_CVLIB("gpu") )
#pragma comment( lib, CC_CVLIB("highgui") )
#pragma comment( lib, CC_CVLIB("imgproc") )
#pragma comment( lib, CC_CVLIB("legacy") )
#pragma comment( lib, CC_CVLIB("ml") )
#pragma comment( lib, CC_CVLIB("ocl") )
#pragma comment( lib, CC_CVLIB("nonfree") )
#pragma comment( lib, CC_CVLIB("objdetect") )
#pragma comment( lib, CC_CVLIB("photo") )
#pragma comment( lib, CC_CVLIB("stitching") )
#pragma comment( lib, CC_CVLIB("superres") )
#pragma comment( lib, CC_CVLIB("ts") )
#pragma comment( lib, CC_CVLIB("video") )
#pragma comment( lib, CC_CVLIB("videostab") )

#ifndef OPENCV_DLLIB
#pragma comment(lib, CC_LIB("zlib"))
#endif
#endif //CV_MAJOR_VERSION
