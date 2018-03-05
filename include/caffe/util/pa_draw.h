
#pragma once

#include <cv.h>

#define ORG_Center			-1000000
void paDrawString(cv::Mat& dst, const std::string& str, cv::Point org, cv::Scalar color, int fontSize = 12, bool bold = false, bool italic = false, bool underline = false);