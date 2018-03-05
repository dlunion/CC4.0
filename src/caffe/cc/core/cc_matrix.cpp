

#include "caffe/cc/core/cc_matrix.h"
#include <highgui.h>

using namespace cv;

namespace cc{

	//从图片数据构造
	Matrix::Matrix(cbytes image_data, size_t length, int color_flags){
		init();

		//目前先默认为图像解码
		try{
			data = cv::imdecode(Mat(1, length, CV_8U, (void*)image_data), color_flags);
		}
		catch (...){}
	}

	//从Mat构造
	Matrix::Matrix(const Mat& mat, bool copy_data){
		init();

		CvMat m = mat;
		data = Mat(&m, copy_data);
	}

	//从文件构造，如果是图片文件，color_flags就是imread的参数，如果是自定义格式，就没有意义
	Matrix::Matrix(const char* file, int color_flags){
		init();

		data = cv::imread(file, color_flags);
	}

	//从Matrix构造
	Matrix::Matrix(const Matrix& other, bool copy_data){
		init();

		CvMat m = other.data;
		this->data = Mat(&m, copy_data);
		copyFrom(other);
	}

	Matrix::Matrix(){
		init();
	}

	void Matrix::init(){
	}

	Matrix& Matrix::operator = (const Matrix& other){
		this->data = other.data;
		copyFrom(other);
		return *this;
	}

	Matrix& Matrix::operator = (const Mat& other){
		this->data = other;
		return *this;
	}

	Matrix::operator Mat(){
		return this->data;
	}

	void Matrix::copyFrom(const Matrix& other){

	}
};
