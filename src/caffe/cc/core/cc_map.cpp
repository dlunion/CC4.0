
#ifndef CC_MATRIX_H
#define CC_MATRIX_H

#include <cv.h>

using cv::Mat;

namespace cc{

	typedef const char* cbytes;

	class Matrix{

	public:
		//凭空构造
		Matrix();

		//从数据构造
		Matrix(cbytes image_data, size_t length, int color_flags = 1);
		
		//从Mat构造
		Matrix(const Mat& mat, bool copy_data = false);

		//从文件构造，如果是图片文件，color_flags就是imread的参数，如果是自定义格式，就没有意义
		Matrix(const char* file, int color_flags = 1);

		//从Matrix构造
		Matrix(const Matrix& other, bool copy_data = false);

		operator Mat();
		Matrix& operator = (const Matrix& other);
		Matrix& operator = (const Mat& other);
		const Mat mat() { return this->data; }
		Mat mutable_mat() { return this->data; }

	private:
		void init();


	private:
		Mat data;
		void copyFrom(const Matrix& other);
	};
};
#endif //CC_MATRIX_H