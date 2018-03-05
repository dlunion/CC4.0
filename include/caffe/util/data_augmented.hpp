#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

namespace caffe {
#ifdef USE_OPENCV
	//整图增广
	//提供的图，请提供0-1范围的值的图片
	namespace GlobalAugmented{

		//随机噪声，很小的噪声，提供的图全部是float的
		void noize(Mat& img);
		//点光源
		void pointLight(Mat& img, float minLimit, float maxLimit);
		//现实生活输入的图，是不可能出现超出1的
		void limitRange(Mat& img);
		void globalLight(Mat& img);
		void augment(Mat& img, float minLimit = -1, float maxLimit = 1);
	}
#endif
}  // namespace caffe
