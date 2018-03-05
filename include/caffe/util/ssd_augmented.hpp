
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace caffe {
#ifdef USE_OPENCV
	
	using namespace cv;
	using namespace std;

	void samplerBatch(vector<Mat>& in_ims, Size dstsize, vector<Rect>& bboxs, const vector<bool>& positive_flags, vector<Mat>& out_ims, vector<Rect>& out_bboxs, int min_size = 30, float max_acc = 1.5);

#endif
}  // namespace caffe
