#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;


namespace caffe {
#ifdef USE_OPENCV
	struct SampleInfo{
		Mat im;
		int label;				// 0, -1, 1
		float offx1, offx2, offy1, offy2;
	};

	void genSamples(const vector<Mat>& ims, const vector<vector<Rect>>& boxs, Size resize_size, 
		int num_positive, int num_negitive, int num_part, vector<SampleInfo>& out_samples, 
		float min_negitive_scale, float max_negitive_scale, bool augmented = false, bool flip = false);
#endif
}  // namespace caffe
