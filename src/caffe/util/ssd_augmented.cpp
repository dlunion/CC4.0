
#include "caffe/util/ssd_augmented.hpp"

namespace caffe {
#ifdef USE_OPENCV

	using namespace cv;
	using namespace std;

#define min(a, b)  ((a)<(b)?(a):(b))
#define max(a, b)  ((a)>(b)?(a):(b))

	//从minval到maxval（包括哦）
	static int randr(int minval, int maxval){
		if (minval > maxval) std::swap(minval, maxval);
		return (rand() % (maxval - minval + 1)) + minval;
	}

	//从minval到maxval（包括哦）
	static float randrf(float minval, float maxval){
		if (minval > maxval) std::swap(minval, maxval);
		float acc = rand() / (float)(RAND_MAX);
		return minval + (maxval - minval) * acc;
	}

	static float IoU(const Rect& a, const Rect& b){
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.x + a.width - 1, b.x + b.width - 1);
		float ymin = min(a.y + a.height - 1, b.y + b.height - 1);

		//Union
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float u = uw * uh;
		//return u / (a.area() + b.area() - u);
		return u / min(a.area(), b.area());
	}

	static Rect randbbox_positive(Size imsize, Rect bbox, int min_size = 30, float max_acc = 1.5){
		int pw = min(bbox.width, bbox.height);
		int min_bbox_dim = min(min_size, pw);
		float val_acc = pw / (float)min_bbox_dim;
		float sel_acc = randrf(1, val_acc);
		float acc = min(sel_acc, max(max_acc, 1));
		int bbox_r = bbox.x + bbox.width - 1;
		int bbox_b = bbox.y + bbox.height - 1;
		int w = randr(max(bbox.width, bbox.height), imsize.width * acc);
		int h = w / (float)imsize.width * imsize.height;
		int x = randr(bbox_r - w, bbox.x);
		int y = randr(bbox_b - h, bbox.y);
		return Rect(x, y, w, h);
	}

	static Rect randbbox(Size imsize, Size dstsize, Size minsize){
		int max_w = imsize.width;
		int max_w_with_h = max_w / (float)dstsize.width * dstsize.height;
		int max_h = min(imsize.height, max_w_with_h);
		int min_w = min(minsize.width, max_w);
		int min_h = min(minsize.height, max_h);
		int w = randr(min_w, max_w);
		int h = w / (float)dstsize.width * dstsize.height;
		h = max(min_h, h);
		int x = randr(0, imsize.width - w);
		int y = randr(0, imsize.height - h);
		return Rect(x, y, w, h);
	}

	static Rect randbbox_negitive(Size imsize){
		int w = randr(min(imsize.width, 100), imsize.width * 1.5);
		int h = w / (float)imsize.width * imsize.height;
		float acc = 0.5;
		int x = randr(-w*acc, w*acc);
		int y = randr(-h*acc, h*acc);
		return Rect(x, y, w, h);
	}

	static Rect flipBox(Size limit, Rect box, int code){
		if (code == -1){
			Rect b = flipBox(limit, box, 0);
			b = flipBox(limit, b, 1);
			return b;
		}

		int limw = limit.width - 1;
		int limh = limit.height - 1;
		int x = box.x;
		int y = box.y;
		int r = box.width + x - 1;
		int b = box.height + y - 1;
		int nx, ny, nr, nb;
		if (code == 1){
			//x方向，左右对调
			nx = limw - r;
			ny = y;
			nr = limw - x;
			nb = b;
		}
		else{
			//y方向，上下对调
			nx = x;
			ny = limh - b;
			nr = r;
			nb = limh - y;
		}
		return Rect(nx, ny, nr - nx + 1, nb - ny + 1);
	}

	static void randFlipImageAndBox_(Mat& im, Rect& box){
		int code = randr(-1, 1);

		flip(im, im, code);
		box = flipBox(Size(im.cols, im.rows), box, code);
	}

	static void randFlipImageAndBox(Mat& im, Rect& box){
		int trials = randr(0, 2);
		for (int i = 0; i < trials; ++i)
			randFlipImageAndBox_(im, box);
	}

	//通过原始对象的bbox，和采样的box，计算新图像的bbox位置
	static Rect resizeBBox(const Rect& bbox, Size srcsize, Size dstsize){
		float x = bbox.x;
		float y = bbox.y;
		float r = bbox.x + bbox.width - 1;
		float b = bbox.y + bbox.height - 1;
		float xscale = dstsize.width / (float)srcsize.width;
		float yscale = dstsize.height / (float)srcsize.height;
		int nx = x * xscale + 0.5;
		int ny = y * yscale + 0.5;
		int nr = r * xscale + 0.5;
		int nb = b * yscale + 0.5;
		return Rect(nx, ny, nr - nx + 1, nb - ny + 1);
	}

	static void light_aduject(Mat& im, int light){
		Mat hsv;
		Mat ms[3];
		cvtColor(im, hsv, CV_BGR2HSV);
		split(hsv, ms);
		ms[2] += light;
		merge(ms, 3, hsv);
		cvtColor(hsv, im, CV_HSV2BGR);
	}

	static Mat sampler(const Mat& im, Size dstsize, Rect bbox, bool is_positive, Rect& out_bbox, int min_size, float max_acc){

		bbox = bbox & Rect(0, 0, im.cols, im.rows);

		Mat tim = im.clone();
		light_aduject(tim, randr(-50, 50));

		Mat rmat;
		Rect sample_bbox(0, 0, im.cols, im.rows);
		if (is_positive)
			out_bbox = resizeBBox(bbox, sample_bbox.size(), dstsize);

		if (randr(0, 1) == 0){
			if (is_positive){
				sample_bbox = randbbox_positive(im.size(), bbox, min_size, max_acc);

				Rect pos_roi(bbox.x - sample_bbox.x, bbox.y - sample_bbox.y, bbox.width, bbox.height);
				out_bbox = resizeBBox(pos_roi, sample_bbox.size(), dstsize);
			}
			else{
				sample_bbox = randbbox_negitive(im.size());
			}
		}

		rmat = Mat(sample_bbox.size(), CV_8UC3);
		cv::randu(rmat, 0, 255);

		int x = sample_bbox.x;
		int y = sample_bbox.y;
		int r = sample_bbox.x + sample_bbox.width - 1;
		int b = sample_bbox.y + sample_bbox.height - 1;
		int ux = 0;
		int uy = 0;
		if (x > 0)
			ux = x;

		if (y > 0)
			uy = y;

		r = min(r, im.cols - 1);
		b = min(b, im.rows - 1);
		Rect roi(ux, uy, r - ux + 1, b - uy + 1);
		int offsetx = x < 0 ? -x : 0;
		int offsety = y < 0 ? -y : 0;
		tim(roi).copyTo(rmat(Rect(offsetx, offsety, roi.width, roi.height)));
		resize(rmat, rmat, dstsize, 0, 0, CV_INTER_CUBIC);
		return rmat;
	}

	void samplerBatch(vector<Mat>& in_ims, Size dstsize, vector<Rect>& bboxs, const vector<bool>& positive_flags, vector<Mat>& out_ims, vector<Rect>& out_bboxs, int min_size, float max_acc){
		//这里需要保证比例，即正负样本比例
		out_ims.resize(in_ims.size());
		out_bboxs.resize(in_ims.size());

		for (int i = 0; i < in_ims.size(); ++i){
			randFlipImageAndBox(in_ims[i], bboxs[i]);
			out_ims[i] = sampler(in_ims[i], dstsize, bboxs[i], positive_flags[i], out_bboxs[i], min_size, max_acc);
		}
	}
#endif
}  // namespace caffe