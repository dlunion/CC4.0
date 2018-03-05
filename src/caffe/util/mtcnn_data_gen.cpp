#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"
#include "caffe/util/mtcnn_data_gen.hpp"
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

namespace caffe {
#ifdef USE_OPENCV

#define max(a, b)  ((a) > (b) ? (a) : (b))
#define min(a, b)  ((a) < (b) ? (a) : (b))

	vector<Rect> loadbbox(const string& file){
		vector<Rect> r;
		FILE* f = fopen(file.c_str(), "rb");
		if (f){
			int num, tmp;
			char buf[100];
			fscanf(f, "%d,%d\n", &num, &tmp);

			r.resize(num);
			for (int i = 0; i < num; ++i){
				fscanf(f, "%d,%d,%d,%d,%d,%s\n", &r[i].x, &r[i].y, &r[i].width, &r[i].height, &tmp, &buf);
				r[i].width = r[i].width - r[i].x + 1;
				r[i].height = r[i].height - r[i].y + 1;
			}
			fclose(f);
		}
		return r;
	}

	float IoU(const Rect& a, const Rect& b){
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.x + a.width - 1, b.x + b.width - 1);
		float ymin = min(a.y + a.height - 1, b.y + b.height - 1);

		//Union
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float iou = uw * uh;
		return iou / min(a.area(), b.area());
	}

	//从minval到maxval（包括哦）
	int randr(int minval, int maxval){
		if (minval > maxval) std::swap(minval, maxval);
		return (rand() % (maxval - minval + 1)) + minval;
	}

	//从minval到maxval（包括哦）
	float randrf(float minval, float maxval){
		if (minval > maxval) std::swap(minval, maxval);
		float acc = rand() / (float)(RAND_MAX);
		return minval + (maxval - minval) * acc;
	}

	//minacc最小波动的比例系数
	//maxacc最大波动的比例系数
	Rect randBox(Rect box, float size_minacc, float size_maxacc, float off_minacc, float off_maxacc){
		float sacc = randrf(size_minacc, size_maxacc);
		float xacc = randrf(off_minacc, off_maxacc);
		float yacc = randrf(off_minacc, off_maxacc);

		//保证wacc + xacc最大值是maxacc
		float x = box.x + box.width * 0.5 + box.width * xacc;
		float y = box.y + box.height * 0.5 + box.height * yacc;
		float w = box.width * sacc;
		float h = box.height * sacc;
		x = x - w * 0.5;
		y = y - h * 0.5;
		w = round(w);
		h = round(h);
		return Rect(x, y, w, h);
	}

	//使得宽高比满足acc要求
	Rect sameAccBox(Rect box, float acc){
		float cacc = box.width / (float)box.height;
		if (fabs(cacc - acc) <= 0.001)
			return box;

		//nw / nh = acc
		//nw = acc * nh
		//nw = w + n;
		//nh = h + m;
		//w + n = acc * (h + m);
		//如果n = 0
		//w = acc * (h + m)，如果这时候m也大于等于0，就满足要求啦，否则就置m为0求n
		//m = w / acc - h;

		float m = box.width / acc - box.height;		//n = 0时m的值
		float n = acc * box.height - box.width;		//m = 0时n的值
		float cx = box.x + box.width * 0.5;
		float cy = box.y + box.height * 0.5;
		float nw = box.width;
		float nh = box.height;

		//w + n = acc * (h + m);
		//n = acc * h - w  //这时候设置m为0，求n
		if (m >= 0){
			//如果m满足条件，就说明n为0是可以接受的
			nh = box.height + m;
		}
		else{
			//此时n >= 0, m = 0
			nw = box.width + n;
		}
		return Rect(cx - nw * 0.5, cy - nh * 0.5, nw, nh);
	}

	//如果x、y、r、b超出范围，则平移到范围内
	Rect transBox(Rect box, Size limit){
		int x = box.x;
		int y = box.y;
		int r = box.x + box.width - 1;
		int b = box.y + box.height - 1;
		int xval = box.x;
		int yval = box.y;

		if (x < 0){
			xval = 0;
		}
		else if (r >= limit.width){
			xval = limit.width - box.width;
		}

		if (y < 0){
			yval = 0;
		}
		else if (b >= limit.height){
			yval = limit.height - box.height;
		}

		Rect rb(xval, yval, box.width, box.height);
		rb = rb & Rect(0, 0, limit.width, limit.height);
		return rb;
	}

	//获取一个负样本box，此时2个条件，第一是宽高比满足whacc，第二是IoU要在0.3以下
	Rect getNegitiveBox(const vector<Rect>& boxs, float whacc, Size size, Size limit, float min_negitive_scale, float max_negitive_scale, Size resize_size){
		if (boxs.size() == 0) return Rect();

		Rect gen;
		int retryMax = 100;		//如果100次没找到合适的，就返回
		bool found = true;
		while (retryMax > 0){
			gen = sameAccBox(randBox(Rect(0, 0, size.width, size.height), min_negitive_scale, max_negitive_scale, 0, 0), whacc);
			gen.width = max(gen.width, 12);
			gen.height = max(gen.height, 12);
			gen.x = randr(0, limit.width - gen.width);
			gen.y = randr(0, limit.height - gen.height);
			gen = gen & Rect(0, 0, limit.width, limit.height);

			found = true;
			for (int i = 0; i < boxs.size(); ++i){
				if (IoU(boxs[i], gen) > 0.3){
					found = false;
					break;
				}
			}
			if (found) break;
			retryMax--;
		}
		return found ? gen : Rect(0, 0, 0, 0);
	}

	void process(Mat& im){
		Mat mask = Mat::zeros(im.size(), CV_8U);
		float r = randr(100, max(im.cols, im.rows)*0.5);
		circle(mask, Point(randr(10, im.cols - 10), randr(10, im.rows - 10)), r, Scalar::all(255), -1);
		blur(mask, mask, Size(r, r));

		mask.convertTo(mask, CV_32F, 1 / 255.0);
		Mat color(im.size(), CV_32FC3, Scalar::all(randr(-0.5, 0.5)));
		Mat maskMerge;
		cvtColor(mask, maskMerge, CV_GRAY2BGR);

		Mat t = color.mul(maskMerge);
		cv::add(im, t, im);
	}

	void aug(Mat& im){
		int num = randr(0, 5);
		for (int i = 0; i < num; ++i){
			process(im);
		}
	}

	Rect flipBox(Size limit, Rect box, int code){
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

	void randFlipImageAndBox(Mat& im, Rect& box){
		int code = randr(-1, 1);

		flip(im, im, code);
		box = flipBox(Size(im.cols, im.rows), box, code);
	}

	void genSamples(const vector<Mat>& ims, const vector<vector<Rect>>& boxs, Size resize_size, int num_positive, 
		int num_negitive, int num_part, vector<SampleInfo>& out_samples, 
		float min_negitive_scale, float max_negitive_scale, bool augmented, bool flip){
		out_samples.clear();

		int maked_positive = 0;
		int maked_negitive = 0;
		int maked_part = 0;
		float whacc = resize_size.width / (float)resize_size.height;

#if 0
		printf("boxs.size = %d\n", boxs.size());
		for (int i = 0; i < boxs.size(); ++i){
			for (int j = 0; j < boxs[i].size(); ++j)
				printf("%d.%d,%d,%d,%d\n", i, boxs[i][j].x, boxs[i][j].y, boxs[i][j].width, boxs[i][j].height);
		}
#endif

		while (maked_positive < num_positive || maked_negitive < num_negitive || maked_part < num_part){
			//printf("%d,%d,%d\n", maked_positive, maked_negitive, maked_part);

			int img_index = randr(0, (int)ims.size() - 1);
			int box_index = randr(0, (int)boxs[img_index].size() - 1);
			Mat im = ims[img_index].clone();

			//增广
			if (augmented)
				aug(im);

			Rect box = boxs[img_index].size() > 0 ? boxs[img_index][box_index] : Rect(0, 0, 0, 0);
			Rect baseBox(0, 0, im.cols, im.rows);

			//镜像
			if (flip && box.width > 0 && box.height > 0)
				randFlipImageAndBox(im, box);

			//产生正样本
			if (maked_positive < num_positive && box.width > 0 && box.height > 0){
				//制造一个正样本，正样本，IoU大于0.8为正样本 
				Rect nbox = randBox(box, 0.8, 1.2, -0.2, 0.2);
				while (IoU(nbox, box) < 0.65)
					nbox = randBox(box, 0.8, 1.2, -0.2, 0.2);

				nbox = sameAccBox(nbox, whacc);
				nbox = transBox(nbox, Size(im.cols, im.rows));

				//主要是要保证输入的尺度是一致的
				if (nbox.width > 0 && nbox.height > 0){
					SampleInfo sample;
					resize(im(nbox), sample.im, resize_size);
					sample.label = 1;
					sample.offx1 = (box.x - nbox.x) / (float)nbox.width;
					sample.offx2 = (box.x + box.width - 1 - (nbox.x + nbox.width - 1)) / (float)nbox.width;
					sample.offy1 = (box.y - nbox.y) / (float)nbox.height;
					sample.offy2 = (box.y + box.height - 1 - (nbox.y + nbox.height - 1)) / (float)nbox.height;
					out_samples.push_back(sample);
					maked_positive++;
				}
			}

			//产生负样本
			if (maked_negitive < num_negitive){
				Size usize = box.width > 0 && box.height > 0 ? Size(box.width, box.height) : resize_size;
				Rect nbox = getNegitiveBox(boxs[img_index], whacc, usize, Size(im.cols, im.rows), min_negitive_scale, max_negitive_scale, resize_size);
				if (nbox.width > 0 && nbox.height > 0){
					SampleInfo sample;
					resize(im(nbox), sample.im, resize_size);
					sample.label = 0;
					sample.offx1 = 0;
					sample.offx2 = 0;
					sample.offy1 = 0;
					sample.offy2 = 0;
					out_samples.push_back(sample);
					maked_negitive++;
				}
			}

			//产生部分样本
			if (maked_part < num_part && box.width > 0 && box.height > 0){
				//制造一个部分样本，部分样本，IoU小于0.8为部分样本 
				Rect nbox = randBox(box, 0.8, 1.2, 0.3, -0.3);
				float iou = IoU(nbox, box);
				while (iou > 0.65 || iou < 0.4){
					nbox = randBox(box, 0.8, 1.2, 0.3, -0.3);
					iou = IoU(nbox, box);
				}

				nbox = sameAccBox(nbox, whacc);
				nbox = transBox(nbox, Size(im.cols, im.rows));

				//主要是要保证输入的尺度是一致的
				if (nbox.width > 0 && nbox.height > 0){
					SampleInfo sample;
					resize(im(nbox), sample.im, resize_size);
					sample.label = -1;
					sample.offx1 = (box.x - nbox.x) / (float)nbox.width;
					sample.offx2 = (box.x + box.width - 1 - (nbox.x + nbox.width - 1)) / (float)nbox.width;
					sample.offy1 = (box.y - nbox.y) / (float)nbox.height;
					sample.offy2 = (box.y + box.height - 1 - (nbox.y + nbox.height - 1)) / (float)nbox.height;
					out_samples.push_back(sample);
					maked_part++;
				}
			}
		}
	}

#endif
}  // namespace caffe
