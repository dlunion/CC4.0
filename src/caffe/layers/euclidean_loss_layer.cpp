#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //std::cout << bottom[0]->shape_string() << "  ::  " << bottom[1]->shape_string() << std::endl;
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  if (bottom.size() == 3){
	  CHECK_EQ(bottom[2]->count(1), bottom[0]->count(1))
		  << "Inputs weights(bottom 3) have the same dimension.";
  }
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	if (bottom.size() == 2){
		int count = bottom[0]->count();
		caffe_sub(
			count,
			bottom[0]->cpu_data(),
			bottom[1]->cpu_data(),
			diff_.mutable_cpu_data());
		Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
		Dtype loss = dot / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
		num_labels = bottom[0]->num();
	}
	else if (bottom.size() == 3){
		//a, b, weights
#if 0
		const Dtype* a = bottom[0]->cpu_data();
		const Dtype* b = bottom[1]->cpu_data();
		const Dtype* mask = bottom[2]->cpu_data();
		Dtype* diff = diff_.mutable_cpu_data();
		int channels = bottom[0]->channels();
		int num = bottom[0]->num();
		int w = bottom[0]->width();
		int h = bottom[0]->height();
		int plane = w * h;
		Dtype dot = 0;
		num_labels = 0;

		//通道不同
		for (int n = 0; n < num; ++n){
			for (int i = 0; i < w; ++i){
				for (int j = 0; j < h; ++j){
					
					/*
					Dtype v = *(mask + i + j * w + n * plane);
					if (v != 0){
						for (int c = 0; c < channels; ++c){
							num_labels++;
							const Dtype* pa = a + i + j * w + n * channels * plane + plane * c;
							const Dtype* pb = b + i + j * w + n * channels * plane + plane * c;
							Dtype* pdiff = diff + i + j * w + n * channels * plane + plane * c;
							*pdiff = (*pa - *pb) * (*pa - *pb);
							dot += *pdiff;
						}
					}
					*/
				}
			}
		}
		Dtype loss = num_labels == 0 ? 0 : dot / num_labels / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
#endif

		int count = bottom[0]->count();
		caffe_sub(
			count,
			bottom[0]->cpu_data(),
			bottom[1]->cpu_data(),
			diff_.mutable_cpu_data());
		caffe_mul(count, bottom[2]->cpu_data(), diff_.mutable_cpu_data(), diff_.mutable_cpu_data());

		Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
		Dtype loss = dot / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
		num_labels = bottom[0]->num();
	}
	//printf("修改啦.\n");
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_labels;
			caffe_cpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.cpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_cpu_diff());  // b
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
