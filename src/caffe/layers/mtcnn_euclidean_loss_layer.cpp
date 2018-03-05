#include <vector>

#include "caffe/layers/mtcnn_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
   
  int has_ignore_label = this->layer_param().loss_param().has_ignore_label();
  if (has_ignore_label)
	  CHECK_EQ(bottom.size(), 3) << "has_ignore_label=true but not input label";

  if (!has_ignore_label)
	  CHECK_EQ(bottom.size(), 2) << "has_ignore_label=false but input mismatch";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int has_ignore_label = this->layer_param().loss_param().has_ignore_label();
  int ignore_label = has_ignore_label ? this->layer_param().loss_param().ignore_label() : -1;

  if (has_ignore_label){
	  const Dtype* label = bottom[2]->cpu_data();
	  int countLabel = bottom[2]->num();

	  //label
	  Dtype* diff = diff_.mutable_cpu_data();
	  int channel = bottom[0]->channels();
	  memset(diff, 0, sizeof(Dtype)*count);

	  const Dtype* b0 = bottom[0]->cpu_data();
	  const Dtype* b1 = bottom[1]->cpu_data();
	  Dtype loss = 0;

	  for (int i = 0; i < countLabel; ++i){
		  if (label[i] != ignore_label){
			  caffe_sub(
				  channel,
				  b0 + i * channel,
				  b1 + i * channel,
				  diff + i * channel);
			  Dtype dot = caffe_cpu_dot(channel, diff + i * channel, diff + i * channel);
			  loss += dot / Dtype(2);
		  }
	  }

	  top[0]->mutable_cpu_data()[0] = loss;
  }
  else{
	  caffe_sub(
		  count,
		  bottom[0]->cpu_data(),
		  bottom[1]->cpu_data(),
		  diff_.mutable_cpu_data());
	  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
	  Dtype loss = dot / bottom[0]->num() / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int has_ignore_label = this->layer_param().loss_param().has_ignore_label();
	int ignore_label = has_ignore_label ? this->layer_param().loss_param().ignore_label() : -1;

	if (has_ignore_label){
		const Dtype* label = bottom[2]->cpu_data();
		int countLabel = bottom[2]->num();
		int channels = bottom[0]->channels();
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				memset(bottom[i]->mutable_cpu_diff(), 0, sizeof(Dtype)*bottom[i]->count());

				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();

				for (int j = 0; j < countLabel; ++j){
					if (label[j] != ignore_label){
						caffe_cpu_axpby(
							channels,							// count
							alpha,                              // alpha
							diff_.cpu_data() + channels * j,                   // a
							Dtype(0),                           // beta
							bottom[i]->mutable_cpu_diff() + channels * j);  // b
					}
				}
			}
		}
	}
	else{
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				caffe_cpu_axpby(
					bottom[i]->count(),              // count
					alpha,                              // alpha
					diff_.cpu_data(),                   // a
					Dtype(0),                           // beta
					bottom[i]->mutable_cpu_diff());  // b
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(MTCNNEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(MTCNNEuclideanLossLayer);
REGISTER_LAYER_CLASS(MTCNNEuclideanLoss);

}  // namespace caffe
