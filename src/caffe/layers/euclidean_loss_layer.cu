#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
#if 0
	int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
#endif



  if (bottom.size() == 2){
	  int count = bottom[0]->count();
	  caffe_gpu_sub(
		  count,
		  bottom[0]->gpu_data(),
		  bottom[1]->gpu_data(),
		  diff_.mutable_gpu_data());
	  Dtype dot;
	  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	  Dtype loss = dot / bottom[0]->num() / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
	  num_labels = bottom[0]->num();
  }
  else if (bottom.size() == 3){

#if 0
	  const Dtype* a = bottom[0]->gpu_data();
	  const Dtype* b = bottom[1]->gpu_data();
	  const Dtype* label = bottom[2]->cpu_data();
	  Dtype* diff = diff_.mutable_gpu_data();
	  int channels = bottom[0]->channels();
	  int num = bottom[0]->num();
	  int w = bottom[0]->width();
	  int h = bottom[0]->height();
	  int plane = w * h;
	  Dtype dot = 0;
	  num_labels = 0;

	  //printf("num = %d, w = %d, h = %d\n", num, w, h);
	  caffe_gpu_memset(sizeof(Dtype)*bottom[0]->count(), 0, diff);

	  //通道不同
	  for (int n = 0; n < num; ++n){
		  for (int i = 0; i < w; ++i){
			  for (int j = 0; j < h; ++j){
				  Dtype v = *(label + i + j * w + n * w * h);
				  if (v != 0){
					  for (int c = 0; c < channels; ++c){
						  num_labels++;
						  const Dtype* pa = a + i + j * w + n * channels * w * h + w * h * c;
						  const Dtype* pb = b + i + j * w + n * channels * w * h + w * h * c;
						  Dtype* pdiff = diff + i + j * w + n * channels * w * h + w * h * c;
						  //*pdiff = (*pa - *pb) * (*pa - *pb);
						  //dot += *pdiff;

						  caffe_gpu_sub(1, pa, pb, pdiff);
					  }
				  }
			  }
		  }
	  }

	  caffe_gpu_dot(bottom[0]->count(), diff, diff, &dot);
	 // printf("num_labels = %d, dot = %f\n", num_labels, dot);
	  Dtype loss = num_labels == 0 ? 0 : dot / num_labels / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
#endif

	  int count = bottom[0]->count();
	  caffe_gpu_sub(
		  count,
		  bottom[0]->gpu_data(),
		  bottom[1]->gpu_data(),
		  diff_.mutable_gpu_data());
	  caffe_gpu_mul(count, bottom[2]->gpu_data(), diff_.mutable_gpu_data(), diff_.mutable_gpu_data());

	  Dtype dot = 0;
	  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);

	  Dtype loss = dot / bottom[0]->num() / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
	  num_labels = bottom[0]->num();
  }

  //printf("修改啦.\n");
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {

	  //printf("num_labels2 = %d\n", num_labels);
      const Dtype sign = (i == 0) ? 1 : -1;
	  const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_labels;
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
