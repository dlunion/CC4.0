#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
	  top[0]->mutable_gpu_data());
  DLOG(INFO) << "Prefetch copied";

  //label
  top[1]->ReshapeLike(batch->label_);
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
	  top[1]->mutable_gpu_data());

  //roi
  top[2]->ReshapeLike(batch->roi_);
  caffe_copy(batch->roi_.count(), batch->roi_.gpu_data(),
	  top[2]->mutable_gpu_data());

  if (output_pts_){
	  //pts
	  top[3]->ReshapeLike(batch->pts_);
	  caffe_copy(batch->pts_.count(), batch->pts_.gpu_data(),
		  top[3]->mutable_gpu_data());
  }

  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
INSTANTIATE_LAYER_GPU_FUNCS(MTCNNDataLayer);

}  // namespace caffe
