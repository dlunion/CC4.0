#include <boost/thread.hpp>
#include "caffe/layer.hpp"
#include "caffe/cc/core/cc.h"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template <typename Dtype>
Layer<Dtype>::~Layer() {
	if (this->cclayer_)
		delete this->cclayer_;
	this->cclayer_ = 0;
}

template <typename Dtype>
Layer<Dtype>::Layer(const LayerParameter& param)
: layer_param_(param), is_shared_(false) {
	// Set phase and copy blobs (if there are any).
	phase_ = param.phase();
	if (layer_param_.blobs_size() > 0) {
		blobs_.resize(layer_param_.blobs_size());
		for (int i = 0; i < layer_param_.blobs_size(); ++i) {
			blobs_[i].reset(new Blob<Dtype>());
			blobs_[i]->FromProto(layer_param_.blobs(i));
		}
	}
	this->cclayer_ = new cc::Layer();
	this->cclayer_->setNative(this);
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
