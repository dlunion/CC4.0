

#include "caffe/cc/core/cc.h"
#include "caffe/layer.hpp"

namespace cc{

#define cvt(p)	((caffe::Layer<float>*)p)
#define ptr		(cvt(this->_native))

	void Layer::setNative(void* native){
		this->_native = native;
	}

	void Layer::setupLossWeights(int num, float* weights){
		for (int i = 0; i < num; ++i)
			ptr->set_loss(i, weights[i]);
	}

	float Layer::lossWeights(int index){
		return ptr->loss(index);
	}

	void Layer::setLossWeights(int index, float weights){
		ptr->set_loss(index, weights);
	}

#ifdef USE_PROTOBUF
	caffe::LayerParameter& Layer::layer_param(){
		return ptr->layer_param_;
	}
#endif
};