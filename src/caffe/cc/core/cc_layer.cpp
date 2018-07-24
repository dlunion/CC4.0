

#include "caffe/cc/core/cc.h"
#include "caffe/layer.hpp"
#include <google/protobuf/text_format.h>

namespace cc{

#define cvt(p)	((caffe::Layer<float>*)p)
#define ptr		(cvt(this->_native))

	using google::protobuf::Message;
	void Layer::setNative(void* native){
		this->_native = native;
	}

	CCString Layer::name(){
		return ptr->layer_param_.name().c_str();
	}

	const char* Layer::type() const{
		return ptr->type();
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

	CCString Layer::paramString(){
		string output;
		google::protobuf::TextFormat::PrintToString(ptr->layer_param(), &output);
		return CCString(output.c_str());
	}

	bool Layer::getParam(const char* path, Value& val){
		return getMessageValue((MessageHandle)&ptr->layer_param(), path, val);
	}

	int Layer::getNumBottom(){
		return ptr->layer_param_.bottom_size();
	}

	int Layer::getNumTop(){
		return ptr->layer_param_.top_size();
	}

	CCString Layer::bottomName(int index){
		if (index >= 0 && index < getNumBottom())
			return ptr->layer_param_.bottom(index).c_str();
		else
			return "";
	}

	CCString Layer::topName(int index){
		if (index >= 0 && index < getNumTop())
			return ptr->layer_param_.top(index).c_str();
		else
			return "";
	}

	Blob* Layer::paramBlob(int index){
		return ptr->blobs_[index]->ccBlob();
	}

	int Layer::getNumParamBlob(){
		return ptr->blobs_.size();
	}

	bool Layer::hasParam(const char* path){
		Value val;
		return getMessageValue((MessageHandle)&ptr->layer_param(), path, val);
	}

#ifdef USE_PROTOBUF
	caffe::LayerParameter& Layer::layer_param(){
		return ptr->layer_param_;
	}
#endif
};