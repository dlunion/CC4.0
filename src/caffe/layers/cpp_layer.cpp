#include <vector>

#include "caffe/layers/cpp_layer.hpp"
#include "glog/logging.h"
#include "caffe/cc/core/cc.h"

static cc::newLayerFunction ccNewLayerFunction = 0;
static cc::customLayerForward ccCustomLayerForward = 0;
static cc::customLayerBackward ccCustomLayerBackward = 0;
static cc::customLayerReshape ccCustomLayerReshape = 0;
static cc::customLayerRelease ccCustomLayerRelease = 0;

CCAPI void CCCALL cc::registerLayerFunction(cc::newLayerFunction newlayerFunc){
	ccNewLayerFunction = newlayerFunc;
}

CCAPI void CCCALL cc::registerLayerForwardFunction(cc::customLayerForward forward){
	ccCustomLayerForward = forward;
}

CCAPI void CCCALL cc::registerLayerBackwardFunction(cc::customLayerBackward backward){
	ccCustomLayerBackward = backward;
}

CCAPI void CCCALL cc::registerLayerReshapeFunction(cc::customLayerReshape reshape){
	ccCustomLayerReshape = reshape;
}

CCAPI void CCCALL cc::registerLayerReleaseFunction(cc::customLayerRelease release){
	ccCustomLayerRelease = release;
}

namespace caffe {
template <typename Dtype>
CPPLayer<Dtype>::CPPLayer(const LayerParameter& param)
	: Layer<Dtype>(param){ 
		this->layerInstance_ = 0;
	}

template <typename Dtype>
CPPLayer<Dtype>::~CPPLayer(){
	if (ccCustomLayerRelease && this->layerInstance_){
		ccCustomLayerRelease(this->layerInstance_);
	}
}

template <typename Dtype>
void CPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	string param_str = this->layer_param_.cpp_param().param_str();
	string name = this->layer_param_.name();
	string type = this->layer_param_.cpp_param().type();

	if (top.size() > 0){
		for (int i = 0; i < top.size(); ++i)
			this->set_loss(i, 1);
	}

	if (ccNewLayerFunction){
		vector<cc::Blob*> bottom_o(bottom.size());
		vector<cc::Blob*> top_o(top.size());
		for (int i = 0; i < bottom.size(); ++i)
			bottom_o[i] = bottom[i]->ccBlob();

		for (int i = 0; i < top_o.size(); ++i)
			top_o[i] = top[i]->ccBlob();

		layerInstance_ = ccNewLayerFunction(
			name.c_str(), type.c_str(), param_str.c_str(),
			static_cast<int>(this->phase_),
			bottom_o.size() > 0 ? &bottom_o[0] : 0, bottom_o.size(),
			top_o.size() > 0 ? &top_o[0] : 0, top_o.size(), this);
	}
	else{
		LOG(FATAL) << "no register cpp_layer callback ccNewLayerFunction";
	}
}

template <typename Dtype>
void CPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if (ccCustomLayerReshape){

		vector<cc::Blob*> bottom_o(bottom.size());
		vector<cc::Blob*> top_o(top.size());
		for (int i = 0; i < bottom.size(); ++i)
			bottom_o[i] = bottom[i]->ccBlob();

		for (int i = 0; i < top_o.size(); ++i)
			top_o[i] = top[i]->ccBlob();

		ccCustomLayerReshape(
			this->layerInstance_, 
			bottom_o.size() > 0 ? &bottom_o[0] : 0, bottom_o.size(), 
			top_o.size() > 0 ? &top_o[0] : 0, top_o.size());
	}
	else{
		LOG(FATAL) << "no register cpp_layer callback ccCustomLayerReshape";
	}
}

template <typename Dtype>
bool CPPLayer<Dtype>::ShareInParallel() const {
	return true;
}

template <typename Dtype>
void CPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	if (ccCustomLayerForward){
		vector<cc::Blob*> bottom_o(bottom.size());
		vector<cc::Blob*> top_o(top.size());
		for (int i = 0; i < bottom.size(); ++i)
			bottom_o[i] = bottom[i]->ccBlob();

		for (int i = 0; i < top_o.size(); ++i)
			top_o[i] = top[i]->ccBlob();

		ccCustomLayerForward(this->layerInstance_, 
			bottom_o.size() > 0 ? &bottom_o[0] : 0, bottom_o.size(), 
			top_o.size() > 0 ? &top_o[0] : 0, top_o.size());
	}
	else{
		LOG(FATAL) << "no register cpp_layer callback ccCustomLayerForward";
	}
}

template <typename Dtype>
void CPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (ccCustomLayerBackward){
		vector<cc::Blob*> bottom_o(bottom.size());
		vector<cc::Blob*> top_o(top.size());
		for (int i = 0; i < bottom.size(); ++i)
			bottom_o[i] = bottom[i]->ccBlob();

		for (int i = 0; i < top_o.size(); ++i)
			top_o[i] = top[i]->ccBlob();

		CHECK_LT(propagate_down.size(), 100) << "error.";
		bool propagate_down_o[100] = { 0 };
		for (int i = 0; i < propagate_down.size(); ++i)
			propagate_down_o[i] = propagate_down[i];

		ccCustomLayerBackward(this->layerInstance_, 
			bottom_o.size() > 0 ? &bottom_o[0] : 0, bottom_o.size(),
			top_o.size() > 0 ? &top_o[0] : 0, top_o.size(), propagate_down_o);
	}
	else{
		LOG(FATAL) << "no register cpp_layer callback ccCustomLayerBackward";
	}
}

INSTANTIATE_CLASS(CPPLayer);
REGISTER_LAYER_CLASS(CPP);
}  // namespace caffe

