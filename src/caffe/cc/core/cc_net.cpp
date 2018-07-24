
#include "caffe/cc/core/cc.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include "caffe/net.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/cc/core/cc.h"

#ifdef USE_CC_PYTHON
#include <memory>

#undef _DEBUG
#include <Python.h>
#endif

using namespace std;
using namespace cv;

namespace cc{

#define cvt(p)	((caffe::Net<float>*)p)
#define ptr		(cvt(this->_native))

	Blob* Net::blob(const char* name){
		const boost::shared_ptr<caffe::Blob<float> > blob = ptr->blob_by_name(name);
		if (blob.get() == NULL)
			return 0;

		return blob->ccBlob();
	}

	void Net::Reshape(){
		ptr->Reshape();
	}

	void Net::setNative(void* native){
		this->_native = native;
	}

	void* Net::getNative(){
		return this->_native;
	}

	void Net::Forward(float* loss){
		ptr->Forward(loss);
	}

	size_t Net::memory_used(){
		return ptr->memory_used_;
	}

	void Net::copyTrainedParamFromFile(const char* file){
		ptr->CopyTrainedLayersFrom(file);
	}

	void Net::ShareTrainedLayersWith(const Net* other){
		ptr->ShareTrainedLayersWith(cvt(other->_native));
	}

	void Net::copyTrainedParamFromData(const void* data, int length){
		ptr->CopyTrainedLayersFromData(data, length);
	}

	int Net::num_layers(){
		return ptr->layers().size();
	}

	Layer* Net::layer(const char* name){
		return ptr->layer_by_name(name)->ccLayer();
	}

	Layer* Net::layer(int index){
		return index < 0 || index >= num_layers() ? 0 : ptr->layers()[index]->ccLayer();
	}

	CCString Net::layer_name(int index){
		Layer* l = layer(index);
		if (l) return l->name();
		return "";
	}

	CCAPI void CCCALL releaseNet(Net* net){
		if (net){
			void* p = net->getNative();
			if (p) delete cvt(p);
		}
	}

	bool Net::has_blob(const char* name){
		return ptr->has_blob(name);
	}

	Blob* Net::blob(int index){
		if (index < 0 || index >= num_blobs())
			return 0;
		return ptr->blobs()[index]->ccBlob();
	}

	bool Net::has_layer(const char* name){
		return ptr->has_layer(name);
	}

	int Net::num_blobs(){
		return ptr->blobs().size();
	}

	CCString Net::blob_name(int index){
		if (index < 0 || index >= num_blobs())
			return "";

		return ptr->blob_names()[index].c_str();
	}

	int Net::num_input_blobs(){
		return ptr->num_inputs();
	}

	int Net::num_output_blobs(){
		return ptr->num_outputs();
	}

	Blob* Net::input_blob(int index){
		CHECK_GE(index, 0);
		CHECK_LT(index, num_input_blobs());
		return ptr->input_blobs()[index]->ccBlob();
	}

	Blob* Net::output_blob(int index){
		CHECK_GE(index, 0);
		CHECK_LT(index, num_output_blobs());
		return ptr->output_blobs()[index]->ccBlob();
	}

	CCAPI Net* CCCALL loadNetFromPrototxt(const char* net_prototxt, int phase){
		caffe::Net<float>* net = new caffe::Net<float>(net_prototxt, phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST);
		return net->ccNet();
	}

	CCAPI Net* CCCALL loadNetFromPrototxtString(const char* net_prototxt, int length, int phase){
		caffe::Net<float>* net = new caffe::Net<float>(net_prototxt, length < 1 ? strlen(net_prototxt) : length, phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST);
		return net->ccNet();
	}

	CCAPI Net* CCCALL newNetFromParam(const caffe::NetParameter& param){
		caffe::Net<float>* net = new caffe::Net<float>(param, 0);
		return net->ccNet();
	}

#ifdef USE_CC_PYTHON
	CCAPI Net* CCCALL loadNetFromPython(const char* pythonfile, const char* func, int phase){
		Net* net_ptr = 0;
		{
			CCPython pp;
			bool ok = pp.load(pythonfile);
			CHECK(ok) << pp.last_error();
			if (!ok) return 0;

			string prototxt = pp.callstringFunction(func);
			net_ptr = loadNetFromPrototxtString(prototxt.c_str(), prototxt.size(), phase);
		}
		return net_ptr;
	}
#endif
}