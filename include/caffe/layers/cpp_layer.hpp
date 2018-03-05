#ifndef CAFFE_CPP_LAYER_HPP_
#define CAFFE_CPP_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"

namespace caffe {

	template <typename Dtype>
	class CPPLayer : public Layer<Dtype> {
	public:
		explicit CPPLayer(const LayerParameter& param);
		virtual ~CPPLayer();
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual bool ShareInParallel() const;
		virtual inline const char* type() const { return "CPP"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	private:
		void* layerInstance_;
	};

}  // namespace caffe

#endif
