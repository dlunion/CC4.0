#ifndef CAFFE_MTCNN_DATA_LAYER_HPP_
#define CAFFE_MTCNN_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

	template <typename Dtype>
	class MTCNNDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit MTCNNDataLayer(const LayerParameter& param);
		virtual ~MTCNNDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// DataLayer uses DataReader instead for sharing for parallelism
		virtual inline bool ShareInParallel() const { return false; }
		virtual inline const char* type() const { return "MTCNNData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }

		//data, label, roi, pts
		//virtual inline int ExactNumTopBlobs() const { return 4; }
		virtual inline int MinTopBlobs() const { return 2; }
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	protected:
		virtual void load_batch(Batch<Dtype>* batch);
		DataReader<MTCNNDatum> reader_;
		vector<int> raw_image_shape_;
	};

}  // namespace caffe

#endif  // CAFFE_MTCNN_DATA_LAYER_HPP_
