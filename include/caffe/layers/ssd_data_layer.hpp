#ifndef CAFFE_SSD_DATA_LAYER_HPP_
#define CAFFE_SSD_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using namespace cv;
using namespace std;

struct ItemInfo{
	string image;
	string xml;
	Rect bbox;
	int label;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class SSDDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
	 explicit SSDDataLayer(const LayerParameter& param)
		 : BasePrefetchingDataLayer<Dtype>(param){ 
			 this->output_labels_ = true; 
			 this->positive_line_id_ = 0;
			 this->negitive_line_id_ = 0;
		}
	 virtual ~SSDDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   
  //virtual void Forward_cpu(
//	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SSDData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  void shuffle_images();
  
 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void load_batch(Batch<Dtype>* batch);
  std::map<std::string, int> label_map_;

  //positive
  std::vector<ItemInfo> positive_items_;
  std::vector<int> positive_order_list_;
  int positive_line_id_;

  //negitive
  std::vector<ItemInfo> negitive_items_;
  std::vector<int> negitive_order_list_;
  int negitive_line_id_;

  Scalar mean_values_;
};


}  // namespace caffe

#endif  // CAFFE_SSD_DATA_LAYER_HPP_
