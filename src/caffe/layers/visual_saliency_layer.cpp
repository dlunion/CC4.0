#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/visual_saliency_layer.hpp"


namespace caffe {

template <typename Dtype>
void VisualSaliencyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom.size())
    << "Only accept a bottom blob";
//  CHECK_EQ(2, bottom[0]->channels())
//    << "The bottom blob must be a saliency map feature blob of 2 channels";
  CHECK_LE(1, bottom[0]->height())
    << "The bottom blob should have a height greater than zero";
  CHECK_LE(1, bottom[0]->width())
    << "The bottom blob should have a width greater than zero";

  if (this->layer_param_.visual_saliency_param().has_save_dir()) {
    this->prefix_ = this->layer_param_.visual_saliency_param().save_dir();
    LOG(INFO) << "SaveDir: " << this->prefix_;
  } else {
    LOG(ERROR) << "Save Dir must be specific";
  }

  visual_interval_ = this->layer_param_.visual_saliency_param().visual_interval();
  visual_num_ = this->layer_param_.visual_saliency_param().visual_num();
  layer_name_ = this->layer_param_.name();
  iter_ = -1;

  CHECK_LE(visual_num_, bottom[0]->num())
    << "Visual num shold not exceed the num of images in an iteration";


}
template <typename Dtype>
void VisualSaliencyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void VisualSaliencyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  iter_++;    
  // Generate a Saliency Map from the blob
  Blob<Dtype>* visual_blob = bottom[0];
  int blob_channel = bottom[0]->channels();
  
  if (iter_ % visual_interval_ != 0)
    return;

  for (int n = 0; n < visual_num_; ++n)
  {
    cv::Mat img(visual_blob->height(), visual_blob->width(), CV_8UC1);
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
          Dtype value = visual_blob->data_at(n, blob_channel-1, h, w);
          // Sigmoid
          //Dtype value_s =  1. / (1. + exp(-value));
          // LOG(INFO) << value <<" " << value_s;
          img.at<uchar>(h, w) = cv::saturate_cast<uchar>(255*value);
      }
    }
    char path[200];
    sprintf(path, "%s/%d_%d_%s.jpg",
          this->prefix_.c_str(), iter_/visual_interval_, n, layer_name_.c_str());
    cv::imwrite(path, img);
  }

}

template <typename Dtype>
void VisualSaliencyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}
INSTANTIATE_CLASS(VisualSaliencyLayer);
REGISTER_LAYER_CLASS(VisualSaliency);

}  // namespace caffe

