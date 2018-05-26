#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/contour_accuracy_layer.hpp"


namespace caffe {

template <typename Dtype>
void ContourAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  confusion_matrix_.clear();
  confusion_matrix_.resize(2); // 2 X 2 D matrix
  ContourAccuracyParameter contour_accuracy_param = this->layer_param_.contour_accuracy_param();
  for (int c = 0; c < contour_accuracy_param.ignore_label_size(); ++c){
    ignore_label_.insert(contour_accuracy_param.ignore_label(c));
  }
}

template <typename Dtype>
void ContourAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(1, bottom[0]->channels())
      << "top_k must be less than or equal to the number of channels (classes).";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1)
    << "The label should have one channel.";
  CHECK_EQ(bottom[0]->channels(), 1)
    << "The data should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";
  //confusion_matrix_.clear(); 
  ContourAccuracyParameter contour_accuracy_param = this->layer_param_.contour_accuracy_param();
  int accuracy_num = 1;
  if (contour_accuracy_param.has_accuracy_num()) {
    accuracy_num = contour_accuracy_param.accuracy_num();
  }
  top[0]->Reshape(1, 1, 1, accuracy_num);
}

template <typename Dtype>
void ContourAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
    
  int index;
  
  Dtype totalMAE = 0;
  int totalNum = 0;
  // remove old predictions if reset() flag is true
  if (this->layer_param_.contour_accuracy_param().reset()) {
    confusion_matrix_.clear();
  }

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
      	index = h * width + w;
	      const int gt_label = static_cast<int>(bottom_label[index]);
        const Dtype pre_label = bottom_data[index];

	     if (ignore_label_.count(gt_label) != 0) {
	       // ignore the pixel with this gt_label
	       continue;
	     } else if (gt_label >= 0) {
	       // current position is not "255", indicating ambiguous position
        totalNum++;
        Dtype mae = gt_label > pre_label ? (gt_label-pre_label):(pre_label-gt_label);
   //     LOG(INFO) << "pre_label:" << pre_label << " mae:" << mae;
        totalMAE += mae;
 
        int pre_label_bin = 0;
        if(pre_label > 0.5) {
          pre_label_bin = 1;
       //  LOG(INFO) << "pre_label" << pre_label;          
        }

	      confusion_matrix_.accumulate(gt_label, pre_label_bin);
	     } else {
	      LOG(FATAL) << "Unexpected label " << gt_label << ". num: " << i 
              << ". row: " << h << ". col: " << w;
        }
      }
    }
    bottom_data  += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
  }
// //  for debug
//   LOG(INFO) << "confusion matrix info:" << confusion_matrix_.numRows() << "," << confusion_matrix_.numCols();
//   confusion_matrix_.printCounts();
  
  // we report all the resuls
  top[0]->mutable_cpu_data()[0] = (Dtype)totalMAE / totalNum;
  top[0]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.maxFmeasure();
  top[0]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.accuracy();
  top[0]->mutable_cpu_data()[3] = (Dtype)confusion_matrix_.avgRecall(false);


}

INSTANTIATE_CLASS(ContourAccuracyLayer);
REGISTER_LAYER_CLASS(ContourAccuracy);

}  // namespace caffe
