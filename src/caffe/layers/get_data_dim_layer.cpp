#include <vector>

#include "caffe/layers/get_data_dim_layer.hpp"

namespace caffe {

template <typename Dtype>
void GetDataDimLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 1) 
    << "Only accept one bottom";
}

template <typename Dtype>
void GetDataDimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int n = bottom[0]->num();
  top[0]->Reshape(n, 1, 1, 2);
}


template <typename Dtype>
void GetDataDimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int h = bottom[0]->height();
  const int w = bottom[0]->width();
  const int n = bottom[0]->num();

  for (int i = 0; i < n; ++i) {
    int top_data_offset = top[0]->offset(i);
    top_data[top_data_offset] = h;
    top_data[top_data_offset+1] = w;
  }  
}

template <typename Dtype>
void GetDataDimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(GetDataDimLayer);
#endif

INSTANTIATE_CLASS(GetDataDimLayer);
REGISTER_LAYER_CLASS(GetDataDim);

}  // namespace caffe
