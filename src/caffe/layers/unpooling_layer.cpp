#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UnpoolingParameter unpool_param = this->layer_param_.unpooling_param();
  CHECK(!unpool_param.has_kernel_size() !=
      !(unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(unpool_param.has_kernel_size() ||
      (unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!unpool_param.has_pad() && unpool_param.has_pad_h()
      && unpool_param.has_pad_w())
      || (!unpool_param.has_pad_h() && !unpool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!unpool_param.has_stride() && unpool_param.has_stride_h()
      && unpool_param.has_stride_w())
      || (!unpool_param.has_stride_h() && !unpool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (unpool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = unpool_param.kernel_size();
  } else {
    kernel_h_ = unpool_param.kernel_h();
    kernel_w_ = unpool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!unpool_param.has_pad_h()) {
    pad_h_ = pad_w_ = unpool_param.pad();
  } else {
    pad_h_ = unpool_param.pad_h();
    pad_w_ = unpool_param.pad_w();
  }
  if (!unpool_param.has_stride_h()) {
    stride_h_ = stride_w_ = unpool_param.stride();
  } else {
    stride_h_ = unpool_param.stride_h();
    stride_w_ = unpool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.unpooling_param().unpool()
        == UnpoolingParameter_UnpoolMethod_AVE
        || this->layer_param_.unpooling_param().unpool()
        == UnpoolingParameter_UnpoolMethod_MAX)
        << "Padding implemented only for average and max unpooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  unpooled_height_ = static_cast<int>(static_cast<float>(height_ - 1) * stride_h_ +  kernel_h_ - 2 * pad_h_);
  unpooled_width_ = static_cast<int>(static_cast<float>(width_ - 1) * stride_w_ + kernel_w_ - 2 * pad_w_);
  if (unpooled_height_%2) {
    unpooled_height_ -= 1;
  }
  if (unpooled_width_%2) {
    unpooled_width_ -= 1;
  }

  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_, unpooled_width_);
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.unpooling_param().unpool() ==
      UnpoolingParameter_UnpoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, height_, width_);
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_mask = NULL;
  const int top_count = top[0]->count();
  // We'll take as input the mask to top[1] if it's of size >1.
  const bool use_bottom_mask = bottom.size() > 1;
  if (use_bottom_mask) {
  	bottom_mask = bottom[1]->cpu_data();
  }
  // Different unpooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
    // Initialize
    caffe_set(top_count, Dtype(0), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            const int index = bottom_mask[h * width_ + w];
	    top_data[index] = bottom_data[h * width_ + w];
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_mask += bottom[1]->offset(0, 1);
      }
    }
    break;
  // TO DO: debug average unpooling
  case UnpoolingParameter_UnpoolMethod_AVE:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          for (int pw = 0; pw < width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, unpooled_height_ + pad_h_);
            int wend = min(wstart + kernel_w_, unpooled_width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, unpooled_height_);
            wend = min(wend, unpooled_width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[h * unpooled_width_ + w] +=
                  bottom_data[ph * width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        top_data += top[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_mask = NULL;
  const bool use_bottom_mask = bottom.size() > 1;
  if (use_bottom_mask) {
  	bottom_mask = bottom[1]->cpu_data();
  }
  // Different unpooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll take as the input the mask in bottom[1] if it's of size >1.
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            const int index = bottom_mask[h * width_ + w];
            bottom_diff[h * width_ + w] = top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        bottom_mask += bottom[1]->offset(0, 1);
      }
    }
    break;
  // TO DO: debug average unpooling
  case UnpoolingParameter_UnpoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          for (int pw = 0; pw < width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, unpooled_height_ + pad_h_);
            int wend = min(wstart + kernel_w_, unpooled_width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, unpooled_height_);
            wend = min(wend, unpooled_width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[ph * width_ + pw] +=
                    top_diff[h * unpooled_width_ + w];
              }
            }
            bottom_diff[ph * width_ + pw] /= pool_size;
          }
        }
        // compute offset
        top_diff += top[0]->offset(0, 1);
        bottom_diff += bottom[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
// template void UnpoolingLayer<float>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);
// template void UnpoolingLayer<double>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);
// template void UnpoolingLayer<float>::Reshape(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);
// template void UnpoolingLayer<double>::Reshape(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);

// template void UnpoolingLayer<float>::Forward_cpu( 
//       const std::vector<Blob<float>*>& bottom, 
//       vector<Blob<float>*>* top); 
// template void UnpoolingLayer<double>::Forward_cpu( 
//       const std::vector<Blob<double>*>& bottom, 
//       vector<Blob<double>*>* top);

// template void UnpoolingLayer<float>::Backward_cpu( 
//     const std::vector<Blob<float>*>& top, 
//     const std::vector<bool>& propagate_down, 
//     std::vector<Blob<float>*>* bottom); 
// template void UnpoolingLayer<double>::Backward_cpu( 
//     const std::vector<Blob<double>*>& top, 
//     const std::vector<bool>& propagate_down, 
//     std::vector<Blob<double>*>* bottom);

}  // namespace caffe
