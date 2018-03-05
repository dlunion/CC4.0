# CC4.0
Caffe for CC4.0，Caffe从没如此简单，更简单的Caffe C++接口，更方便的研究深度学习

# 案例
非常容易在C++里面实现自己的datalayer、losslayer等，自定义数据的输入等
在prototxt中定义如下：
```
layer {
  name: "data"
  type: "CPP"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  cpp_param {
    type: "LstmDataLayer"
    param_str: "batch_size: 16; width: 150; height: 60; num: 6" 
  }
}
```

cpp代码的写法：
```
#include <cc_utils.h>
#pragma comment(lib, "libcaffe.lib")

//define my LstmDataLayer
class LstmDataLayer : public DataLayer{
public:
    SETUP_LAYERFUNC(LstmDataLayer);

    virtual int getBatchCacheSize(){
        return 3;
    }

    virtual void loadBatch(Blob** top, int numTop){

        Blob* image = top[0];
        Blob* label = top[1];

        float* image_ptr = image->mutable_cpu_data();
        float* label_ptr = label->mutable_cpu_data();
        int batch_size = image->num();
        int w = image->width();
        int h = image->height();

        for (int i = 0; i < batch_size; ++i){
           //...
        }
    }

    virtual void setup(
        const char* name, const char* type, const char* param_str, int phase, 
        Blob** bottom, int numBottom, Blob** top, int numTop){
        //...
    }
};

void main(){
    installRegister();

    //register LstmDataLayer
    INSTALL_LAYER(LstmDataLayer);

    WPtr<Solver> solver = loadSolverFromPrototxt("solver-gpu.prototxt");
    //solver->Restore("models/blstmctc_iter_12111.solverstate");
    solver->Solve();
}
```

前向运算
```
void test(){
    //...
    WPtr<Net> net = loadNetFromPrototxt("deploy.prototxt");
    net->copyTrainedParamFromFile("models/blstmctc_iter_6044.caffemodel");

    im.convertTo(im, CV_32F, 1/127.5, -1);
    Blob* input = net->input_blob(0);
    input->Reshape(1, 3, im.rows, im.cols);
    net->Reshape();

    Mat ms[3];
    float* ptr = input->mutable_cpu_data();
    for (int i = 0; i < 3; ++i){
        ms[i] = Mat(input->height(), input->width(), CV_32F, ptr);
        ptr += input->width()*input->height();
    }
    split(im, ms);
    net->Forward();

    Blob* out = net->output_blob(0);
    //...
    //out就是结果
}
```

SSD训练
```
#include <cc_utils.h>
using namespace cc;

class SSDMyDataLayer : public SSDDataLayer{
public:
    SETUP_LAYERFUNC(SSDMyDataLayer);

    SSDMyDataLayer(){
        this->datum_ = createAnnDatum();
        this->label_map_ = loadLabelMap("labelmap_voc.prototxt");
    }

    virtual ~SSDMyDataLayer(){
        releaseAnnDatum(this->datum_);
    }

    virtual int getBatchCacheSize(){
        return 3;
    }

    virtual void* getAnnDatum(){
        if (!loadAnnDatum("00001.jpg", "00001.xml", 0, 0, 0, 0, true, "jpg", "xml", this->label_map_, this->datum_)){
            printf("无法加载.\n");
            exit(0);
        }
        return this->datum_;
    }

    virtual void releaseAnnDatum(void* datum){
    }

private:
    void* datum_;
    void* label_map_;
};

void main(){
    installRegister();
    INSTALL_LAYER(SSDMyDataLayer);

    WPtr<Solver> solver = loadSolverFromPrototxt("solver.prototxt");
    solver->net()->copyTrainedParamFromFile("VGG_ILSVRC_16_layers_fc_reduced.caffemodel");
    //solver->Restore("models/blstmctc_iter_12111.solverstate");
    solver->Solve();
}

prototxt的data层：
layer {
  name: "data"
  type: "CPP"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  cpp_param{
    type: "SSDMyDataLayer"
  }
  transform_param {
    mirror: true
    mean_value: 104
    mean_value: 117
    mean_value: 123
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 300
      width: 300
      interp_mode: LINEAR
      interp_mode: AREA
      interp_mode: NEAREST
      interp_mode: CUBIC
      interp_mode: LANCZOS4
    }
    emit_constraint {
      emit_type: CENTER
    }
  }
  data_param {
    batch_size: 8
  }
  annotated_data_param {
    batch_sampler {
      max_sample: 1
      max_trials: 1
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.1
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.3
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.5
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.7
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.9
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        max_jaccard_overlap: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
  }
}
```