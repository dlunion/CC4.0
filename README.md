# CC4.0
Caffe for CC4.0-Windows，简单的Caffe C++接口，方便简单而更深入的研究深度学习

# 特性
1.只需要一个头文件和一个依赖项libcaffe.lib<br/>
2.能够轻易使用C++写训练过程或调用过程<br/>
3.能够轻易自定义layer（不用编译caffe也不用修改caffe.proto，只修改代码即可使用）、自己实现数据层，不需要lmdb也能高效率训练<br/>
4.能够在训练过程中对自定义layer进行调试查看中间结果<br/>
5.支持LSTM不定长OCR（有案例），支持SSD更轻易的训练起来<br/>
6.有了4.0的支持，很轻易的能够实现任何新的网络结构<br/>
7.可以允许通过自定义层，训练中查看训练效果，更加容易理解CNN在干嘛，学的效果怎么样，不再盲目了<br/>

# 编译
编译环境：VS2013<br/>
CUDA版本：8.0<br/>
CUDNN版本：5.0<br/>
只需要下载3rd目录下的下载地址，解压出来后。安装完cuda8.0即可编译<br/>
如果不想自己编译可以下载下面已经编译好的库文件即可，库文件里面包含了CUDA8.0的下载地址
和所有要用到的工具等的下载地址或文件，直接vs打开后即可编译。编译时请选择ReleaseDLL<br/>

# 下载编译好的库文件和案例等数据
推荐使用VS2013，下载后压缩包已经配置好环境和带好了OpenCV2.4.10静态库<br/>
<del>[CC4.0.3.rar-百度网盘](https://pan.baidu.com/s/1OQDmxWwVpVohER2YMqGbZQ)</del>，里面的依赖可以用，但是头文件和libcaffe.dll不可用（因为有几个bug），等待重新编译并上传

# 案例
非常容易在C++里面实现自己的datalayer、losslayer等，自定义数据的输入等
在prototxt中定义如下：
``` protobuf
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

# cpp代码训练：
``` c++
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

# 前向运算
``` c++
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

# SSD的一步训练
``` c++
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
```

# SSD的train.prototxt的data层：
``` protobuf
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
    }
    emit_constraint {
      emit_type: CENTER
    }
  }
  #... 参考标准SSD的数据层部分即可，主要修改了type和cpp_param
}
```
