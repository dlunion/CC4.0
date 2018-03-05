# CC4.0
Caffe for CC4.0，Caffe从没如此简单，更简单的Caffe C++接口，更方便的研究深度学习

# 案例
非常容易在C++里面实现自己的datalayer、losslayer等，自定义数据的输入等
```
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