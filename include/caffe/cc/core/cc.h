/*
CC深度学习库（Caffe）V4.0
*/

#ifndef CC_H
#define CC_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
//#define USE_PROTOBUF
#ifdef USE_PROTOBUF
#include <caffe/proto/caffe.pb.h>
#endif 

using namespace std;

#ifdef EXPORT_CC_DLL
#define CCAPI __declspec(dllexport)  
#else
#define CCAPI __declspec(dllimport)  
#endif

#define CCCALL __stdcall

namespace cc{

	using cv::Mat;
	static const int PhaseTrain = 0;
	static const int PhaseTest = 1;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class CCAPI Blob{
	public:
		void setNative(void* native);
		void* getNative();
		int shape(int index) const;
		int num_axes() const;
		int count() const;
		int count(int start_axis) const;
		int height() const;
		int width() const;
		int channel() const;
		int num() const;
		int offset(int n) const;;
		void set_cpu_data(float* data);

		const float* cpu_data() const;
		const float* gpu_data() const;
		float* mutable_cpu_data();
		float* mutable_gpu_data();

		const float* cpu_diff();
		const float* gpu_diff();
		float* mutable_cpu_diff();
		float* mutable_gpu_diff();

		void Reshape(int num, int channels, int height, int width);
		void Reshape(int numShape, int* shapeDims);
		void ReshapeLike(const Blob& other);
		void copyFrom(const Blob& other, bool copyDiff = false, bool reshape = false);
		void setDataRGB(int numIndex, const Mat& data);

	private:
		void* _native;
	};

	class CCAPI Layer{
	public:
		void setNative(void* native);
		void setupLossWeights(int num, float* weights);
		float lossWeights(int index);
		void setLossWeights(int index, float weights);

#ifdef USE_PROTOBUF
		caffe::LayerParameter& layer_param();
#endif

	private:
		void* _native;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class CCAPI Net{
	public:
		void setNative(void* native);
		void* getNative();

		Blob* blob(const char* name);
		void Forward(float* loss = 0);
		void Reshape();
		void copyTrainedParamFromFile(const char* file);
		void copyTrainedParamFromData(const void* data, int length);
		void ShareTrainedLayersWith(const Net* other);
		bool has_blob(const char* name);
		bool has_layer(const char* name);
		int num_input_blobs();
		int num_output_blobs();
		Blob* input_blob(int index);
		Blob* output_blob(int index);
		Layer* layer_by_name(const char* name);

	private:
		void* _native;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class CCAPI Solver{
	public:
		Solver();
		virtual ~Solver();

		void setNative(void* native);
		void step(int iters);
		Net* net();
		int num_test_net();
		Net* test_net(int index = 0);
		void* getNative();
		int iter();
		float smooth_loss();
		void Restore(const char* resume_file);
		void Snapshot();
		int max_iter();
		void Solve();
		void installActionSignalOperator();
		void setBaseLearningRate(float rate);
		float getBaseLearningRate();
		void postSnapshotSignal();

#ifdef USE_PROTOBUF
		caffe::SolverParameter& solver_param();
#endif

	private:
		void* signalHandler_;
		void* _native;
	};

	CCAPI Blob* CCCALL newBlob();
	CCAPI Blob* CCCALL newBlobByShape(int num = 1, int channels = 1, int height = 1, int width = 1);
	CCAPI Blob* CCCALL newBlobByShapes(int numShape, int* shapes);
	CCAPI void CCCALL releaseBlob(Blob* blob);
	CCAPI void CCCALL releaseSolver(Solver* solver);
	CCAPI void CCCALL releaseNet(Net* net);
	CCAPI Solver* CCCALL loadSolverFromPrototxt(const char* solver_prototxt);

#ifdef USE_PROTOBUF
	CCAPI Solver* CCCALL newSolverFromProto(const caffe::SolverParameter* solver_param);
#endif

	CCAPI Net* CCCALL loadNetFromPrototxt(const char* net_prototxt, int phase = PhaseTest);
	CCAPI Net* CCCALL loadNetFromPrototxtString(const char* net_prototxt, int length = -1, int phase = PhaseTest);

#ifdef USE_PROTOBUF
	CCAPI Net* CCCALL newNetFromParam(const caffe::NetParameter& param);
#endif

	CCAPI void CCCALL setGPU(int id);

#ifdef USE_PROTOBUF
	CCAPI bool CCCALL ReadProtoFromTextString(const char* str, google::protobuf::Message* proto);
	CCAPI bool CCCALL ReadProtoFromData(const void* data, int length, google::protobuf::Message* proto);
	CCAPI bool CCCALL ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto);
	CCAPI bool CCCALL ReadProtoFromBinaryFile(const char* binaryfilename, google::protobuf::Message* proto);

	CCAPI void CCCALL WriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename);
	CCAPI void CCCALL WriteProtoToBinaryFile(const google::protobuf::Message& proto, const char* filename);
#endif

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class CCAPI AbstractCustomLayer{
	public:
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) = 0;
		virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop) = 0;
		virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down){};
		virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop) = 0;
		virtual const char* type() = 0;
		virtual ~AbstractCustomLayer(){}
		void* getNative();
		void setNative(void* ptr);
		Layer* ccLayer();

	private:
		void* native_;
	};

	class CCAPI AbstractCustomLayerCPP{
	public:
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, vector<Blob*>& bottom, vector<Blob*>& top) = 0;
		virtual void forward(vector<Blob*>& bottom, vector<Blob*>& top) = 0;
		virtual void backward(vector<Blob*>& bottom, const vector<bool>& propagate_down, vector<Blob*>& top) = 0;
		virtual void reshape(vector<Blob*>& bottom, vector<Blob*>& top) = 0;
		virtual const char* type() = 0;
		virtual ~AbstractCustomLayerCPP(){}
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	typedef void* CustomLayerInstance;
	typedef AbstractCustomLayer* (*createLayerFunc)();
	typedef void(*releaseLayerFunc)(AbstractCustomLayer* layer);
	typedef CustomLayerInstance(CCCALL *newLayerFunction)(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop, void* native);
	typedef void(CCCALL *customLayerForward)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop);
	typedef void(CCCALL *customLayerBackward)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	typedef void(CCCALL *customLayerReshape)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop);
	typedef void(CCCALL *customLayerRelease)(CustomLayerInstance instance);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	CCAPI void CCCALL registerLayerFunction(newLayerFunction newlayerFunc);
	CCAPI void CCCALL registerLayerForwardFunction(customLayerForward forward);
	CCAPI void CCCALL registerLayerBackwardFunction(customLayerBackward backward);
	CCAPI void CCCALL registerLayerReshapeFunction(customLayerReshape reshape);
	CCAPI void CCCALL registerLayerReleaseFunction(customLayerRelease release);
	CCAPI void CCCALL installRegister();
	CCAPI void CCCALL installLayer(const char* type, createLayerFunc func, releaseLayerFunc release);


#define INSTALL_LAYER(classes)    {installLayer(#classes, classes::creater, classes::release);};
#define SETUP_LAYERFUNC(classes)  static AbstractCustomLayer* creater(){return new classes();} static void release(AbstractCustomLayer* layer){if (layer) delete layer; };  virtual const char* type(){return #classes;}


	class CCAPI DataLayer : public AbstractCustomLayer{
	public:
		DataLayer();
		virtual ~DataLayer();

		virtual int getBatchCacheSize();
		virtual void loadBatch(Blob** top, int numTop) = 0;
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
		virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
		virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop){}
		void stopBatchLoader();
		virtual int waitForDataTime();

	private:
		void setupBatch(Blob** top, int numTop);
		static void watcher(DataLayer* ptr);
		void startWatcher();
		void stopWatcher();
		void pullBatch(Blob** top, int numTop);

	private:
		volatile bool keep_run_watcher_;
		void* hsem_;
		bool* batch_flags_;
		Blob*** batch_;
		int numTop_;
		int cacheBatchSize_;
	};

	class CCAPI SSDDataLayer : public DataLayer{
	public:
		SSDDataLayer();
		virtual ~SSDDataLayer();

		virtual int getBatchCacheSize();
		virtual void loadBatch(Blob** top, int numTop);
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
		virtual void* getAnnDatum() = 0;
		virtual void releaseAnnDatum(void* datum) = 0;

	private:
		bool has_anno_type_;
		int anno_type_;
		void* batch_samplers_;
		char label_map_file_[500];
		void* transform_param_;
		void* data_transformer_;
		Blob* transformed_data_;
	};

	CCAPI void* CCCALL createAnnDatum();
	CCAPI bool CCCALL loadAnnDatum(
		const char* filename, const char* xml, int resize_width, int resize_height,
		int min_dim, int max_dim, int is_color, const char* encode_type, const char* label_type, void* label_map, void* inplace_anndatum);
	CCAPI void* CCCALL loadLabelMap(const char* prototxt);
	CCAPI void CCCALL releaseLabelMap(void* labelmap);
	CCAPI void CCCALL releaseAnnDatum(void* datum);
};

#endif //CC_H
