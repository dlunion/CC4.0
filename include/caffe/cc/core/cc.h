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

	class CCAPI CCString{
	private:
		char* buffer;
		int capacity_size;
		int length;

	public:
		operator char*(){ return get(); }
		operator const char*(){ return get(); }
		bool operator==(const char* str){ return strcmp(get(), str) == 0; }
		CCString& operator=(const char* str){ set(str); return *this; }
		CCString& operator=(char* str){ set(str); return *this; }
		CCString& operator=(const CCString& str){ set(str.get(), str.len()); return *this; }
		CCString& operator+=(const CCString& str);
		CCString& operator+=(const char* str);
		CCString operator+(const CCString& str);
		CCString operator+(const char* str);
		CCString(const char* other);
		CCString(const CCString& other);
		CCString();
		virtual ~CCString();
		void set(const char* str, int len = -1);
		char* get() const;
		const char* c_str() const{ return get(); };
		int len() const{ return length; }
		void release();
		void append(const char* str, int len = -1);
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum ValueType{
		ValueType_Null,
		ValueType_Int32,
		ValueType_Int64,
		ValueType_String,
		ValueType_Float,
		ValueType_Double,
		ValueType_Bool,
		ValueType_Uint32,
		ValueType_Uint64,
		ValueType_Enum,
		ValueType_Message
	};

	struct CCAPI MessageProperty{
		cc::CCString name;
		ValueType type;
		int count;
	};

	struct CCAPI MessagePropertyList{
		MessageProperty* list;
		int count, capacity_count;

		void init();
		MessagePropertyList();
		MessagePropertyList(const MessagePropertyList& other);
		MessagePropertyList& operator=(const MessagePropertyList& other);
		void resize(int size);
		void copyFrom(const MessagePropertyList& other);
		void release();
		virtual ~MessagePropertyList();
	};

	typedef const void* MessageHandle;
	typedef int cint32;
	typedef __int64 cint64;
	typedef unsigned int cuint32;
	typedef unsigned __int64 cuint64;

	struct CCAPI Value{
		union {
			cint32 int32Val;
			cint64 int64Val;
			cc::CCString* stringVal;
			float floatVal;
			double doubleVal;
			cuint32 uint32Val;
			cuint64 uint64Val;
			bool boolVal;
			cc::CCString* enumVal;
			MessageHandle messageVal;

			//repeated
			float* floatRepVal;
			cint32* cint32RepVal;
			cuint32* cuint32RepVal;
			cint64* cint64RepVal;
			cuint64* cuint64RepVal;
			double* doubleRepVal;
			bool* boolRepVal;
			cc::CCString* stringRepVal;
			cc::CCString* enumRepVal;
			MessageHandle* messageRepVal;
		};

		//for enum type
		int enumIndex;
		int* enumRepIndex;

		ValueType type;
		bool repeated;		//对于基本元素，是否为重复的，如果是，则推广为指针
		int numElements;

		void init();
		Value(cc::CCString* repeatedValue, int length);
		Value(cc::CCString* repeatedValue, int* enumIndex, int length);
		Value(MessageHandle* repeatedValue, int length);
		Value(float* repeatedValue, int length);
		Value(cint32* repeatedValue, int length);
		Value(cuint32* repeatedValue, int length);
		Value(cint64* repeatedValue, int length);
		Value(cuint64* repeatedValue, int length);
		Value(double* repeatedValue, int length);
		Value(bool* repeatedValue, int length);

		Value(int val);
		Value(cuint32 val);
		Value(cint64 val);
		Value(cuint64 val);
		Value(float val);
		Value(double val);
		Value(bool val);
		Value(const char* stringVal);
		Value(const char* enumName, int enumIndex);
		Value(MessageHandle message);
		Value();

		cint32 getInt(int index = 0);
		cuint32 getUint(int index = 0);
		cint64 getInt64(int index = 0);
		cuint64 getUint64(int index = 0);
		float getFloat(int index = 0);
		double getDouble(int index = 0);
		cc::CCString getString(int index = 0);
		cc::CCString toString();
		void release();
		void copyFrom(const Value& other);
		Value& operator=(const Value& other);
		Value(const Value& other);
		virtual ~Value();
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//轻量级的Blob
	class Blob;
	struct CCAPI BlobData{
		float* list;
		int num;
		int channels;
		int height;
		int width;
		int capacity_count;		//保留空间的元素个数长度，字节数请 * sizeof(float)

		BlobData();
		virtual ~BlobData();
		bool empty() const;
		int count() const;
		void reshape(int num, int channels, int height, int width);
		void reshapeLike(const BlobData* other);
		void copyFrom(const BlobData* other);
		void copyFrom(const Blob* other);
		void reshapeLike(const Blob* other);
		void release();
	};

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

		void Reshape(int num = 1, int channels = 1, int height = 1, int width = 1);
		void Reshape(int numShape, int* shapeDims);
		void ReshapeLike(const Blob& other);
		void copyFrom(const Blob& other, bool copyDiff = false, bool reshape = false);
		void copyFrom(const BlobData& other);
		void setDataRGB(int numIndex, const Mat& data);
		CCString shapeString();

	private:
		void* _native;
	};

	class CCAPI Layer{
	public:
		void setNative(void* native);
		void setupLossWeights(int num, float* weights);
		float lossWeights(int index);
		void setLossWeights(int index, float weights);
		const char* type() const;
		CCString paramString();
		bool getParam(const char* path, Value& val);
		bool hasParam(const char* path);
		CCString name();
		MessageHandle* param();
		int getNumBottom();
		int getNumTop();
		CCString bottomName(int index);
		CCString topName(int index);
		Blob* paramBlob(int index);
		int getNumParamBlob();

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
		Blob* blob(int index);
		void Forward(float* loss = 0);
		void Reshape();
		void copyTrainedParamFromFile(const char* file);
		void copyTrainedParamFromData(const void* data, int length);
		void ShareTrainedLayersWith(const Net* other);
		bool has_blob(const char* name);
		bool has_layer(const char* name);
		int num_input_blobs();
		int num_output_blobs();
		int num_blobs();
		CCString blob_name(int index);
		CCString layer_name(int index);
		Blob* input_blob(int index);
		Blob* output_blob(int index);
		int num_layers();
		Layer* layer(const char* name);
		Layer* layer(int index);
		size_t memory_used();

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
		void Restore(const char* solvestate_file);
		void Snapshot(const char* caffemodel_savepath = 0, bool save_solver_state = true);
		int max_iter();
		void Solve();
		void installActionSignalOperator();
		void setBaseLearningRate(float rate);
		float getBaseLearningRate();
		void postSnapshotSignal();
		void TestAll();
		bool getParam(const char* path, Value& val);
		MessageHandle param();
		
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
	CCAPI Solver* CCCALL loadSolverFromPrototxt(const char* solver_prototxt, const char* netstring = 0);
	CCAPI Solver* CCCALL loadSolverFromPrototxtString(const char* solver_prototxt_string, const char* netstring = 0);

#ifdef USE_CC_PYTHON
	class CCAPI CCPython{
	public:
		CCPython();
		virtual ~CCPython();
		bool load(const char* pyfile);
		CCString callstringFunction(const CCString& name, CCString& errmsg = CCString());
		CCString train_ptototxt();
		CCString deploy_prototxt();
		CCString solver();
		CCString last_error();

	private:
		void* module_;
		CCString lasterror_;
	};

	CCAPI Solver* CCCALL loadSolverFromPython(const char* pythonfile);

	//phase指定加载train_prototxt还是deploy_prototxt
	CCAPI Net* CCCALL loadNetFromPython(const char* pythonfile, const char* func="deploy_prototxt", int phase = PhaseTest);
#endif

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
	class CCAPI LMDB{
	public:
		LMDB(const char* folder);
		void put(const char* key, const void* data, int length);
		void putAnnotatedDatum(const char* key, void* datum);
		void putDatum(const char* key, void* datum);
		void release();
		virtual ~LMDB();

	private:
		void* native_;
	};

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
		DataLayer(int batchCacheSize = 3, int watcherSize = 1);
		virtual ~DataLayer();

		virtual void loadBatch(Blob** top, int numTop) = 0;
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
		virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
		virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);
		void stopBatchLoader();
		int getWatcherIndex();
		virtual int waitForDataTime();
		void setPrintWaitData(bool wait);

	private:
		void setupBatch(Blob** top, int numTop);
		static void watcher(DataLayer* ptr, int ind);
		void startWatcher();
		void stopWatcher();
		void pullBatch(Blob** top, int numTop);

	private:
		volatile bool keep_run_watcher_;
		void** hsem_;
		bool** batch_flags_;
		Blob**** batch_;					//batch_
		int numTop_;
		int cacheBatchSize_;
		int watcherSize_;
		void* watcher_map_;
		bool print_waitdata_;
	};

	class CCAPI SSDDataLayer : public DataLayer{
	public:
		SSDDataLayer(int batchCacheSize = 3, int watcherSize = 1);
		virtual ~SSDDataLayer();

		virtual int getBatchCacheSize();
		virtual int getWatcherSize();
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

	CCAPI void* CCCALL loadDatum(const char* path, int label);
	CCAPI void CCCALL releaseDatum(void* datum);


	/////////////////////////////////////////////////////////////////////////////
	CCAPI MessageHandle CCCALL loadMessageNetCaffemodel(const char* filename);
	CCAPI MessageHandle CCCALL loadMessageNetFromPrototxt(const char* filename);
	CCAPI MessageHandle CCCALL loadMessageSolverFromPrototxt(const char* filename);
	CCAPI bool CCCALL getMessageValue(MessageHandle message, const char* pathOfGet, Value& val);
	CCAPI MessagePropertyList CCCALL listProperty(MessageHandle message_);

};

#endif //CC_H