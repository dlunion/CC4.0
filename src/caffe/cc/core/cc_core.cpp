

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/cc/core/cc.h"
#include "caffe/cc/core/cc_utils.h"
#include <map>
#include <string>
#include <import-staticlib.h>
#include <Windows.h>
#include <highgui.h>
#include <thread>
#include "caffe/layers/cpp_layer.hpp"

#include <io.h>
#include <direct.h> 

#include <google/protobuf/text_format.h>
#include <io.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/strtod.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/tokenizer.h>
#include <google/protobuf/any.h>
#include <google/protobuf/stubs/stringprintf.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/map_util.h>
#include <google/protobuf/stubs/stl_util.h>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#undef GetMessage

using namespace std;
using namespace cc;
using namespace cv;
using namespace google::protobuf;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::GzipInputStream;

static struct LayerInfo{
	createLayerFunc creater;
	releaseLayerFunc release;
	AbstractCustomLayer* instance;
};

static map<string, LayerInfo> g_custom_layers;

static LayerInfo* createLayer(const char* type, void* native){
	map<string, LayerInfo>::iterator itr = g_custom_layers.find(type);
	if (itr == g_custom_layers.end()){
		LOG(FATAL) << "unknow custom layer type:" << type << ", no register.";
		return 0;
	}
	LayerInfo* layer = new LayerInfo(itr->second);
	layer->instance = layer->creater();
	layer->instance->setNative(native);
	return layer;
}

static void releaseLayer(LayerInfo* layer){
	if (layer){
		layer->release(layer->instance);
		layer->instance = 0;
		delete layer;
	}
}

static CustomLayerInstance CCCALL NewLayerFunction(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop, void* native){
	LayerInfo* layer = createLayer(type, native);
	layer->instance->setup(name, type, param_str, phase, bottom, numBottom, top, numTop);
	return layer;
}

static void CCCALL CustomLayerForward(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop){
	((LayerInfo*)instance)->instance->forward(bottom, numBottom, top, numTop);
}

static void CCCALL CustomLayerBackward(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down){
	((LayerInfo*)instance)->instance->backward(bottom, numBottom, top, numTop, propagate_down);
}

static void CCCALL CustomLayerReshape(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop){
	((LayerInfo*)instance)->instance->reshape(bottom, numBottom, top, numTop);
}

static void CCCALL CustomLayerRelease(CustomLayerInstance instance){
	releaseLayer((LayerInfo*)instance);
}

namespace cc{

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	CCString::CCString(){
		buffer = 0;
		length = 0;
		capacity_size = 0;
	}

	CCString::~CCString(){
		release();
	}

	void CCString::release(){
		if (buffer) delete[]buffer;
		length = 0;
		buffer = 0;
		capacity_size = 0;
	}

	CCString::CCString(const char* other){
		buffer = 0;
		length = 0;
		capacity_size = 0;
		set(other);
	}

	CCString::CCString(const CCString& other){
		buffer = 0;
		length = 0;
		capacity_size = 0;
		set(other.get());
	}

	char* CCString::get() const{
		return length == 0 ? "" : buffer;
	}

	CCString CCString::operator + (const CCString& str){
		CCString newstr = *this;
		newstr.append(str.c_str(), str.len());
		return newstr;
	}

	CCString CCString::operator+(const char* str){
		CCString newstr = *this;
		newstr.append(str);
		return newstr;
	}

	CCString& CCString::operator += (const CCString& str){
		append(str.c_str());
		return *this;
	}

	CCString& CCString::operator+=(const char* str){
		append(str);
		return *this;
	}

	void CCString::append(const char* str, int strlength){
		int slen = strlength < 0 ? (str ? strlen(str) : 0) : strlength;
		if (slen == 0)
			return;

		int copyoffset = this->length;
		if (this->length + slen + 1 > capacity_size){
			char* oldbuffer = buffer;
			int oldlength = this->length;
			capacity_size = (this->length + slen)*1.3 + 1;
			buffer = new char[capacity_size];

			if (oldbuffer){
				if (oldlength > 0)
					memcpy(buffer, oldbuffer, oldlength);

				delete[]oldbuffer;
				copyoffset = oldlength;
			}
		}

		length += slen;
		memcpy(buffer + copyoffset, str, slen);
		buffer[length] = 0;
	}

	void CCString::set(const char* str, int strlength){
		int slen = strlength < 0 ? (str ? strlen(str) : 0) : strlength;
		if (slen == 0){
			length = 0;
			return;
		}

		if (slen + 1 > capacity_size){
			release();
			capacity_size = slen*1.3 + 1;
			buffer = new char[capacity_size];
		}

		length = slen;
		memcpy(buffer, str, slen);
		buffer[length] = 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////

	CCAPI void CCCALL installLayer(const char* type, createLayerFunc creater, releaseLayerFunc release){
		if (g_custom_layers.find(type) != g_custom_layers.end()){
			LOG(FATAL) << "layer " << type << " already register.";
		}
		g_custom_layers[type].creater = creater;
		g_custom_layers[type].release = release;
	}

	CCAPI void CCCALL installRegister(){
		registerLayerFunction(NewLayerFunction);
		registerLayerForwardFunction(CustomLayerForward);
		registerLayerBackwardFunction(CustomLayerBackward);
		registerLayerReshapeFunction(CustomLayerReshape);
		registerLayerReleaseFunction(CustomLayerRelease);
	}

	CCAPI void CCCALL setGPU(int id){
		if (id == -1){
			caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
		}
		else{
#ifdef CPU_ONLY
			caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
#else
			caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
			caffe::Caffe::SetDevice(id);
#endif
		}
	}

	////////////////////////////////////////////////////////////////////////////////////
	typedef map<std::thread::id, int> watcher_thread_map;

	DataLayer::DataLayer(int batchCacheSize, int watcherSize){
		this->cacheBatchSize_ = batchCacheSize;
		this->cacheBatchSize_ = max(1, cacheBatchSize_);
		this->cacheBatchSize_ = min(1000, cacheBatchSize_);
		this->watcher_map_ = new watcher_thread_map();

		this->watcherSize_ = watcherSize;
		this->watcherSize_ = max(1, watcherSize_);
		this->watcherSize_ = min(100, watcherSize_);

		this->batch_ = new Blob***[watcherSize_];
		this->batch_flags_ = new bool*[watcherSize_];
		this->hsem_ = new void*[watcherSize_];
		for (int i = 0; i < watcherSize_; ++i){
			this->batch_[i] = new Blob**[cacheBatchSize_];
			this->batch_flags_[i] = new bool[cacheBatchSize_];
			this->hsem_[i] = (void*)CreateSemaphoreA(0, 0, 1, 0);

			memset(this->batch_flags_[i], 0, sizeof(bool)*cacheBatchSize_);
			memset(this->batch_[i], 0, sizeof(Blob**)*cacheBatchSize_);
		}
		this->numTop_ = 0;
		setPrintWaitData(true);
	}

	void DataLayer::stopBatchLoader(){
		stopWatcher(); 
	}

	void DataLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop){

		return;

		bool doReshape = false;
		if (cacheBatchSize_ > 0){
			for (int w = 0; w < watcherSize_; ++w){
				for (int i = 0; i < numTop; ++i){
					if (batch_[w][0][i]->count() != top[i]->count()){
						doReshape = true;
						break;
					}
				}
			}
		}

		if (doReshape){
			stopWatcher();

			for (int w = 0; w < watcherSize_; ++w){
				for (int i = 0; i < cacheBatchSize_; ++i){
					for (int j = 0; j < this->numTop_; ++j)
						batch_[w][i][j]->ReshapeLike(*top[j]);

					batch_flags_[w][i] = false;
				}
			}
			startWatcher();
		}
	}

	DataLayer::~DataLayer(){
		stopWatcher();
		for (int w = 0; w < watcherSize_; ++w){
			for (int i = 0; i < cacheBatchSize_; ++i){
				for (int j = 0; j < this->numTop_; ++j)
					releaseBlob(batch_[w][i][j]);

				delete[] this->batch_[w][i];
			}
			delete[] this->batch_[w];
			delete[] this->batch_flags_[w];
			CloseHandle(this->hsem_[w]);
		}

		delete[] this->hsem_;
		delete[] this->batch_flags_;
		delete[] this->batch_;
		if (this->watcher_map_)
			delete (watcher_thread_map*)this->watcher_map_;
	}

	void DataLayer::setupBatch(Blob** top, int numTop){
		this->numTop_ = numTop;

		for (int w = 0; w < watcherSize_; ++w){
			batch_[w] = new Blob**[cacheBatchSize_];
			batch_flags_[w] = new bool[cacheBatchSize_];
			for (int i = 0; i < cacheBatchSize_; ++i){
				batch_[w][i] = new Blob*[numTop];
				batch_flags_[w][i] = false;
				for (int j = 0; j < numTop; ++j){
					batch_[w][i][j] = newBlobByShape(top[j]->num(), top[j]->channel(), top[j]->height(), top[j]->width());
					batch_[w][i][j]->mutable_cpu_data();	//分配内存
					batch_[w][i][j]->mutable_gpu_data();	//分配内存
				}
			}
		}
	}

	int DataLayer::waitForDataTime(){
		return 1000;
	}

	void DataLayer::watcher(DataLayer* ptr, int ind){
		static int count_run = 0;
		//srand(++count_run);
		for (int i = 0; i < ptr->cacheBatchSize_; ++i){
			//cout << "loadBatch threadID: " << GetCurrentThreadId() << endl;
			ptr->loadBatch(ptr->batch_[ind][i], ptr->numTop_);
			ptr->batch_flags_[ind][i] = true;
		}

		while (ptr->keep_run_watcher_){
			for (int i = 0; i < ptr->cacheBatchSize_; ++i){
				if (!ptr->batch_flags_[ind][i]){
					//cout << "loadBatch threadID: " << GetCurrentThreadId() << endl;
					ptr->loadBatch(ptr->batch_[ind][i], ptr->numTop_);
					ptr->batch_flags_[ind][i] = true;
				}
			}
			sleep_cc(1);
		}
		ReleaseSemaphore((HANDLE)ptr->hsem_[ind], 1, 0);
	}

	void DataLayer::startWatcher(){
		keep_run_watcher_ = true;

		watcher_thread_map& threadmap = *(watcher_thread_map*)this->watcher_map_;
		threadmap.clear();

		for (int i = 0; i < watcherSize_; ++i){
			thread t(watcher, this, i);
			threadmap[t.get_id()] = i;
			t.detach();
		}
	}

	int DataLayer::getWatcherIndex(){
		std::thread::id id = this_thread::get_id();
		watcher_thread_map& threadmap = *(watcher_thread_map*)this->watcher_map_;

		watcher_thread_map::const_iterator indIter = threadmap.find(id);
		if (indIter == threadmap.end())
			return -1;

		return indIter->second;
	}

	void DataLayer::stopWatcher(){
		if (keep_run_watcher_){
			keep_run_watcher_ = false;

			for (int i = 0; i < watcherSize_; ++i)
				WaitForSingleObject((HANDLE)hsem_[i], -1);

			watcher_thread_map& threadmap = *(watcher_thread_map*)this->watcher_map_;
			threadmap.clear();
		}
	}

	void DataLayer::setPrintWaitData(bool wait){
		this->print_waitdata_ = wait;
	}
	 
	void DataLayer::pullBatch(Blob** top, int numTop){
		double tick = getTickCount();
		double prevtime = tick;
		vector<int> indWatcher(watcherSize_);
		vector<int> indCacheBatch(cacheBatchSize_);

		for (int i = 0; i < indWatcher.size(); ++i)
			indWatcher[i] = i;
		
		for (int i = 0; i < indCacheBatch.size(); ++i)
			indCacheBatch[i] = i;

		std::random_shuffle(indWatcher.begin(), indWatcher.end());
		std::random_shuffle(indCacheBatch.begin(), indCacheBatch.end());

		bool fristwait = true;
		while (true){
			for (int w = 0; w < watcherSize_; ++w){
				int indw = indWatcher[w];
				for (int i = 0; i < cacheBatchSize_; ++i){
					int indc = indCacheBatch[i];
					if (batch_flags_[indw][indc]){
						for (int j = 0; j < numTop; ++j)
							top[j]->copyFrom(*batch_[indw][indc][j], false, true);
						batch_flags_[indw][indc] = false;
						return;
					}
				}
			}

			sleep_cc(1);

			if (print_waitdata_){
				if (fristwait){
					LOG(INFO) << "wait for data.";
					fristwait = false;
				}

				float waitTime = (getTickCount() - tick) / getTickFrequency() * 1000;
				float printTime = (getTickCount() - prevtime) / getTickFrequency() * 1000;
				if (printTime > waitForDataTime()){
					prevtime = getTickCount();
					LOG(INFO) << "wait for data: " << waitTime << " ms";
				}
			}
		}
	}

	void DataLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){
		setupBatch(top, numTop);
		startWatcher();
	}

	void DataLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop){
		pullBatch(top, numTop);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void* AbstractCustomLayer::getNative(){
		return this->native_;
	}

	void AbstractCustomLayer::setNative(void* ptr){
		this->native_ = ptr;
	}

	Layer* AbstractCustomLayer::ccLayer(){
		return ((caffe::CPPLayer<float>*)this->native_)->ccLayer();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	SSDDataLayer::SSDDataLayer(int batchCacheSize, int watcherSize):
		DataLayer(batchCacheSize, watcherSize){

	}

	SSDDataLayer::~SSDDataLayer(){
		stopBatchLoader();
		delete ((std::vector<caffe::BatchSampler>*)this->batch_samplers_);
		delete ((caffe::DataTransformer<float>*)this->data_transformer_);
		delete ((caffe::TransformationParameter*)this->transform_param_);
		releaseBlob(this->transformed_data_);
	}

	int SSDDataLayer::getBatchCacheSize(){
		return 3;
	}

	int SSDDataLayer::getWatcherSize(){
		return 5;
	}

	void SSDDataLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){
		caffe::CPPLayer<float>* ptr = (caffe::CPPLayer<float>*)this->getNative();

		const int batch_size = ptr->layer_param_.data_param().batch_size();
		this->batch_samplers_ = new std::vector<caffe::BatchSampler>();
		const caffe::AnnotatedDataParameter& anno_data_param =
			ptr->layer_param_.annotated_data_param();
		for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
			(*(std::vector<caffe::BatchSampler>*)this->batch_samplers_).push_back(anno_data_param.batch_sampler(i));
		}
		//label_map_file_ = anno_data_param.label_map_file();
		strcpy(label_map_file_, anno_data_param.label_map_file().c_str());

		// Make sure dimension is consistent within batch.
		const caffe::TransformationParameter& transform_param =
			ptr->layer_param_.transform_param();

		if (transform_param.has_resize_param()) {
			if (transform_param.resize_param().resize_mode() ==
				caffe::ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
				CHECK_EQ(batch_size, 1)
					<< "Only support batch size of 1 for FIT_SMALL_SIZE.";
			}
		}

		// Read a data point, and use it to initialize the top blob.
		caffe::AnnotatedDatum& anno_datum = *((caffe::AnnotatedDatum*)getAnnDatum());
		this->transform_param_ = new caffe::TransformationParameter(ptr->layer_param_.transform_param());
		this->data_transformer_ = new caffe::DataTransformer<float>(*(caffe::TransformationParameter*)this->transform_param_, phase == PhaseTest ? caffe::Phase::TEST : caffe::Phase::TRAIN);
		((caffe::DataTransformer<float>*)this->data_transformer_)->InitRand();
		this->transformed_data_ = newBlob();

		// Use data_transformer to infer the expected blob shape from anno_datum.
		vector<int> top_shape =
			((caffe::DataTransformer<float>*)this->data_transformer_)->InferBlobShape(anno_datum.datum());
		this->transformed_data_->Reshape(top_shape.size(), &top_shape[0]);
		// Reshape top[0] and prefetch_data according to the batch_size.
		top_shape[0] = batch_size;
		top[0]->Reshape(top_shape.size(), &top_shape[0]);

		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channel() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
		vector<int> label_shape(4, 1);
		if (has_anno_type_) {
			anno_type_ = (int)anno_datum.type();
			if (anno_data_param.has_anno_type()) {
				// If anno_type is provided in AnnotatedDataParameter, replace
				// the type stored in each individual AnnotatedDatum.
				LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
				anno_type_ = (int)anno_data_param.anno_type();
			}
			// Infer the label shape from anno_datum.AnnotationGroup().
			int num_bboxes = 0;
			if (anno_type_ == (int)caffe::AnnotatedDatum_AnnotationType_BBOX) {
				// Since the number of bboxes can be different for each image,
				// we store the bbox information in a specific format. In specific:
				// All bboxes are stored in one spatial plane (num and channels are 1)
				// And each row contains one and only one box in the following format:
				// [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
				// Note: Refer to caffe.proto for details about group_label and
				// instance_id.
				for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
					num_bboxes += anno_datum.annotation_group(g).annotation_size();
				}
				label_shape[0] = 1;
				label_shape[1] = 1;
				// BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
				// cpu_data and gpu_data for consistent prefetch thread. Thus we make
				// sure there is at least one bbox.
				label_shape[2] = max(num_bboxes, 1);
				label_shape[3] = 8;
			}
			else {
				LOG(FATAL) << "Unknown annotation type.";
			}
		}
		else {
			label_shape[0] = batch_size;
		}
		top[1]->Reshape(label_shape.size(), &label_shape[0]);
		releaseAnnDatum(&anno_datum);

		DataLayer::setup(name, type, param_str, phase, bottom, numBottom, top, numTop);
	}

	void SSDDataLayer::loadBatch(Blob** top, int numTop){
		CHECK_GE(numTop, 2) << "numTop error";
		CHECK(this->transformed_data_->count());

		caffe::CPPLayer<float>* ptr = (caffe::CPPLayer<float>*)this->getNative();

		// Reshape according to the first anno_datum of each batch
		// on single input batches allows for inputs of varying dimension.
		const int batch_size = ptr->layer_param_.data_param().batch_size();
		const caffe::AnnotatedDataParameter& anno_data_param =
			ptr->layer_param_.annotated_data_param();
		const caffe::TransformationParameter& transform_param =
			ptr->layer_param_.transform_param();

		float* top_data = top[0]->mutable_cpu_data();
		float* top_label = top[1]->mutable_cpu_data();

		// Store transformed annotation.
		map<int, vector<caffe::AnnotationGroup> > all_anno;
		int num_bboxes = 0;
		Scalar mean_values;
		CHECK(transform_param.mean_value_size() <= 3);

		for (int i = 0; i < transform_param.mean_value_size(); ++i)
			mean_values[i] = transform_param.mean_value(i);

		float scale_value = transform_param.scale();
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a anno_datum
			caffe::AnnotatedDatum& anno_datum = *((caffe::AnnotatedDatum*)getAnnDatum());
			caffe::AnnotatedDatum distort_datum;
			caffe::AnnotatedDatum* expand_datum = NULL;
			if (transform_param.has_distort_param()) {
				distort_datum.CopyFrom(anno_datum);
				((caffe::DataTransformer<float>*)this->data_transformer_)->DistortImage(anno_datum.datum(),
					distort_datum.mutable_datum());
				if (transform_param.has_expand_param()) {
					expand_datum = new caffe::AnnotatedDatum();
					((caffe::DataTransformer<float>*)this->data_transformer_)->ExpandImage(distort_datum, expand_datum);
				}
				else {
					expand_datum = &distort_datum;
				}
			}
			else {
				if (transform_param.has_expand_param()) {
					expand_datum = new caffe::AnnotatedDatum();
					((caffe::DataTransformer<float>*)this->data_transformer_)->ExpandImage(anno_datum, expand_datum);
				}
				else {
					expand_datum = &anno_datum;
				}
			}
			caffe::AnnotatedDatum* sampled_datum = NULL;
			bool has_sampled = false;
			if (((std::vector<caffe::BatchSampler>*)batch_samplers_)->size() > 0) {
				// Generate sampled bboxes from expand_datum.
				vector<caffe::NormalizedBBox> sampled_bboxes;
				caffe::GenerateBatchSamples(*expand_datum, *((std::vector<caffe::BatchSampler>*)batch_samplers_), &sampled_bboxes);
				if (sampled_bboxes.size() > 0) {
					// Randomly pick a sampled bbox and crop the expand_datum.
					int rand_idx = caffe::caffe_rng_rand() % sampled_bboxes.size();
					sampled_datum = new caffe::AnnotatedDatum();
					((caffe::DataTransformer<float>*)this->data_transformer_)->CropImage(*expand_datum,
						sampled_bboxes[rand_idx],
						sampled_datum);
					has_sampled = true;
				}
				else {
					sampled_datum = expand_datum;
				}
			}
			else {
				sampled_datum = expand_datum;
			}
			CHECK(sampled_datum != NULL);
			vector<int> shape =
				((caffe::DataTransformer<float>*)this->data_transformer_)->InferBlobShape(sampled_datum->datum());
			if (transform_param.has_resize_param()) {
				if (transform_param.resize_param().resize_mode() ==
					caffe::ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
					this->transformed_data_->Reshape(shape.size(), &shape[0]);
					top[0]->Reshape(shape.size(), &shape[0]);
					top_data = top[0]->mutable_cpu_data();
				}
				else {
				}
			}
			else {
			}
			// Apply data transformations (mirror, scale, crop...)
			int offset = top[0]->offset(item_id);
			this->transformed_data_->set_cpu_data(top_data + offset);
			vector<caffe::AnnotationGroup> transformed_anno_vec;
			if (has_anno_type_) {
				// Make sure all data have same annotation type.
				CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
				if (anno_data_param.has_anno_type()) {
					sampled_datum->set_type((caffe::AnnotatedDatum_AnnotationType)anno_type_);
				}
				else {
					CHECK_EQ((caffe::AnnotatedDatum_AnnotationType)anno_type_, sampled_datum->type()) <<
						"Different AnnotationType.";
				}
				// Transform datum and annotation_group at the same time
				transformed_anno_vec.clear();
				((caffe::DataTransformer<float>*)this->data_transformer_)->Transform(*sampled_datum,
					&(*(caffe::Blob<float>*)this->transformed_data_->getNative()),
					&transformed_anno_vec);
				if (anno_type_ == caffe::AnnotatedDatum_AnnotationType_BBOX) {
					// Count the number of bboxes.
					for (int g = 0; g < transformed_anno_vec.size(); ++g) {
						num_bboxes += transformed_anno_vec[g].annotation_size();
					}
				}
				else {
					LOG(FATAL) << "Unknown annotation type.";
				}
				all_anno[item_id] = transformed_anno_vec;
			}
			else {
				((caffe::DataTransformer<float>*)this->data_transformer_)->Transform(sampled_datum->datum(),
					&(*(caffe::Blob<float>*)this->transformed_data_->getNative()));
				// Otherwise, store the label from datum.
				CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
				top_label[item_id] = sampled_datum->datum().label();
			}

			releaseAnnDatum(&anno_datum);
			// clear memory
			if (has_sampled) {
				delete sampled_datum;
			}
			if (transform_param.has_expand_param()) {
				delete expand_datum;
			}
		}

		// Store "rich" annotation if needed.
		vector<int> label_shape(4);
		if (anno_type_ == caffe::AnnotatedDatum_AnnotationType_BBOX) {
			label_shape[0] = 1;
			label_shape[1] = 1;
			label_shape[3] = 8;
			if (num_bboxes == 0) {
				// Store all -1 in the label.
				label_shape[2] = 1;
				top[1]->Reshape(label_shape.size(), &label_shape[0]);
				caffe::caffe_set<float>(8, -1, top[1]->mutable_cpu_data());
			}
			else {
				// Reshape the label and store the annotation.
				label_shape[2] = num_bboxes;
				top[1]->Reshape(label_shape.size(), &label_shape[0]);
				top_label = top[1]->mutable_cpu_data();
				int idx = 0;
				for (int item_id = 0; item_id < batch_size; ++item_id) {
					const vector<caffe::AnnotationGroup>& anno_vec = all_anno[item_id];
					for (int g = 0; g < anno_vec.size(); ++g) {
						const caffe::AnnotationGroup& anno_group = anno_vec[g];
						for (int a = 0; a < anno_group.annotation_size(); ++a) {
							const caffe::Annotation& anno = anno_group.annotation(a);
							const caffe::NormalizedBBox& bbox = anno.bbox();
							top_label[idx++] = item_id;
							top_label[idx++] = anno_group.group_label();
							top_label[idx++] = anno.instance_id();
							top_label[idx++] = bbox.xmin();
							top_label[idx++] = bbox.ymin();
							top_label[idx++] = bbox.xmax();
							top_label[idx++] = bbox.ymax();
							top_label[idx++] = bbox.difficult();
						}
					}
				}
			}
		}
		else {
			LOG(FATAL) << "Unknown annotation type.";
		}
	}

	CCAPI void* CCCALL createAnnDatum(){
		return new caffe::AnnotatedDatum();
	}

	CCAPI void CCCALL releaseAnnDatum(void* datum){
		if (datum)
			delete (caffe::AnnotatedDatum*)datum;
	}

	CCAPI bool CCCALL loadAnnDatum(
		const char* filename, const char* xml, int resize_width, int resize_height,
		int min_dim, int max_dim, int is_color, const char* encode_type, const char* label_type, void* label_map, void* inplace_anndatum)
	{
		bool status = caffe::ReadRichImageToAnnotatedDatum(filename, xml, resize_height,
			resize_width, min_dim, max_dim, is_color, encode_type, caffe::AnnotatedDatum_AnnotationType_BBOX, label_type,
			*(std::map<std::string, int>*)label_map, (caffe::AnnotatedDatum*)inplace_anndatum);
		((caffe::AnnotatedDatum*)inplace_anndatum)->set_type(caffe::AnnotatedDatum_AnnotationType_BBOX);
		return status;
	}

	CCAPI void* CCCALL loadDatum(const char* path, int label){
		caffe::Datum* d = new caffe::Datum();
		bool ok = caffe::ReadFileToDatum(path, vector<float>(1, label), d);
		if (!ok){
			delete d;
			return 0;
		}
		return d;
	}

	CCAPI void CCCALL releaseDatum(void* datum){
		delete (caffe::Datum*)datum;
	}

	CCAPI void* CCCALL loadLabelMap(const char* prototxt){
		std::map<std::string, int>* name_to_label = new std::map<std::string, int>();
		caffe::LabelMap label_map;
		CHECK(caffe::ReadProtoFromTextFile(prototxt, &label_map))
			<< "Failed to read label map file.";
		CHECK(caffe::MapNameToLabel(label_map, false, name_to_label));
		return name_to_label;
	}

	CCAPI void CCCALL releaseLabelMap(void* labelmap){
		if (labelmap)
			delete (std::map<std::string, int>*)labelmap;
	}

#ifdef USE_PROTOBUF
	CCAPI bool CCCALL ReadProtoFromTextString(const char* str, google::protobuf::Message* proto){
		return caffe::ReadProtoFromTextString(str, proto);
	}

	CCAPI bool CCCALL ReadProtoFromData(const void* data, int length, google::protobuf::Message* proto){
		return caffe::ReadProtoFromData(data, length, proto);
	}

	CCAPI bool CCCALL ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto){
		return caffe::ReadProtoFromTextFile(filename, proto);
	}

	CCAPI bool CCCALL ReadProtoFromBinaryFile(const char* binaryfilename, google::protobuf::Message* proto){
		return caffe::ReadProtoFromBinaryFile(binaryfilename, proto);
	}

	CCAPI void CCCALL WriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename){
		return caffe::WriteProtoToTextFile(proto, filename);
	}

	CCAPI void CCCALL WriteProtoToBinaryFile(const google::protobuf::Message& proto, const char* filename){
		return caffe::WriteProtoToBinaryFile(proto, filename);
	}
#endif



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum MessagePathNodeType{
		MessagePathNodeType_Key,
		MessagePathNodeType_GetElement,
		MessagePathNodeType_GetSize
	};

	struct MessagePathNode{
		string name;
		int index;			//repeated index
		MessagePathNodeType type;

		MessagePathNode(){
			index = 0;
		}
	};

	void Value::init(){
		enumIndex = 0;
		enumRepIndex = 0;
		repeated = false;
		numElements = 0;
		type = ValueType_Null;
		stringVal = 0;
	}

#define DEFCONSTRUCT(cpptype, utype)										\
	Value::Value(cpptype* repeatedValue, int length){						\
		init();																\
		repeated = true;													\
		type = ValueType_##utype;											\
		numElements = length;												\
		cpptype##RepVal = 0;												\
		if (length > 0){													\
			cpptype##RepVal = new cpptype[length];							\
			memcpy(cpptype##RepVal, repeatedValue, sizeof(cpptype)*length);	\
		}																	\
	}

	Value::Value(cc::CCString* repeatedValue, int length){
		init();																
		repeated = true;													
		type = ValueType_String;											
		numElements = length;												
		stringRepVal = 0;												
		if (length > 0){													
			stringRepVal = new cc::CCString[length];
			for (int i = 0; i < length; ++i)
				stringRepVal[i] = repeatedValue[i];
		}																	
	}

	Value::Value(cc::CCString* repeatedValue, int* enumIndex, int length){
		init();
		repeated = true;
		type = ValueType_String;
		numElements = length;
		enumRepVal = 0;
		enumRepIndex = 0;
		if (length > 0){
			enumRepVal = new cc::CCString[length];
			enumRepIndex = new int[length];
			for (int i = 0; i < length; ++i){
				enumRepVal[i] = repeatedValue[i];
				enumRepIndex[i] = enumIndex[i];
			}
		}
	}

	Value::Value(MessageHandle* repeatedValue, int length){
		init();
		repeated = true;
		type = ValueType_Message;
		numElements = length;
		messageRepVal = 0;
		if (length > 0){
			messageRepVal = new MessageHandle[length];
			for (int i = 0; i < length; ++i){
				messageRepVal[i] = repeatedValue[i];
			}
		}
	}

	DEFCONSTRUCT(float, Float);
	DEFCONSTRUCT(cint32, Int32);
	DEFCONSTRUCT(cuint32, Uint32);
	DEFCONSTRUCT(cint64, Int64);
	DEFCONSTRUCT(cuint64, Uint64);
	DEFCONSTRUCT(double, Double);
	DEFCONSTRUCT(bool, Bool);

	Value::Value(int val){ init(); int32Val = val; type = ValueType_Int32; }
	Value::Value(cuint32 val){ init(); uint32Val = val; type = ValueType_Uint32; }
	Value::Value(__int64 val){ init(); int64Val = val; type = ValueType_Int64; }
	Value::Value(cuint64 val){ init(); uint64Val = val; type = ValueType_Uint64; }
	Value::Value(float val){ init(); floatVal = val; type = ValueType_Float; }
	Value::Value(double val){ init(); doubleVal = val; type = ValueType_Double; }
	Value::Value(bool val){ init(); boolVal = val; type = ValueType_Bool; }
	Value::Value(const char* val){ init(); stringVal = new cc::CCString();  *stringVal = val; this->type = ValueType_String; }
	Value::Value(const char* enumName, int enumIndex){ init(); enumVal = new cc::CCString(); *enumVal = enumName; this->enumIndex = enumIndex; this->type = ValueType_Enum; }
	Value::Value(MessageHandle message){ init(); messageVal = message; this->type = ValueType_Message; }
	Value::Value(){ init(); }

#define ReturnConvt(strcvtmethod)									\
	if (!this->repeated){											\
	switch (type){													\
	case ValueType_Bool:   return boolVal;							\
	case ValueType_Int32:   return int32Val;						\
	case ValueType_Int64:   return int64Val;						\
	case ValueType_Float:   return floatVal;						\
	case ValueType_Double:   return doubleVal;						\
	case ValueType_Uint32:   return uint32Val;						\
	case ValueType_Uint64:   return uint64Val;						\
	case ValueType_String:  return strcvtmethod(stringVal->c_str());	\
	case ValueType_Enum:    return enumIndex;							\
	default:  return 0;	}												\
	}else{																\
		switch (type){													\
	case ValueType_Bool:   return boolRepVal[index];						\
	case ValueType_Int32:   return cint32RepVal[index];						\
	case ValueType_Int64:   return cint64RepVal[index];						\
	case ValueType_Float:   return floatRepVal[index];						\
	case ValueType_Double:   return doubleRepVal[index];					\
	case ValueType_Uint32:   return cuint32RepVal[index];					\
	case ValueType_Uint64:   return cuint64RepVal[index];					\
	case ValueType_String:  return strcvtmethod(stringRepVal[index].c_str());	\
	case ValueType_Enum:    return enumRepIndex[index];							\
	default:  return 0;												\
	}																	\
	}

	int Value::getInt(int index){ ReturnConvt(atoi); }
	cuint32 Value::getUint(int index){ ReturnConvt(atoi); }
	__int64 Value::getInt64(int index){ ReturnConvt(atoll); }
	cuint64 Value::getUint64(int index){ ReturnConvt(atoll); }
	float Value::getFloat(int index){ ReturnConvt(atof); }
	double Value::getDouble(int index){ ReturnConvt(atof); }
	cc::CCString Value::getString(int index){
		if (!this->repeated){
			char val[1000];
			switch (type){
			case ValueType_Bool:   sprintf(val, "%s", boolVal ? "true" : "false"); break;
			case ValueType_Int32:  sprintf(val, "%d", int32Val); break;
			case ValueType_Int64:  sprintf(val, "%I64d", int64Val); break;
			case ValueType_Float:  sprintf(val, "%f", floatVal); break;
			case ValueType_Double: sprintf(val, "%lf", doubleVal); break;
			case ValueType_Uint32: sprintf(val, "%u", uint32Val); break;
			case ValueType_Uint64: sprintf(val, "%uI64d", uint64Val); break;
			case ValueType_String: return stringVal ? *stringVal : ""; break;
			case ValueType_Enum: return enumVal ? *enumVal : ""; break;
			case ValueType_Message:{
				string out;
				if (messageVal)
					google::protobuf::TextFormat::PrintToString(*(Message*)messageVal, &out);
				return out.c_str();
			}
			default:  return "";
			}
			return val;
		}
		else{
			char val[1000] = {0};
			string outval;
			switch (type){
			case ValueType_Bool:   sprintf(val, "%s", boolRepVal[index] ? "true" : "false"); outval = val; break;
			case ValueType_Int32:  sprintf(val, "%d", cint32RepVal[index]); outval = val; break;
			case ValueType_Int64:  sprintf(val, "%I64d", cint64RepVal[index]); outval = val; break;
			case ValueType_Float:  sprintf(val, "%f", floatRepVal[index]); outval = val; break;
			case ValueType_Double: sprintf(val, "%lf", doubleRepVal[index]); outval = val; break;
			case ValueType_Uint32: sprintf(val, "%u", cuint32RepVal[index]); outval = val; break;
			case ValueType_Uint64: sprintf(val, "%uI64d", cuint64RepVal[index]); outval = val; break;
			case ValueType_String: outval = stringRepVal ? stringRepVal[index].c_str() : ""; break;
			case ValueType_Enum: outval = enumRepVal ? enumRepVal[index].c_str() : ""; break;
			case ValueType_Message:{
				string v;
				if (messageRepVal && messageRepVal[index])
					google::protobuf::TextFormat::PrintToString(*(Message*)messageRepVal[index], &v);
				return v.c_str();
			}
			default: 
				break;
			}
			return outval.c_str();
		}
	}

	cc::CCString Value::toString(){
		if (!this->repeated){
			return getString();
		}
		else{
			cc::CCString out;
			for (int i = 0; i < numElements; ++i){
				out += getString(i) + "\n";
			}
			return out.c_str();
		}
	}
	
	void Value::release(){
		if (!repeated){
			if (type == ValueType_String && stringVal){
				delete stringVal;
				stringVal = 0;
			}
			else if (type == ValueType_Enum && enumVal){
				delete enumVal;
				enumVal = 0;
			}
		}
		else{
#define DefDeleteListFunc(cpptype, utype)								\
			else if (type == ValueType_##utype && cpptype##RepVal){		\
				delete[] cpptype##RepVal;								\
				cpptype##RepVal = 0;									\
				type = ValueType_Null;									\
			}

			if (0){}
			DefDeleteListFunc(float, Float)
			DefDeleteListFunc(cint32, Int32)
			DefDeleteListFunc(cuint32, Uint32)
			DefDeleteListFunc(cint64, Int64)
			DefDeleteListFunc(cuint64, Uint64)
			DefDeleteListFunc(double, Double)
			DefDeleteListFunc(string, String)
			
			else if (type == ValueType_Enum && enumRepVal && enumRepIndex){
				delete[] enumRepVal;
				delete[] enumRepIndex;
				enumRepVal = 0;
				enumRepIndex = 0;
				type = ValueType_Null;
			}
			else if (type == ValueType_Message && messageRepVal){
				delete[] messageRepVal;
				messageRepVal = 0;
			}
		}
		stringVal = 0;
		type = ValueType_Null;
	}

	void Value::copyFrom(const Value& other){
		release();

		memcpy(this, &other, sizeof(Value));
		if (!other.repeated){
			this->type = other.type;
			if (this->type == ValueType_String){
				this->stringVal = new cc::CCString();
				*this->stringVal = *other.stringVal;
			}
			else if (this->type == ValueType_Enum){
				this->enumVal = new cc::CCString();
				*this->enumVal = *other.enumVal;
			}
		}
		else{
			if (other.numElements > 0){
				if (this->type == ValueType_String){
					this->stringRepVal = new cc::CCString[other.numElements];
					for (int i = 0; i < other.numElements; ++i)
						this->stringRepVal[i] = other.stringRepVal[i];
				}
				else if (this->type == ValueType_Enum){
					this->enumRepVal = new cc::CCString[other.numElements];
					for (int i = 0; i < other.numElements; ++i)
						this->enumRepVal[i] = other.enumRepVal[i];

					this->enumRepIndex = new int[other.numElements];
					memcpy(this->enumRepIndex, other.enumRepIndex, sizeof(int)*other.numElements);
				}
				else if (this->type == ValueType_Message){
					this->messageRepVal = new MessageHandle[other.numElements];
					for (int i = 0; i < other.numElements; ++i)
						this->messageRepVal[i] = other.messageRepVal[i];
				}

#define DefCopyFunc(cpptype, utype)																					\
				else if (this->type == ValueType_##utype){															\
					this->cpptype##RepVal = new cpptype[other.numElements];											\
					memcpy(this->cpptype##RepVal, other.cpptype##RepVal, sizeof(cpptype)*other.numElements);		\
				}

				DefCopyFunc(float, Float)
				DefCopyFunc(double, Double)
				DefCopyFunc(cint32, Int32)
				DefCopyFunc(cuint32, Uint32)
				DefCopyFunc(cint64, Int64)
				DefCopyFunc(cuint64, Uint64)
			}
			else{
				this->numElements = 0;
				this->stringRepVal = 0;
			}
		}
	}

	Value& Value::operator=(const Value& other){ copyFrom(other); return *this; }
	Value::Value(const Value& other){ init(); copyFrom(other); }
	Value::~Value(){ release(); }

	struct MessagePath{
		string strpath;
		vector<MessagePathNode> nodes;
	};

	void MessagePropertyList::init(){
		this->list = 0;
		this->count = 0;
		this->capacity_count = 0;
	}

	MessagePropertyList::MessagePropertyList(const MessagePropertyList& other){
		init();
		copyFrom(other);
	}

	MessagePropertyList& MessagePropertyList::operator = (const MessagePropertyList& other){
		copyFrom(other);
		return *this;
	}

	MessagePropertyList::MessagePropertyList(){
		init();
	}
	
	void MessagePropertyList::copyFrom(const MessagePropertyList& other){
		resize(other.count);

		for (int i = 0; i < other.count; ++i)
			this->list[i] = other.list[i];
	}

	void MessagePropertyList::resize(int size){
		if (size <= 0){
			this->count = 0;
			return;
		}

		if (size > this->capacity_count){
			MessageProperty* old = this->list;
			int oldCount = count;

			this->count = size;
			this->capacity_count = size;
			this->list = new MessageProperty[size];

			if (old){
				int num = min(oldCount, count);
				for (int i = 0; i < num; ++i)
					this->list[i] = old[i];

				delete[] old;
			}
		}
		else{
			this->count = size;
		}
	}

	void MessagePropertyList::release(){
		if (this->list)
			delete[] this->list;
		this->count = 0;
		this->list = 0;
		this->capacity_count = 0;
	}

	MessagePropertyList::~MessagePropertyList(){
		release();
	}

	//param@				--> getSize
	//param[0].name			--> getItem(0).name
	//param.name			--> name
	MessagePath parsePath(const char* path){
		MessagePath empty;
		if (!path) return empty;

		string s = path;
		if (s.empty()) return empty;
		MessagePath messagePath;

		char* str = &s[0];
		while (true){
			char* p = strchr(str, '.');
			if (p) *p = 0;
		
			int len = strlen(str);
			if (*str == 0 || len == 0){
				printf("path was worng: %s\n", path);
				return empty;
			}

			char* lastToken = str + len - 1;
			MessagePathNode node;
			if (*lastToken == '@'){
				node.type = MessagePathNodeType_GetSize;
				*lastToken = 0;
				node.name = str;
			}
			else if (*lastToken == ']'){
				*lastToken = 0;
				char* prevToken = strrchr(str, '[');
				if (!prevToken){
					printf("missing [\n");
					return empty;
				}

				*prevToken = 0;
				node.index = atoi(prevToken + 1);
				node.type = MessagePathNodeType_GetElement;
				node.name = str;
			}
			else{
				//normal
				node.name = str;
				node.type = MessagePathNodeType_Key;
			}

			messagePath.nodes.emplace_back(node);
			if (!p) break;
			str = p + 1;
		}

		messagePath.strpath = path;
		return messagePath;
	}

	static bool getValue2(const Message* message, MessagePath& path, int ind, Value& val){
		const Descriptor* descriptor = message->GetDescriptor();
		const Reflection* reflection = message->GetReflection();
		std::vector<const FieldDescriptor*> fields;
		reflection->ListFields(*message, &fields);

		bool lastNode = ind == (int)path.nodes.size() - 1;
		MessagePathNode& node = path.nodes[ind];
	
		for (int i = 0; i < fields.size(); ++i){
			auto item = fields[i];
			if (item->name().compare(node.name) == 0){

				if (!lastNode){
					if (node.type == MessagePathNodeType_GetElement){
						if (!item->is_repeated()){
							printf("repeated not match: %s[%d], is not repeated.\n", node.name.c_str(), node.index);
							return false;
						}

						int size = 0;
						if (item->is_repeated())
							size = reflection->FieldSize(*message, item);
						else if (reflection->HasField(*message, item))
							size = 1;

						if (node.index < 0 || node.index >= size){
							printf("index out of range node.index[=%d] < size[=%d] && node.index[=%d] >= 0.\n", node.index, size, node.index);
							return false;
						}

						const Message& msg = reflection->GetRepeatedMessage(*message, item, node.index);
						return getValue2(&msg, path, ind + 1, val);
					}
					else if (node.type == MessagePathNodeType_GetSize){
						printf("syntax error: %s\n", path.strpath.c_str());
						return false;
					}
					else{
						if (item->is_repeated()){
							printf("type not match: %s. is a repeated\n", node.name.c_str());
							return false;
						}

						const Message& msg = reflection->GetMessage(*message, item);
						return getValue2(&msg, path, ind + 1, val);
					}
				}
				else{
					int size = 0;
					if (item->is_repeated())
						size = reflection->FieldSize(*message, item);
					else if (reflection->HasField(*message, item))
						size = 1;

					//没有后续路径，根节点
					if (node.type == MessagePathNodeType_GetSize){
						val = size;
						return true;
					}

					bool isgetElement = node.type == MessagePathNodeType_GetElement;
					bool isrepeated = item->is_repeated();

					if (isgetElement && !isrepeated){
						printf("type not match: item.repeated: %s != node.type is repeated: %s\n", 
							item->is_repeated() ? "True" : "False", 
							isgetElement ? "True" : "False");
						return false;
					}
				
					if (isrepeated && isgetElement){
						if (node.index < 0 || node.index >= size){
							printf("index out of range node.index[=%d] < size[=%d] && node.index[=%d] >= 0.\n", node.index, size, node.index);
							return false;
						}
					}

	#define SetValueFromField(cpptype, method, localtype)												\
						case FieldDescriptor::CPPTYPE_##cpptype:										\
						if (isgetElement){																\
							val = reflection->GetRepeated##method(*message, item, node.index); }		\
						else{																			\
							if (isrepeated){															\
								vector<localtype> buffer(size);											\
								for (int k = 0; k < size; ++k)											\
									buffer[k] = reflection->GetRepeated##method(*message, item, k);		\
									val = size > 0 ? Value((localtype*)&buffer[0], size) : Value((localtype*)0, 0);		\
							}																			\
							else{																		\
								val = reflection->Get##method(*message, item);							\
							}																			\
						}																				\
						break;																

					switch (item->cpp_type()){
						SetValueFromField(BOOL, Bool, bool);
						SetValueFromField(INT32, Int32, cint32);
						SetValueFromField(INT64, Int64, cint64);
						SetValueFromField(UINT32, UInt32, cuint32);
						SetValueFromField(UINT64, UInt64, cuint64);
						SetValueFromField(FLOAT, Float, float);
						SetValueFromField(DOUBLE, Double, double);

					case FieldDescriptor::CPPTYPE_STRING:
					{
						string scratch;
						if (isgetElement){
							const string& value = reflection->GetRepeatedStringReference(*message, item, node.index, &scratch);
							val = value.c_str();
						}
						else{
							if (isrepeated){
								vector<cc::CCString> arr(size);
								for (int k = 0; k < size; ++k){
									const string& value = reflection->GetRepeatedStringReference(*message, item, k, &scratch);
									arr[k] = value.c_str();
								}
								val = arr.size()>0 ? Value(&arr[0], size) : Value((cc::CCString*)0, 0);
							}
							else{
								const string& value = reflection->GetStringReference(*message, item, &scratch);
								val = value.c_str();
							}
						}
						break;
					}
					case FieldDescriptor::CPPTYPE_ENUM:
					{
						string scratch;
						if (isgetElement){
							int enum_value = reflection->GetRepeatedEnumValue(*message, item, node.index);
							const EnumValueDescriptor* enum_desc =
								item->enum_type()->FindValueByNumber(enum_value);
							if (enum_desc != NULL)
								val = Value(enum_desc->name().c_str(), enum_value);
						}
						else{
							if (isrepeated){
								vector<cc::CCString> arr(size);
								vector<int> indexs(size);
								for (int k = 0; k < size; ++k){
									int enum_value = reflection->GetRepeatedEnumValue(*message, item, k);
									const EnumValueDescriptor* enum_desc =
										item->enum_type()->FindValueByNumber(enum_value);
									if (enum_desc != NULL){
										arr[k] = enum_desc->name().c_str();
										indexs[k] = enum_value;
									}
								}
								val = arr.size()>0 ? Value(&arr[0], &indexs[0], size) : Value((cc::CCString*)0, (int*)0, 0);
							}
							else{
								int enum_value = reflection->GetEnumValue(*message, item);
								const EnumValueDescriptor* enum_desc =
									item->enum_type()->FindValueByNumber(enum_value);
								if (enum_desc != NULL)
									val = Value(enum_desc->name().c_str(), enum_value);
							}
						}
						break;
					}
					case FieldDescriptor::CPPTYPE_MESSAGE:
					{
						if (isgetElement){
							const Message& msg = reflection->GetRepeatedMessage(*message, item, node.index);
							val = Value((void*)&msg);
						}
						else{
							if (isrepeated){
								vector<MessageHandle> handles(size);
								for (int k = 0; k < size; ++k){
									const Message& msg = reflection->GetRepeatedMessage(*message, item, k);
									handles[k] = &msg;
								}
								val = size > 0 ? Value(&handles[0], size) : Value((MessageHandle*)0, 0);
							}
							else{
								const Message& msg = reflection->GetMessage(*message, item);
								val = Value((void*)&msg);
							}
						}
						break;
					}
					default:

						if (item->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE)
							printf("unsupport field type: %d(CPPTYPE_MESSAGE), is a layer node: %s\n", item->cpp_type(), item->name().c_str());
						else
							printf("unsupport field type: %d, %s\n", item->cpp_type(), item->name().c_str());
						return false;
					}
					return true;
				}
			}
		}

		//printf("not found path: %s\n", path.strpath.c_str());
		return false;
	}

	enum CppType {
		CPPTYPE_INT32 = 1,     // TYPE_INT32, TYPE_SINT32, TYPE_SFIXED32
		CPPTYPE_INT64 = 2,     // TYPE_INT64, TYPE_SINT64, TYPE_SFIXED64
		CPPTYPE_UINT32 = 3,     // TYPE_UINT32, TYPE_FIXED32
		CPPTYPE_UINT64 = 4,     // TYPE_UINT64, TYPE_FIXED64
		CPPTYPE_DOUBLE = 5,     // TYPE_DOUBLE
		CPPTYPE_FLOAT = 6,     // TYPE_FLOAT
		CPPTYPE_BOOL = 7,     // TYPE_BOOL
		CPPTYPE_ENUM = 8,     // TYPE_ENUM
		CPPTYPE_STRING = 9,     // TYPE_STRING, TYPE_BYTES
		CPPTYPE_MESSAGE = 10,    // TYPE_MESSAGE, TYPE_GROUP

		MAX_CPPTYPE = 10,    // Constant useful for defining lookup tables
		// indexed by CppType.
	};

	static ValueType cvtCPPType(FieldDescriptor::CppType type){
		switch (type){
		case CPPTYPE_INT32:   return ValueType_Int32;
		case CPPTYPE_INT64:   return ValueType_Int64;
		case CPPTYPE_UINT32:   return ValueType_Uint32;
		case CPPTYPE_UINT64:   return ValueType_Uint64;
		case CPPTYPE_DOUBLE:   return ValueType_Double;
		case CPPTYPE_FLOAT:   return ValueType_Float;
		case CPPTYPE_BOOL:   return ValueType_Bool;
		case CPPTYPE_ENUM:   return ValueType_Enum;
		case CPPTYPE_STRING:   return ValueType_String;
		case CPPTYPE_MESSAGE:   return ValueType_Message;
		default: return ValueType_Null;
		}
	}

	CCAPI MessagePropertyList CCCALL listProperty(MessageHandle message_){
		MessagePropertyList out;
		if (!message_) return out;

		const Message* message = (const Message*)message_;
		const Descriptor* descriptor = message->GetDescriptor();
		const Reflection* reflection = message->GetReflection();
		std::vector<const FieldDescriptor*> fields;
		reflection->ListFields(*message, &fields);

		if (fields.size() == 0)
			return out;

		out.resize(fields.size());
		for (int i = 0; i < fields.size(); ++i){
			int size = 0;
			if (fields[i]->is_repeated())
				size = reflection->FieldSize(*message, fields[i]);
			else if (reflection->HasField(*message, fields[i]))
				size = 1;

			MessageProperty& mp = out.list[i];
			mp.name = fields[i]->name().c_str();
			mp.count = size;
			mp.type = cvtCPPType(fields[i]->cpp_type());
		}
		return out;
	}

	CCAPI bool CCCALL getMessageValue(MessageHandle message, const char* pathOfGet, Value& val){
		val.release();
		if (!message || !pathOfGet) return false;

		MessagePath path = parsePath(pathOfGet);
		if (path.nodes.empty())
			return false;

		return getValue2((const Message*)message, path, 0, val);
	}

	static bool _localReadProtoFromTextFile(const char* filename, Message* proto) {
		int fd = open(filename, O_RDONLY);
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		close(fd);
		return success;
	}

	CCAPI MessageHandle CCCALL loadMessageSolverFromPrototxt(const char* filename) {

		caffe::SolverParameter* solver = new caffe::SolverParameter();
		bool success = cc::_localReadProtoFromTextFile(filename, solver);

		if (success) return solver;
		delete solver;
		return 0;
	}

	CCAPI MessageHandle CCCALL loadMessageNetFromPrototxt(const char* filename) {
		caffe::NetParameter* net = new caffe::NetParameter();
		bool success = cc::_localReadProtoFromTextFile(filename, net);

		if (success) return net;
		delete net;
		return 0;
	}

	static bool _localReadProtoFromBinaryFile(const char* filename, Message* proto) {
		int fd = open(filename, O_RDONLY | O_BINARY);
		if (fd == -1) return false;

		bool success = false;
		{
			ZeroCopyInputStream* raw_input = new FileInputStream(fd);
			CodedInputStream* coded_input = new CodedInputStream(raw_input);
			coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

			success = proto->ParseFromCodedStream(coded_input);

			delete coded_input;
			delete raw_input;
		}
		close(fd);
		return success;
	}

	CCAPI MessageHandle CCCALL loadMessageNetCaffemodel(const char* filename) {
		caffe::NetParameter* net = new caffe::NetParameter();
		bool success = cc::_localReadProtoFromBinaryFile(filename, net);
		if (success) return net;
		delete net;
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	static struct LMDB_Native{
		caffe::db::DB* db;
		caffe::db::Transaction* transaction;
		int count;
	};

	#ifdef WIN32
	#define ACCESS(fileName,accessMode) _access(fileName,accessMode)
	#define MKDIR(path) _mkdir(path)
	#else
	#define ACCESS(fileName,accessMode) access(fileName,accessMode)
	#define MKDIR(path) mkdir(path,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
	#endif

	// 从左到右依次判断文件夹是否存在,不存在就创建
	// example: /home/root/mkdir/1/2/3/4/
	// 注意:最后一个如果是文件夹的话,需要加上 '\' 或者 '/'
	int32_t createDirectory(const std::string &directoryPath){
		uint32_t dirPathLen = directoryPath.length();
		if (dirPathLen > 256)
			return -1;
		
		char tmpDirPath[256] = { 0 };
		for (uint32_t i = 0; i < dirPathLen; ++i){
			tmpDirPath[i] = directoryPath[i];
			if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/'){
				if (ACCESS(tmpDirPath, 0) != 0){
					int32_t ret = MKDIR(tmpDirPath);
					if (ret != 0)
						return ret;
				}
			}
		}
		return MKDIR(directoryPath.c_str());
	}

	LMDB::LMDB(const char* folder){
		createDirectory(folder);

		LMDB_Native* native = new LMDB_Native();
		this->native_ = native;

		native->db = caffe::db::GetDB("lmdb");
		native->db->Open(folder, caffe::db::NEW);
		native->transaction = native->db->NewTransaction();
		native->count = 0;
	}

	void LMDB::put(const char* key, const void* data, int length){
		LMDB_Native* native = (LMDB_Native*)this->native_;
		native->transaction->Put(key, string((char*)data, (char*)data + length));
		native->count++;

		if (native->count % 1000 == 0){
			LOG(INFO) << "process " << native->count << " puts.";
			native->transaction->Commit();
			delete native->transaction;
			native->transaction = native->db->NewTransaction();
		}
	}
	 
	void LMDB::putDatum(const char* key, void* datum){
		caffe::Datum* anndatum = (caffe::Datum*)datum;
		LMDB_Native* native = (LMDB_Native*)this->native_;
		string str;
		if (!anndatum->SerializeToString(&str)){
			LOG(INFO) << "SerializeToString fail[" << key << "]";
		}
		else{
			put(key, str.c_str(), str.size());
		}
	}

	void LMDB::putAnnotatedDatum(const char* key, void* datum){
		caffe::AnnotatedDatum* anndatum = (caffe::AnnotatedDatum*)datum;
		LMDB_Native* native = (LMDB_Native*)this->native_;
		string str;
		if (!anndatum->SerializeToString(&str)){
			LOG(INFO) << "SerializeToString fail[" << key << "]";
		}
		else{
			put(key, str.c_str(), str.size());
		}
	}

	LMDB::~LMDB(){
		release();
	}

	void LMDB::release(){
		if (this->native_){
			LMDB_Native* native = (LMDB_Native*)this->native_;
			if (native->count % 1000 != 0)
				native->transaction->Commit();

			native->db->Close();
			delete native->transaction;
			delete native->db;
			delete native;
			this->native_ = 0;
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}