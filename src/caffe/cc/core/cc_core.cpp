

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/cc/core/cc.h"
#include <map>
#include <string>
#include <import-staticlib.h>
#include <Windows.h>
#include <highgui.h>
#include <thread>
#include "caffe/layers/cpp_layer.hpp"

using namespace std;
using namespace cc;
using namespace cv;

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
	DataLayer::DataLayer(){
		this->batch_ = new Blob**[getBatchCacheSize()];
		this->batch_flags_ = new bool[getBatchCacheSize()];
		this->numTop_ = 0;
	}

	int DataLayer::getBatchCacheSize(){
		return 3;
	}

	DataLayer::~DataLayer(){
		stopWatcher();
		for (int i = 0; i < getBatchCacheSize(); ++i){
			for (int j = 0; j < this->numTop_; ++j)
				releaseBlob(batch_[i][j]);

			delete[] this->batch_[i];
		}
		delete[] this->batch_flags_;
		delete[] this->batch_;
	}

	void DataLayer::setupBatch(Blob** top, int numTop){
		this->numTop_ = numTop;
		for (int i = 0; i < getBatchCacheSize(); ++i){
			batch_[i] = new Blob*[numTop];
			for (int j = 0; j < numTop; ++j){
				batch_[i][j] = newBlobByShape(top[j]->num(), top[j]->channel(), top[j]->height(), top[j]->width());
			}
		}
	}

	void DataLayer::watcher(DataLayer* ptr){
		for (int i = 0; i < ptr->getBatchCacheSize(); ++i){
			ptr->batch_flags_[i] = true;
			ptr->loadBatch(ptr->batch_[i], ptr->numTop_);
		}

		while (ptr->keep_run_watcher_){
			for (int i = 0; i < ptr->getBatchCacheSize(); ++i){
				if (!ptr->batch_flags_[i]){
					ptr->loadBatch(ptr->batch_[i], ptr->numTop_);
					ptr->batch_flags_[i] = true;
				}
			}
			Sleep(10);
		}
		ReleaseSemaphore((HANDLE)ptr->hsem_, 1, 0);
	}

	void DataLayer::startWatcher(){
		hsem_ = (void*)CreateSemaphoreA(0, 0, 1, 0);
		keep_run_watcher_ = true;
		thread(watcher, this).detach();
	}

	void DataLayer::stopWatcher(){
		keep_run_watcher_ = false;
		WaitForSingleObject((HANDLE)hsem_, -1);
	}

	void DataLayer::pullBatch(Blob** top, int numTop){
		double tick = getTickCount();
		double prevtime = tick;
		while (true){
			for (int i = 0; i < getBatchCacheSize(); ++i){
				if (batch_flags_[i]){
					for (int j = 0; j < numTop; ++j)
						top[j]->copyFrom(*batch_[i][j], false, true);
					batch_flags_[i] = false;
					return;
				}
			}

			Sleep(10);
			float waitTime = (getTickCount() - tick) / getTickFrequency() * 1000;
			float printTime = (getTickCount() - prevtime) / getTickFrequency() * 1000;
			if (printTime > 3000){
				prevtime = tick;
				printf("wait for data: %.2f ms\n", waitTime);
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
	SSDDataLayer::SSDDataLayer(){

	}

	SSDDataLayer::~SSDDataLayer(){
		delete ((std::vector<caffe::BatchSampler>*)this->batch_samplers_);
		delete ((caffe::DataTransformer<float>*)this->data_transformer_);
		delete ((caffe::TransformationParameter*)this->transform_param_);
		releaseBlob(this->transformed_data_);
	}

	int SSDDataLayer::getBatchCacheSize(){
		return 3;
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
}