

#include "caffe/cc/core/cc.h"
#include "caffe/blob.hpp"
#include <vector>
using namespace std;

namespace cc{

#define cvt(p)	((caffe::Blob<float>*)p)
#define ptr		(cvt(this->_native))

	CCAPI Blob* CCCALL newBlob(){
		caffe::Blob<float>* blob = new caffe::Blob<float>();
		return blob->ccBlob();
	}

	CCAPI void CCCALL releaseBlob(Blob* blob){
		if (blob){
			caffe::Blob<float>* p = cvt(blob->getNative());
			if (p) delete p;
		}
	}

	CCAPI Blob* CCCALL newBlobByShape(int num, int channels, int height, int width){
		caffe::Blob<float>* blob = new caffe::Blob<float>(num, channels, height, width);
		return blob->ccBlob();
	}

	CCAPI Blob* CCCALL newBlobByShapes(int numShape, int* shapes){
		caffe::Blob<float>* blob = new caffe::Blob<float>(vector<int>(shapes, shapes+numShape));
		return blob->ccBlob();
	}

	void Blob::setDataRGB(int numIndex, const Mat& data){
		CHECK(!data.empty()) << "data is empty";
		//CHECK_EQ(CV_MAT_DEPTH(data.type()), CV_32F) << "data type not match.";
		CHECK_EQ(data.channels(), this->channel()) << "data channel error";

		int w = this->width();
		int h = this->height();
		Mat udata = data;
		if (udata.size() != cv::Size(w, h))
			resize(udata, udata, cv::Size(w, h));

		if (CV_MAT_DEPTH(udata.type()) != CV_32F)
			udata.convertTo(udata, CV_32F);

		int channel_size = w*h;
		int num_size = this->channel() * channel_size;
		float* input_data = this->mutable_cpu_data() + num_size * numIndex;
		vector<cv::Mat> mats(data.channels());
		for (int i = 0; i < mats.size(); ++i)
			mats[i] = cv::Mat(h, w, CV_32F, input_data + channel_size * i);

		split(udata, mats);
		CHECK_EQ((float*)mats[0].data, input_data) << "error, split pointer fail.";
	}

	int Blob::shape(int index) const {
		return ptr->shape(index);
	}

	int Blob::num_axes() const {
		return ptr->num_axes();
	}

	int Blob::offset(int n) const{
		return ptr->offset(n);
	}

	void Blob::set_cpu_data(float* data){
		ptr->set_cpu_data(data);
	}

	void Blob::setNative(void* native){
		this->_native = native;
	}

	void* Blob::getNative(){
		return this->_native;
	}

	const float* Blob::cpu_data() const{
		return ptr->cpu_data();
	}

	const float* Blob::gpu_data() const{
		return ptr->gpu_data();
	}

	int Blob::count() const {
		return ptr->count();
	}

	CCString Blob::shapeString(){
		char buf[100];
		sprintf(buf, "%dx%dx%dx%d", num(), channel(), height(), width());
		return buf;
	}

	int Blob::count(int start_axis) const {
		return ptr->count(start_axis);
	}

	void Blob::Reshape(int num, int channels, int height, int width){
		num = num == -1 ? ptr->num() : num;
		channels = channels == -1 ? ptr->channels() : channels;
		height = height == -1 ? ptr->height() : height;
		width = width == -1 ? ptr->width() : width;
		ptr->Reshape(num, channels, height, width);
	}

	void Blob::Reshape(int numShape, int* shapeDims){
		ptr->Reshape(vector<int>(shapeDims, shapeDims + numShape));
	}

	void Blob::ReshapeLike(const Blob& other){
		ptr->ReshapeLike(*cvt(other._native));
	}

	void Blob::copyFrom(const Blob& other, bool copyDiff, bool reshape){
		ptr->CopyFrom(*cvt(other._native), copyDiff, reshape);
	}

	void Blob::copyFrom(const BlobData& other){
		if (ptr->num() != other.num || ptr->channels() != other.channels || ptr->width() != other.width || ptr->height() != other.height){
			ptr->Reshape(other.num, other.channels, other.height, other.width);
		}

		if (other.count() > 0){
			memcpy(ptr->mutable_cpu_data(), other.list, sizeof(float)*other.count());
		}
	}

	float* Blob::mutable_cpu_data(){
		return ptr->mutable_cpu_data();
	}

	float* Blob::mutable_gpu_data(){
		return ptr->mutable_gpu_data();
	}

	const float* Blob::cpu_diff(){
		return ptr->cpu_diff();
	}

	const float* Blob::gpu_diff(){
		return ptr->gpu_diff();
	}

	float* Blob::mutable_cpu_diff(){
		return ptr->mutable_cpu_diff();
	}

	float* Blob::mutable_gpu_diff(){
		return ptr->mutable_gpu_diff();
	}

	int Blob::height() const {
		return ptr->height();
	}

	int Blob::width() const {
		return ptr->width();
	}

	int Blob::channel() const{
		return ptr->channels();
	}

	int Blob::num() const {
		return ptr->num();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BlobData::BlobData()
		:list(0), num(0), height(0), width(0), channels(0), capacity_count(0)
	{}

	BlobData::~BlobData(){
		release();
	}

	bool BlobData::empty() const{
		return count() < 1;
	}

	int BlobData::count() const{
		return num*height*width*channels;
	}

	void BlobData::reshape(int num, int channels, int height, int width){
		this->num = num;
		this->channels = channels;
		this->height = height;
		this->width = width;

		if (this->capacity_count < this->count()){
			if (this->list)
				delete[] this->list;

			this->list = this->count() > 0 ? new float[this->count()] : 0;
			this->capacity_count = this->count();
		}
	}

	void BlobData::copyFrom(const Blob* other){
		reshapeLike(other);
		if (other->count() > 0){
			memcpy(this->list, other->cpu_data(), this->count()*sizeof(float));
		}
	}

	void BlobData::reshapeLike(const Blob* other){
		reshape(other->num(), other->channel(), other->height(), other->width());
	}

	void BlobData::reshapeLike(const BlobData* other){
		reshape(other->num, other->channels, other->height, other->width);
	}

	void BlobData::copyFrom(const BlobData* other){
		reshapeLike(other);
		if (other->count() > 0){
			memcpy(this->list, other->list, this->count()*sizeof(float));
		}
	}

	void BlobData::release(){
		if (list){
			delete[]list;
			list = 0;
		}
	}

}