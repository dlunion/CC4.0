

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
		CHECK_EQ(CV_MAT_DEPTH(data.type()), CV_32F) << "data type not match.";
		CHECK_EQ(data.channels(), this->channel()) << "data channel error";

		int w = this->width();
		int h = this->height();
		Mat udata = data;
		if (udata.size() != cv::Size(w, h))
			resize(udata, udata, cv::Size(w, h));

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

	int Blob::count(int start_axis) const {
		return ptr->count(start_axis);
	}

	void Blob::Reshape(int num, int channels, int height, int width){
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
}
