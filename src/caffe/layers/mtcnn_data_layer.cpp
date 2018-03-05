#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/mtcnn_data_gen.hpp"

namespace caffe {

	template <typename Dtype>
	MTCNNDataLayer<Dtype>::MTCNNDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param),
		reader_(param) {
			
		}

	template <typename Dtype>
	MTCNNDataLayer<Dtype>::~MTCNNDataLayer() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void MTCNNDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Read a data point, and use it to initialize the top blob.
		MTCNNDatum& datum = *(reader_.full().peek());
		this->output_labels_ = top.size() > 1;
		this->output_roi_ = top.size() > 2;
		this->output_pts_ = top.size() > 3;

		int batch_size = this->layer_param_.data_param().batch_size();
		int num_positive = this->layer_param_.mtcnn_data_param().num_positive();
		int num_negitive = this->layer_param_.mtcnn_data_param().num_negitive();
		int num_part = this->layer_param_.mtcnn_data_param().num_part();
		bool augmented = this->layer_param_.mtcnn_data_param().augmented();
		int resize_width = this->layer_param_.mtcnn_data_param().resize_width();
		int resize_height = this->layer_param_.mtcnn_data_param().resize_height();

		num_positive = num_positive == -1 ? batch_size : num_positive;
		num_negitive = num_negitive == -1 ? batch_size : num_negitive;
		num_part = num_part == -1 ? batch_size : num_part;
		batch_size = num_positive + num_negitive + num_part;
		
		//data, label, roi, pts
		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
		raw_image_shape_ = top_shape;
		//保证转换器的原图尺寸
		this->transformed_data_.Reshape(raw_image_shape_);
		// Reshape top[0] and prefetch_data according to the batch_size.

		//小图的尺寸
		top_shape[0] = batch_size;
		top_shape[2] = resize_height;
		top_shape[3] = resize_width;

		//data
		top[0]->Reshape(top_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();

		// label
		vector<int> label_shape(2);
		label_shape[0] = batch_size;
		label_shape[1] = datum.datum().labels_size() == 0 ? 1 : datum.datum().labels_size();
		top[1]->Reshape(label_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(label_shape);
		}

		//roi
		vector<int> roi_shape(2);
		roi_shape[0] = batch_size;
		roi_shape[1] = 4;
		top[2]->Reshape(roi_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].roi_.Reshape(roi_shape);
		}

		//pts
		if (output_pts_){
			vector<int> pts_shape(2);
			pts_shape[0] = batch_size;
			pts_shape[1] = datum.pts_size();
			top[3]->Reshape(pts_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].pts_.Reshape(pts_shape);
			}
		}
	}

	template <typename Dtype>
	void MTCNNDataLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
		// Reshape to loaded data.
		top[0]->ReshapeLike(batch->data_);
		// Copy the data
		caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
			top[0]->mutable_cpu_data());
		DLOG(INFO) << "Prefetch copied";

		//label
		top[1]->ReshapeLike(batch->label_);
		caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
			top[1]->mutable_cpu_data());

		//roi
		top[2]->ReshapeLike(batch->roi_);
		caffe_copy(batch->roi_.count(), batch->roi_.cpu_data(),
			top[2]->mutable_cpu_data());

		if (output_pts_){
			//pts
			top[3]->ReshapeLike(batch->pts_);
			caffe_copy(batch->pts_.count(), batch->pts_.cpu_data(),
				top[3]->mutable_cpu_data());
		}

		prefetch_free_.push(batch);
	}

	// This function is called on prefetch thread
	template<typename Dtype>
	void MTCNNDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());

		// Reshape according to the first datum of each batch
		// on single input batches allows for inputs of varying dimension.
		const int raw_batch_size = this->layer_param_.data_param().batch_size();
		int num_positive = this->layer_param_.mtcnn_data_param().num_positive();
		int num_negitive = this->layer_param_.mtcnn_data_param().num_negitive();
		int num_part = this->layer_param_.mtcnn_data_param().num_part();
		bool augmented = this->layer_param_.mtcnn_data_param().augmented();
		bool fliped = this->layer_param_.mtcnn_data_param().flip();
		int resize_width = this->layer_param_.mtcnn_data_param().resize_width();
		int resize_height = this->layer_param_.mtcnn_data_param().resize_height();
		float min_negitive_scale = this->layer_param_.mtcnn_data_param().min_negitive_scale();
		float max_negitive_scale = this->layer_param_.mtcnn_data_param().max_negitive_scale();

		num_positive = num_positive == -1 ? raw_batch_size : num_positive;
		num_negitive = num_negitive == -1 ? raw_batch_size : num_negitive;
		num_part = num_part == -1 ? raw_batch_size : num_part;
		const int batch_size = num_positive + num_negitive + num_part;

		MTCNNDatum& datum = *(reader_.full().peek());
		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		top_shape[2] = resize_height;
		top_shape[3] = resize_width;
		batch->data_.Reshape(top_shape);

		//为大图准备一个缓冲区，该缓冲区的图，以平面放的，需要变换成分通道
		Mat buffer(raw_image_shape_[2], raw_image_shape_[3], CV_32FC(raw_image_shape_[1]));
		Dtype* top_data = batch->data_.mutable_cpu_data();
		Dtype* top_label = batch->label_.mutable_cpu_data();
		Dtype* top_roi = batch->roi_.mutable_cpu_data();

		vector<Mat> ims;
		vector<vector<Rect>> bboxs;
		//Dtype* top_pts = output_pts_ ? batch->pts_.mutable_cpu_data() : 0;
		for (int item_id = 0; item_id < raw_batch_size; ++item_id) {
			timer.Start();
			// get a datum
			MTCNNDatum& datum = *(reader_.full().pop("Waiting for data"));
			const Datum& data = datum.datum();
			//printf("****************datum.pts_size() = %d\n", datum.pts_size());  

			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply data transformations (mirror, scale, crop...)
			this->transformed_data_.set_cpu_data((Dtype*)buffer.data);
			this->data_transformer_->Transform(datum.datum(), &(this->transformed_data_));
			//原图转换完毕了
			if (buffer.channels() > 1){
				vector<Mat> chs;
				Mat mergeImage;
				for (int k = 0; k < buffer.channels(); ++k){
					chs.push_back(Mat(buffer.rows, buffer.cols, CV_32F, (char*)((float*)buffer.data + buffer.rows * buffer.cols)));
				}
				merge(chs, mergeImage);
				ims.push_back(mergeImage);
			}
			else{
				ims.push_back(buffer.clone());
			}

			vector<Rect> boxs;
			for (int k = 0; k < datum.rois_size(); ++k){
				Rect box(
					datum.rois(k).xmin(), datum.rois(k).ymin(),
					datum.rois(k).xmax() - datum.rois(k).xmin() + 1,
					datum.rois(k).ymax() - datum.rois(k).ymin() + 1);
				boxs.push_back(box);
			}
			bboxs.push_back(boxs);
			trans_time += timer.MicroSeconds();
			reader_.free().push(const_cast<MTCNNDatum*>(&datum));
		}

		vector<SampleInfo> samples;
		genSamples(ims, bboxs, Size(resize_width, resize_height), num_positive, num_negitive, num_part, samples, min_negitive_scale, max_negitive_scale, augmented, fliped);

		for (int i = 0; i < samples.size(); ++i){
			int offset = batch->data_.offset(i);
			Dtype* image = top_data + offset;

			//分通道存
			if(samples[i].im.channels() > 1){
				Dtype* ptr = image;
				Mat ch;
				for (int k = 0; k < samples[i].im.channels(); ++k){
					cv::extractChannel(samples[i].im, ch, k);
					memcpy(ptr, ch.data, ch.rows*ch.cols*sizeof(Dtype));
					ptr += ch.rows*ch.cols;
				}
			}
			else{
				memcpy(image, samples[i].im.data, samples[i].im.rows*samples[i].im.cols*sizeof(Dtype));
			}

			top_label[i] = samples[i].label;
			top_roi[i * 4 + 0] = samples[i].offx1;
			top_roi[i * 4 + 1] = samples[i].offy1;
			top_roi[i * 4 + 2] = samples[i].offx2;
			top_roi[i * 4 + 3] = samples[i].offy2;
		}
#if 0
		Dtype* top_data = batch->data_.mutable_cpu_data();
		Dtype* top_label = batch->label_.mutable_cpu_data();
		Dtype* top_roi = batch->roi_.mutable_cpu_data();
		//Dtype* top_pts = output_pts_ ? batch->pts_.mutable_cpu_data() : 0;
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			timer.Start();
			// get a datum
			MTCNNDatum& datum = *(reader_.full().pop("Waiting for data"));
			const Datum& data = datum.datum();
			//printf("****************datum.pts_size() = %d\n", datum.pts_size());  

			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply data transformations (mirror, scale, crop...)
			int offset = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset);
			this->data_transformer_->Transform(datum.datum(), &(this->transformed_data_));
			// Copy label.
			if (data.labels_size() > 0){
				for (int labi = 0; labi < data.labels_size(); ++labi)
					top_label[item_id*data.labels_size() + labi] = data.labels(labi);
			}
			else{
				top_label[item_id] = data.label();
			}

			//copy roi
			//for (int roii = 0; roii < datum.ro)
			top_roi[item_id * 4 + 0] = datum.roi().xmin();
			top_roi[item_id * 4 + 1] = datum.roi().ymin();
			top_roi[item_id * 4 + 2] = datum.roi().xmax();
			top_roi[item_id * 4 + 3] = datum.roi().ymax();

			//printf("datum.pts_size() = %d\n", datum.pts_size());
			//copy pts
			if (output_pts_){
				for (int ptsi = 0; ptsi < datum.pts_size(); ++ptsi)
					top_pts[item_id * datum.pts_size() + ptsi] = datum.pts(ptsi);
			}

			trans_time += timer.MicroSeconds();
			reader_.free().push(const_cast<MTCNNDatum*>(&datum));
		}
#endif
		timer.Stop();
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}
#ifdef CPU_ONLY
	STUB_GPU(MTCNNDataLayer);
#endif

	INSTANTIATE_CLASS(MTCNNDataLayer);
	REGISTER_LAYER_CLASS(MTCNNData);

}  // namespace caffe
