#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/data_augmented.hpp"
#include <cv.h>
#include <vector>
using namespace cv;
using namespace std;

namespace caffe {

	template <typename Dtype>
	DataLayer<Dtype>::DataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param),
		reader_(param) {
		}

	template <typename Dtype>
	DataLayer<Dtype>::~DataLayer() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int batch_size = this->layer_param_.data_param().batch_size();
		// Read a data point, and use it to initialize the top blob.
		Datum& datum = *(reader_.full().peek());

		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
		this->transformed_data_.Reshape(top_shape);
		// Reshape top[0] and prefetch_data according to the batch_size.
		top_shape[0] = batch_size;
		top[0]->Reshape(top_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		if (this->output_labels_) {
			vector<int> label_shape(2, batch_size);
			label_shape[1] = datum.labels_size() == 0 ? 1 : datum.labels_size();

			top[1]->Reshape(label_shape);
			//top[1]->Reshape(batch_size, 4, 1, 1);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].label_.Reshape(label_shape);
				//this->prefetch_[i].label_.Reshape(batch_size, 4, 1, 1);
			}
		}
	}

	// This function is called on prefetch thread
	template<typename Dtype>
	void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());

		// Reshape according to the first datum of each batch
		// on single input batches allows for inputs of varying dimension.
		const int batch_size = this->layer_param_.data_param().batch_size();
		bool global_augmented = this->layer_param_.data_param().global_augmented();
		Datum& datum = *(reader_.full().peek());
		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* top_data = batch->data_.mutable_cpu_data();
		Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

		if (this->output_labels_) {
			top_label = batch->label_.mutable_cpu_data();
		}
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			timer.Start();
			// get a datum
			Datum& datum = *(reader_.full().pop("Waiting for data"));
			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply data transformations (mirror, scale, crop...)
			int offset = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset);
			this->data_transformer_->Transform(datum, &(this->transformed_data_));
			int imw = this->transformed_data_.width();
			int imh = this->transformed_data_.height();

			if(global_augmented){
				vector<Mat> ms(this->transformed_data_.channels());

				for (int m = 0; m < ms.size(); ++m){
					Dtype* ptr = top_data + imw * imh * m + item_id * imw * imh * this->transformed_data_.channels();
					ms[m] = Mat(imh, imw, CV_32F, ptr);
				}
				Mat mm;
				merge(ms, mm);

				//imshow("augment-before", mm);
				GlobalAugmented::augment(mm, -1, 1);
				//imshow("augment-after", mm);
				split(mm, ms); 
				//waitKey(0);
				for (int f = 0; f < ms.size(); ++f){
					CV_Assert((Dtype*)ms[f].data == top_data + imw * imh * f + item_id * imw * imh * this->transformed_data_.channels());
				}
			}
#if 0
			if (imw == 368 && imh == 368 && this->transformed_data_.channels() == 3){
				Mat ms[3];
				for (int m = 0; m < 3; ++m){
					Dtype* ptr = top_data + imw * imh * m + item_id * imw * imh * 3;
					ms[m] = Mat(imh, imw, CV_32F, ptr);
				}
				Mat mm;
				merge(ms, 3, mm);
				aug(mm);
				split(mm, ms);
				CV_Assert((Dtype*)ms[0].data == top_data + imw * imh * 0 + item_id * imw * imh * 3);
				CV_Assert((Dtype*)ms[1].data == top_data + imw * imh * 1 + item_id * imw * imh * 3);
				CV_Assert((Dtype*)ms[2].data == top_data + imw * imh * 2 + item_id * imw * imh * 3);
			}
#endif

			// Copy label.
			if (this->output_labels_) {
				//printf("datum.labels_size() = %d\n", datum.labels_size());
				//printf("label: ");
				if (datum.labels_size() > 0){
					for (int labi = 0; labi < datum.labels_size(); ++labi){
						top_label[item_id*datum.labels_size() + labi] = datum.labels(labi);
						//top_label[item_id*4 + labi] = datum.labels(labi);
						//printf("%f ", datum.labels(labi));
					}
				}
				else{
					top_label[item_id] = datum.label();
				}
				//printf("\n");

				//top_label[item_id] = datum.label();
			}
			trans_time += timer.MicroSeconds();

			reader_.free().push(const_cast<Datum*>(&datum));
		}
		timer.Stop();
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(DataLayer);
	REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
