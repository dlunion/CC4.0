#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/ssd_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/pa_file.h"
#include "caffe/util/xml/tinyxml.h"
#include "caffe/util/ssd_augmented.hpp"
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

using namespace cv;
using namespace std;
#define max(a, b)		((a)>(b)?(a):(b))

char *strlwr(char *s)
{
	char *str;
	str = s;
	while (*str != '\0')
	{
		if (*str >= 'A' && *str <= 'Z') {
			*str += 'a' - 'A';
		}
		str++;
	}
	return s;
}

template <typename Dtype>
SSDDataLayer<Dtype>::~SSDDataLayer<Dtype>() {
  this->StopInternalThread();
}

static Rect loadbbox(const string& xmlfile, string& class_name, bool* fatal){
	TiXmlDocument doc;
	if (doc.LoadFile(xmlfile.c_str())){
		TiXmlHandle docH(&doc);
		TiXmlHandle object = docH.FirstChildElement("annotation").FirstChildElement("object");
		TiXmlNode* node = object.Node();

		if (node){
			TiXmlElement* item = node->ToElement();
			string name = item->FirstChildElement("name")->GetText();
			class_name = name;

			TiXmlElement* bndbox = item->FirstChildElement("bndbox");
			int xmin = atoi(bndbox->FirstChildElement("xmin")->GetText());
			int ymin = atoi(bndbox->FirstChildElement("ymin")->GetText());
			int xmax = atoi(bndbox->FirstChildElement("xmax")->GetText());
			int ymax = atoi(bndbox->FirstChildElement("ymax")->GetText());

			//has next
			if (node->NextSibling()){
				LOG(WARNING) << "xml(" << xmlfile << "), too maye object node(must one object).";
				if (fatal) *fatal = true;
			}

			return Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
		}
		LOG(WARNING) << " xml(" << xmlfile << "), bbox is empty";
	}
	else{
		LOG(WARNING) << "can't parse xml(" << doc.ErrorDesc() << "): " << xmlfile;
	}

	if (fatal) *fatal = true;
	return Rect();
}

static vector<string> loadLabels(const string& file){
	ifstream inf(file, ios::in);
	string line;
	vector<string> out;
	while (std::getline(inf, line)){
		strlwr((char*)line.c_str());
		out.emplace_back(line);
	}
	return out;
}

static bool has_take_negitive(Size imsize, Rect bbox, Size minsize){
	int remain_width = max(bbox.x, imsize.width - (bbox.width + bbox.x));
	int remain_height = max(bbox.y, imsize.height - (bbox.height + bbox.y));
	return remain_width > minsize.width && remain_height > minsize.height;
}

template <typename Dtype>
void SSDDataLayer<Dtype>::shuffle_images(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(positive_order_list_.begin(), positive_order_list_.end(), prefetch_rng);
	shuffle(negitive_order_list_.begin(), negitive_order_list_.end(), prefetch_rng);
}

template <typename Dtype>
void SSDDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SSDDataParameter& ssd_param = this->layer_param_.ssd_data_param();
	const int height = ssd_param.height();
	const int width = ssd_param.width();
	const bool is_color = ssd_param.is_color();
	string folder = ssd_param.folder();
	string labelmap_file = folder + "/" + ssd_param.labelmap_file();
	const int batch_size = ssd_param.batch_size();
	const int mean_value_count = ssd_param.mean_value_size();
	const int minsize_width = ssd_param.minsize_width();
	const int minsize_height = ssd_param.minsize_height();
	Size minsize(minsize_width, minsize_height);
	const float pixal_scale = ssd_param.scale();
	const int top_channels = is_color ? 3 : 1;
	const char* file_filter = "*.jpg;*.png;*.bmp;*.tif;*.jpeg";
	vector<string> def_labels = loadLabels(labelmap_file);
	map<string, int> def_labelmap;
	const string background_name = "background";

    CHECK(height > 0 && width > 0) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
	
	CHECK_EQ(mean_value_count, top_channels) << "Mean value must match channels(is_color == true ? 3 : 1)";
	CHECK_GT(batch_size, 0);

	for (int i = 0; i < mean_value_count; ++i)
		mean_values_[i] = ssd_param.mean_value(i);

	LOG(INFO) << "verifity labels file:" << labelmap_file;
	for (int i = 0; i < def_labels.size(); ++i){
		CHECK_NE(def_labels[i].compare(background_name), 0) << "labels name can't equal 'background'";
		def_labelmap[def_labels[i]] = i + 1;
	}

	LOG(INFO) << "load files from: " << folder;

	PaVfiles vds;
	paFindFilesShort(folder, vds, "*", false, true, PaFindFileType_Directory);
	//vds.erase(vds.begin() + 2, vds.end());

	label_map_.clear();
	positive_items_.clear();
	positive_order_list_.clear();
	negitive_items_.clear();
	negitive_order_list_.clear();
	label_map_[background_name] = 0;

	//order by digits
	std::sort(vds.begin(), vds.end(), [](const string& a, const string& b){
		auto to_digits = [](const string& v){
			const char* ptr = v.c_str();
			while (*ptr){
				if (*ptr >= '0' && *ptr <= '9'){
					return atoi(ptr);
				}
				ptr++;
			}
			return 0;
		};

		return to_digits(a) < to_digits(b);
	});

	LOG(INFO) << "find and check image/xml..., class_num: " << vds.size();
	int label_index = def_labelmap.size() + 1;
	bool fatal = false;
	int num_fatal = 0;
	vector<string> labelmap_vec = def_labels;
	bool has_new_labelmap = false;

	for (int n_directory = 0; n_directory < vds.size(); ++n_directory){
		//make lower case
		string lwr_name = vds[n_directory];
		strlwr((char*)lwr_name.c_str());

		const string& class_name = lwr_name;
		bool is_background = lwr_name.compare(background_name) == 0;
		vector<string> files;
		bool new_label = true;
		int use_label = label_index;

		if (!is_background){
			if (def_labelmap.find(lwr_name) != def_labelmap.end()){
				use_label = def_labelmap[lwr_name];
				label_map_[lwr_name] = use_label;
				new_label = false;
			}
			else{
				label_map_[lwr_name] = label_index;
			}
		}

		paFindFiles(folder + "/" + vds[n_directory], files, file_filter, is_background, false);
		//files.erase(files.begin() + 10, files.end());
		if (!is_background){
			for (int j = 0; j < files.size(); ++j){
				int p = files[j].rfind('.');
				string xml;
				ItemInfo item;
				string read_class_name;

				if (p == -1){
					LOG(WARNING) << "error file path: " << files[j];
					goto fatal_code;
				}

				if (paGetFileSize(files[j].c_str()) < 1000){
					LOG(WARNING) << "invalid image file: " << files[j];
					goto fatal_code;
				}

				files[j][p] = 0;
				xml = files[j].c_str() + string(".xml");
				files[j][p] = '.';

				if (!paFileExists(xml.c_str())){
					LOG(WARNING) << "can't found xml: " << xml;
					goto fatal_code;
				}

				bool fatal_local = false;
				item.bbox = loadbbox(xml, read_class_name, &fatal_local);

				if (fatal_local)
					goto fatal_code;

				if (read_class_name.compare(class_name) != 0){
					LOG(WARNING) << "xml class missmatch(file class: " << read_class_name << ", expect class: " << class_name <<"), "<< xml;
					goto fatal_code;
				}
				item.image = files[j];
				item.xml = xml;
				item.label = use_label;
				positive_items_.emplace_back(item);
				continue;

			fatal_code:
				fatal = true;
				num_fatal++;
			}
		}

		if (!is_background){
			if (new_label){
				has_new_labelmap = true;
				labelmap_vec.push_back(lwr_name);
				label_index++;
			}
		}
		else{
			for (int j = 0; j < files.size(); ++j){
				ItemInfo item;
				if (paGetFileSize(files[j].c_str()) < 1000){
					LOG(WARNING) << "invalid image file: " << files[j];
					goto fatal2_code;
				}

				item.label = 0;
				item.image = files[j];
				negitive_items_.emplace_back(item);
				continue;

			fatal2_code:
				fatal = true;
				num_fatal++;
			}
		}
	}


	if (fatal){
		LOG(INFO) << "foud error: " << num_fatal;
	}

	LOG(INFO) << "positive_items_ count:" << positive_items_.size();
	LOG(INFO) << "negitive_items_ count:" << negitive_items_.size();
	CHECK_GT(positive_items_.size(), 0) << "positive items must > 0";

	if (has_new_labelmap){
		FILE* f = !labelmap_file.empty() ? fopen(labelmap_file.c_str(), "wb") : 0;
		LOG(INFO) << "labelmap_vec size: " << labelmap_vec.size();
		for (int i = 0; i < labelmap_vec.size(); ++i){
			LOG(INFO) << labelmap_vec[i];

			if (f) fprintf(f, "%s\n", labelmap_vec[i].c_str());
		}
		if (f) fclose(f);
	}

	positive_order_list_.resize(positive_items_.size());
	for (int i = 0; i < positive_order_list_.size(); ++i)
		positive_order_list_[i] = i;

	negitive_order_list_.resize(negitive_items_.size());
	for (int i = 0; i < negitive_order_list_.size(); ++i)
		negitive_order_list_[i] = i;

	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	shuffle_images();

	vector<int> top_shape = { batch_size, top_channels, height, width };
	top[0]->Reshape(top_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}

	LOG(INFO) << "output data size: " << top[0]->num() << ","
		<< top[0]->channels() << "," << top[0]->height() << ","
		<< top[0]->width();

	vector<int> label_shape = { 1, 1, 1, 8 };
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(label_shape);
	}
}

// This function is called on prefetch thread
template <typename Dtype>
void SSDDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

	const SSDDataParameter& ssd_param = this->layer_param_.ssd_data_param();

	//正负比例，正样本为1的时候，负样本为多少
	const int positive_ratio_count = ssd_param.positive_ratio();
	const int negitive_ratio_count = ssd_param.negitive_ratio();
	const int all_count = positive_ratio_count + negitive_ratio_count;
	const float positive_ratio = positive_ratio_count / (float)all_count;
	const float negitive_ratio = negitive_ratio_count / (float)all_count;
	const float max_pos_size_acc = ssd_param.max_pos_acc();
	const float min_pos_size = ssd_param.min_pos_size();
	const bool debug_image = ssd_param.debug_image();
	const int minsize_width = ssd_param.minsize_width();
	const int minsize_height = ssd_param.minsize_height();
	Size minsize(minsize_width, minsize_height);
	const int height = ssd_param.height();
	const int width = ssd_param.width();
	const bool is_color = ssd_param.is_color();
	const int batch_size = ssd_param.batch_size();
	const int mean_value_count = ssd_param.mean_value_size();
	const float pixal_scale = ssd_param.scale();
	const int top_channels = is_color ? 3 : 1;
	const int background_label = 0;
	const int batch_num_positive = batch_size * positive_ratio;
	const int batch_num_negitive = batch_size - batch_num_positive;

	CHECK_EQ(batch_num_positive + batch_num_negitive, batch_size);
	CHECK_GT(batch_num_negitive, 0);
	CHECK_GT(batch_num_positive, 0);

	batch->data_.Reshape({ batch_size, top_channels, height, width });
	batch->label_.Reshape({ 1, 1, batch_num_positive, 8 });

	//从列表中选取相对应的图片
	vector<ItemInfo> batch_samples;
	vector<bool> batch_samples_flags;
	for (int i = 0; i < batch_num_positive; ++i){
		int sel_item_index = this->positive_order_list_[this->positive_line_id_];
		batch_samples.push_back(this->positive_items_[sel_item_index]);
		batch_samples_flags.push_back(true);

		this->positive_line_id_++;
		if (this->positive_line_id_ == this->positive_items_.size()){
			caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
			shuffle(positive_order_list_.begin(), positive_order_list_.end(), prefetch_rng);
			this->positive_line_id_ = 0;
		}
	}

	for (int i = 0; i < batch_num_negitive; ++i){

		//0.33的几率选择一个正样本，作为负样本（即不使用bbox，使用其背景）
		bool is_ok = false;

		do{
			int sel_item_index = this->negitive_order_list_[this->negitive_line_id_];
			Mat im = imread(this->negitive_items_[sel_item_index].image, is_color);
			int nw = minsize.width;
			int nh = nw * height / (float)width;
			if (im.cols > nw && im.rows > nh){
				batch_samples.push_back(this->negitive_items_[sel_item_index]);
				batch_samples_flags.push_back(false);
				is_ok = true;
			}

			this->negitive_line_id_++;
			if (this->negitive_line_id_ == this->negitive_items_.size()){
				caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
				shuffle(negitive_order_list_.begin(), negitive_order_list_.end(), prefetch_rng);
				this->negitive_line_id_ = 0;
			}
		} while (!is_ok);
	}

	CHECK_EQ(batch_samples.size(), batch_size);

	vector<int> order_index(batch_samples.size());
	vector<Mat> ms(batch_samples.size());
	vector<Rect> bboxs(batch_samples.size());
	for (int i = 0; i < batch_samples.size(); ++i){
		ms[i] = imread(batch_samples[i].image, is_color);
		bboxs[i] = batch_samples[i].bbox;
		order_index[i] = i;
	}
	
	vector<Rect> sampler_bbox;
	vector<Mat> sampler_ims;

	double tick = getTickCount();
	samplerBatch(ms, Size(width, height), bboxs, batch_samples_flags, sampler_ims, sampler_bbox, min_pos_size, max_pos_size_acc);
	tick = (getTickCount() - tick) / cv::getTickFrequency() * 1000;
	if (debug_image){
		LOG(INFO) << "samplerBatch fee: " << tick << " ms";
	}

	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(order_index.begin(), order_index.end(), prefetch_rng);

	Mat floatM;
	Dtype* top_data = batch->data_.mutable_cpu_data();
	Dtype* top_label = batch->label_.mutable_cpu_data();
	Dtype* ptr_data = top_data;
	int label_index = 0;
	//caffe_set<Dtype>(batch->label_.count(), -1, batch->label_.mutable_cpu_data());

	for (int i = 0; i < batch_size; ++i){
		int sel_index = order_index[i];
		Mat ms = sampler_ims[sel_index];
		int label = batch_samples_flags[sel_index] ? batch_samples[sel_index].label : background_label;
		Rect bbox = sampler_bbox[sel_index];
		float xmin = bbox.x / (float)ms.cols;
		float ymin = bbox.y / (float)ms.rows;
		float xmax = (bbox.x + bbox.width - 1) / (float)ms.cols;
		float ymax = (bbox.y + bbox.height - 1) / (float)ms.rows;

		ms.convertTo(floatM, CV_32F);
		if (debug_image){
			putText(ms, format("label: %d", label), Point(10, 30), 1, 1, Scalar(0, 0, 255));
			rectangle(ms, bbox, Scalar(0, 255), 2);

			char name[100];
			sprintf(name, "%d", i);
			imshow(name, ms);
		}

		if(mean_value_count > 0)
			floatM -= this->mean_values_;

		if (pixal_scale != 1)
			floatM *= pixal_scale;

		vector<Mat> vec_m(top_channels);
		for (int j = 0; j < top_channels; ++j){
			vec_m[j] = Mat(height, width, CV_32F, (void*)ptr_data);
			ptr_data += width * height;
		}

		split(floatM, vec_m);
		CHECK_EQ((Dtype*)vec_m[0].data, top_data + i * top_channels * height * width);

		if (batch_samples_flags[sel_index]){
			top_label[label_index++] = i;
			top_label[label_index++] = label;
			top_label[label_index++] = 0;
			top_label[label_index++] = xmin;
			top_label[label_index++] = ymin;
			top_label[label_index++] = xmax;
			top_label[label_index++] = ymax;
			top_label[label_index++] = 0;
		}
	}

	if (debug_image)
		waitKey(1);	
}

INSTANTIATE_CLASS(SSDDataLayer);
REGISTER_LAYER_CLASS(SSDData);

}  // namespace caffe
#endif  // USE_OPENCV
