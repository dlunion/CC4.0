
#include "caffe/net.hpp"
#include "caffe/cc/core/cc_utils.h"
#include <map>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "caffe/util/model_crypt.h"

using namespace std;
using namespace cc;

#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))

namespace cc{

#define CLASSIFIER_MODEL_FROM_DATMODEL					0
#define CLASSIFIER_MODEL_FROM_PROTOTXT_CAFFEMODEL		1

	CCAPI Classifier* CCCALL loadClassifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID){
		return new Classifier(prototxt, caffemodel, scale, numMeans, meanValue, gpuID);
	}

	CCAPI Classifier* CCCALL loadClassifier2(const char* datmodel, float scale, int numMeans, float* meanValue, int gpuID){
		return new Classifier(datmodel, scale, numMeans, meanValue, gpuID);
	}

	CCAPI void CCCALL releaseClassifier(Classifier* clas){
		if (clas) delete clas;
	}

	Classifier::Classifier(const char* datmodel, float scale, int numMeans, float* meanValue, int gpuID){
		this->contextInited_ = false;
		this->datmodel_ = datmodel;
		this->scale_ = scale;
		this->num_mean_ = numMeans;
		this->gpuID_ = gpuID;
		this->modelFrom_ = CLASSIFIER_MODEL_FROM_DATMODEL;

		memset(this->mean_, 0, sizeof(this->mean_));

		for (int i = 0; i < min(numMeans, 3); ++i)
			this->mean_[i] = meanValue[i];
	}

	Classifier::Classifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID){
		this->contextInited_ = false;
		this->prototxt_ = prototxt;
		this->caffemodel_ = caffemodel;
		this->scale_ = scale;
		this->num_mean_ = numMeans;
		this->gpuID_ = gpuID;
		this->modelFrom_ = CLASSIFIER_MODEL_FROM_PROTOTXT_CAFFEMODEL;
		memset(this->mean_, 0, sizeof(this->mean_));

		for (int i = 0; i < min(numMeans, 3); ++i)
			this->mean_[i] = meanValue[i];
	}

	bool Classifier::isInitContext(){
		return this->contextInited_;
	}

	Blob* Classifier::inputBlob(int index){
		return this->net_->input_blob(index);
	}

	Blob* Classifier::outputBlob(int index){
		return this->net_->output_blob(index);
	}

	void Classifier::initContext(){
		if (this->contextInited_) return;
		this->contextInited_ = true;

		if (this->modelFrom_ == CLASSIFIER_MODEL_FROM_DATMODEL){
			PackageDecode decoder;
			if (!decoder.decode(this->datmodel_)){
				throw "无法解析模型文件: " + string(this->datmodel_);
			}

			int indDeploy = decoder.find(PART_TYPE_DEPLOY);
			int indCaffemodel = decoder.find(PART_TYPE_CAFFEMODEL);

			if (indDeploy == -1){
				throw "模型没有有效的deploy数据";
			}

			if (indCaffemodel == -1){
				throw "模型没有有效的caffemodel数据";
			}

			uchar* deploy = decoder.data(indDeploy);
			uchar* caffemodel = decoder.data(indCaffemodel);
			this->net_ = loadNetFromPrototxtString((char*)deploy, decoder.size(indDeploy), PhaseTest);
			this->net_->copyTrainedParamFromData(caffemodel, decoder.size(indCaffemodel));
		}
		else if (this->modelFrom_ == CLASSIFIER_MODEL_FROM_PROTOTXT_CAFFEMODEL){
			this->net_ = loadNetFromPrototxt(this->prototxt_, PhaseTest);
			this->net_->copyTrainedParamFromFile(this->caffemodel_);
		}
		else{
			throw "错误的modelFrom_";
		}
	}

	void Classifier::reshape(int num, int channels, int height, int width){
		Blob* input = net_->input_blob(0);
		num = num == -1 ? input->num() : num;
		channels = channels == -1 ? input->channel() : channels;
		height = height == -1 ? input->height() : height;
		width = width == -1 ? input->width() : width;

		input->Reshape(num, channels, height, width);
		net_->Reshape();
	}

	Classifier::~Classifier(){
	}

	void Classifier::forward(const Mat& im){
		forward(1, &im);
	}

	void Classifier::getBlob(const char* name, BlobData* data){
		if (data){
			Blob* blob = getBlob(name);
			if (blob)
				data->copyFrom(blob);
		}
	}

	void Classifier::forward(int num, const Mat* ims){
		initContext();

		Mat fm;
		Blob* input = net_->input_blob(0);
		int w = input->width();
		int h = input->height();

		//fix，如果输入的通道不匹配会报错
		if (num != input->num()){
			input->Reshape(num, input->channel(), input->height(), input->width());
			net_->Reshape();
		}
		 
		for (int i = 0; i < num; ++i){
			float* image_ptr = input->mutable_cpu_data() + input->offset(i);
			CHECK_EQ(ims[i].channels(), input->channel()) << "channels not match";
			ims[i].copyTo(fm);

			if (CV_MAT_TYPE(fm.type()) != CV_32F)
				fm.convertTo(fm, CV_32F);

			if (fm.size() != cv::Size(w, h))
				cv::resize(fm, fm, cv::Size(w, h));

			if (this->num_mean_ > 0)
				fm -= cv::Scalar(this->mean_[0], this->mean_[1], this->mean_[2]);

			if (this->scale_ != 1)
				fm *= this->scale_;

			Mat ms[3];
			float* check = image_ptr;
			for (int c = 0; c < input->channel(); ++c){
				ms[c] = Mat(h, w, CV_32F, image_ptr);
				image_ptr += w * h;
			}

			split(fm, ms);
			CHECK_EQ((float*)ms[0].data, check) << "data ptr error";
		}

		if (this->gpuID_ != CLASSIFIER_INVALID_DEVICE_ID)
			setGPU(this->gpuID_);

		net_->Forward();
	}

	void Classifier::reshape2(int width, int height){
		Blob* input = net_->input_blob(0);
		input->Reshape(input->num(), input->channel(), height, width);
		net_->Reshape();
	}

	Blob* Classifier::getBlob(const char* name){
		return net_->blob(name);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	CCAPI void CCCALL releaseBlobData(BlobData* ptr){
		if (ptr) delete ptr;
	}

	CCAPI BlobData* CCCALL newBlobData(int num, int channels, int height, int width){
		BlobData* data = new BlobData();
		data->reshape(num, channels, height, width);
		return data;
	}

	CCAPI BlobData* CCCALL newBlobDataFromBlobShape(Blob* blob){
		return newBlobData(blob->num(), blob->channel(), blob->height(), blob->width());
	}

	CCAPI void CCCALL copyFromBlob(BlobData* dest, Blob* blob){
		dest->reshape(1, blob->channel(), blob->height(), blob->width());

		if (blob->count()>0)
			memcpy(dest->list, blob->cpu_data(), blob->count()*sizeof(float));
	}

	CCAPI void CCCALL copyOneFromBlob(BlobData* dest, Blob* blob, int numIndex){
		dest->reshape(1, blob->channel(), blob->height(), blob->width());

		int numSize = blob->channel()*blob->height()*blob->width();
		if (blob->count()>0)
			memcpy(dest->list, blob->cpu_data() + numIndex * numSize, numSize*sizeof(float));
	}

	CCAPI void CCCALL releaseObjectDetectList(ObjectDetectList* list){
		if (list) delete list;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	//任务池定义
	struct _TaskPool{
		Classifier* model;
		int count_worker;

		volatile int* operType;
		volatile Mat* recImgs;
		volatile char** blobNames;

		volatile int* cacheOperType;
		volatile Mat* cacheImgs;
		volatile char** cacheBlobNames;

		volatile int recNum;
		volatile BlobData** outBlobs;
		volatile BlobData** cacheOutBlobs;
		volatile ObjectDetectList** recDetection;
		volatile int job_cursor;
		semaphore* semaphoreWait;
		volatile semaphore** cacheSemaphoreGetResult;
		volatile semaphore** semaphoreGetResult;
		semaphore* semaphoreGetResultFinish;
		criticalsection jobCS;
		volatile bool flag_run;
		volatile bool flag_exit;
		int gpu_id;
	};

	static void swapCache(TaskPool* pool_){
		_TaskPool* pool = (_TaskPool*)pool_;

		pool->recNum = pool->job_cursor;
		if (pool->recNum > 0){
			enterCriticalSection(&pool->jobCS);
			pool->recNum = pool->job_cursor;
			std::swap(pool->cacheImgs, pool->recImgs);
			std::swap(pool->cacheBlobNames, pool->blobNames);
			std::swap(pool->cacheOperType, pool->operType);
			std::swap(pool->cacheSemaphoreGetResult, pool->semaphoreGetResult);
			std::swap(pool->cacheOutBlobs, pool->outBlobs);
			pool->job_cursor = 0;
			leaveCriticalSection(&pool->jobCS);
			releaseSemaphore(pool->semaphoreWait, pool->recNum);
		}

		if (pool->recNum == 0)
			sleep_cc(1);
	}

	//从blob数据中，得到检测结果列表，
	static vector<ObjectDetectList*> detectObjectMulti(BlobData* fr){

		vector<ObjectDetectList*> ods;
		if (!fr || !fr->list) return ods;

		float* p = fr->list;
		int num = *p++;
		ods.resize(num);
		for (int i = 0; i < num; ++i){
			int numobj = *p++;
			int totalbbox = 0;
			ods[i] = new ObjectDetectList();
			for (int j = 0; j < numobj; ++j){
				int numbox = *p++;
				p += numbox * 7;
				totalbbox += numbox;
			}

			ods[i]->count = totalbbox;
			if (ods[i]->count > 0)
				ods[i]->list = new ObjectInfo[ods[i]->count];
			else
				ods[i]->list = 0;
		}

		p = fr->list;
		num = *p++;
		for (int i = 0; i < num; ++i){
			int numobj = *p++;
			int box_index = 0;
			for (int j = 0; j < numobj; ++j){
				int numbox = *p++;
				for (int b = 0; b < numbox; ++b){
					auto& obj = ods[i]->list[box_index++];
					obj.image_id = p[0];
					obj.label = p[1];
					obj.score = p[2];
					obj.xmin = p[3];
					obj.ymin = p[4];
					obj.xmax = p[5];
					obj.ymax = p[6];
					p += 7;
				}
			}
			CV_Assert(box_index == ods[i]->count);
		}
		return ods;
	}

	static void poolThread(void* param){
		_TaskPool* pool = (_TaskPool*)param;

		// GPU是线程上下文相关的
		cc::setGPU(pool->gpu_id);
		pool->model->initContext();
		WPtr<BlobData> ssd_cacheBlob = new BlobData();

		//vector<Mat> ims;
		while (pool->flag_run){
			swapCache(pool);
			//printf("recnum = %d\n", pool->recNum);
			if (pool->recNum > 0){
				pool->model->reshape(pool->recNum);
				pool->model->forward(pool->recNum, (Mat*)pool->recImgs);

				if (pool->operType[0] == operType_Detection){
					for (int i = 0; i < pool->recNum; ++i){
						pool->recDetection[i] = 0;
					}

					Blob* outblob = pool->model->getBlob((const char*)pool->blobNames[0]);
					if (outblob){
						copyFromBlob(ssd_cacheBlob, outblob);

						vector<ObjectDetectList*> detectResult = detectObjectMulti(ssd_cacheBlob);
						if (detectResult.size() != pool->recNum){
							printf("detectResult.size()[%d] != recNum[%d]\n", detectResult.size(), pool->recNum);
						}
						else{
							for (int i = 0; i < pool->recNum; ++i){
								pool->recDetection[i] = detectResult[i];
								if (pool->recDetection[i]){
									int width = ((Mat*)pool->recImgs)->cols;
									int height = ((Mat*)pool->recImgs)->rows;
									for (int j = 0; j < pool->recDetection[i]->count; ++j){
										pool->recDetection[i]->list[j].xmin *= width;
										pool->recDetection[i]->list[j].ymin *= height;
										pool->recDetection[i]->list[j].xmax *= width;
										pool->recDetection[i]->list[j].ymax *= height;
									}
								}
							}
						}
					}
					else{
						printf("没有找到outblob: %s\n", (const char*)pool->blobNames[0]);
					}
				}
				else{
					for (int i = 0; i < pool->recNum; ++i){
						copyOneFromBlob((BlobData*)pool->outBlobs[i], pool->model->getBlob((const char*)pool->blobNames[i]), i);
					}
				}

				for (int i = 0; i < pool->recNum; ++i)
					releaseSemaphore((semaphore*)pool->semaphoreGetResult[i], 1);

				for (int i = 0; i < pool->recNum; ++i)
					waitSemaphore(pool->semaphoreGetResultFinish);
			}
		}
		pool->flag_run = false;
		pool->flag_exit = true;
	}

	CCAPI TaskPool* CCCALL buildPool(Classifier* model, int gpu_id, int batch_size){
		if (model->isInitContext()){
			printf("Warning:如果分类器已经被初始化，此时会造成多GPU时CUDNN错误，如果报错，请保证初始化在TaskPool内进行的\n");
		}

		batch_size = batch_size < 1 ? 1 : batch_size;

		_TaskPool* pool = new _TaskPool();
		memset(pool, 0, sizeof(*pool));
		pool->model = model;
		pool->count_worker = batch_size;

		pool->recImgs = new volatile Mat[batch_size];
		pool->blobNames = new volatile char*[batch_size];
		pool->operType = new volatile int[batch_size];

		pool->cacheImgs = new volatile Mat[batch_size];
		pool->cacheBlobNames = new volatile char*[batch_size];
		pool->cacheOperType = new volatile int[batch_size];

		pool->semaphoreWait = createSemaphore(batch_size, batch_size);
		pool->outBlobs = new volatile BlobData*[batch_size];
		pool->cacheOutBlobs = new volatile BlobData*[batch_size];
		pool->recDetection = new volatile ObjectDetectList*[batch_size];

		//pool->semaphoreGetResult = CreateSemaphoreA(0, 0, batch_size, 0);
		pool->semaphoreGetResultFinish = createSemaphore(0, batch_size);
		pool->gpu_id = gpu_id;
		pool->flag_exit = false;
		pool->flag_run = true;
		pool->semaphoreGetResult = new volatile semaphore*[batch_size];
		pool->cacheSemaphoreGetResult = new volatile semaphore*[batch_size];
		for (int i = 0; i < batch_size; ++i){
			pool->semaphoreGetResult[i] = createSemaphore(0, 1);
			pool->cacheSemaphoreGetResult[i] = createSemaphore(0, 1);
		}

		initializeCriticalSection(&pool->jobCS);
		thread(poolThread, pool).detach();
		return pool;
	}

	CCAPI ObjectDetectList* CCCALL forwardSSDByTaskPool(TaskPool* pool_, const Mat& img, const char* blob_name){
		_TaskPool* pool = (_TaskPool*)pool_;
		if (pool == 0) return 0;
		if (img.empty()) return 0;
		if (!pool->flag_run) return 0;

		Mat im;
		img.copyTo(im);
		if (im.empty()) return 0;

		waitSemaphore(pool->semaphoreWait);
		if (!pool->flag_run) return 0;

		enterCriticalSection(&pool->jobCS);
		int cursor = pool->job_cursor;
		volatile semaphore* semap = pool->cacheSemaphoreGetResult[cursor];
		((Mat*)pool->cacheImgs)[cursor] = im;
		pool->cacheBlobNames[cursor] = (char*)blob_name;
		pool->cacheOperType[cursor] = operType_Detection;
		pool->job_cursor++;
		leaveCriticalSection(&pool->jobCS);

		waitSemaphore((semaphore*)semap);
		volatile ObjectDetectList* result = pool->recDetection[cursor];
		releaseSemaphore(pool->semaphoreGetResultFinish, 1);
		return (ObjectDetectList*)result;
	}

	CCAPI bool CCCALL forwardByTaskPool(TaskPool* pool_, const Mat& img, const char* blob_name, BlobData* inplace_blobData){
	//CCAPI BlobData* CCCALL forwardByTaskPool(TaskPool* pool_, const Mat& img, const char* blob_name){
		_TaskPool* pool = (_TaskPool*)pool_;
		if (pool == 0) return false;
		if (img.empty()) return false;
		if (!pool->flag_run) return false;
		if (inplace_blobData == 0) return false;

		Mat im;
		img.copyTo(im);
		if (im.empty()) return false;

		waitSemaphore(pool->semaphoreWait);
		if (!pool->flag_run) return false;

		enterCriticalSection(&pool->jobCS);
		int cursor = pool->job_cursor;
		volatile semaphore* semap = pool->cacheSemaphoreGetResult[cursor];
		((Mat*)pool->cacheImgs)[cursor] = im;
		pool->cacheBlobNames[cursor] = (char*)blob_name;
		pool->cacheOperType[cursor] = operType_Forward;
		pool->cacheOutBlobs[cursor] = inplace_blobData;
		pool->job_cursor++;
		leaveCriticalSection(&pool->jobCS);

		waitSemaphore((semaphore*)semap);
		releaseSemaphore(pool->semaphoreGetResultFinish, 1);
		return true;
	}

	CCAPI void CCCALL releaseTaskPool(TaskPool* pool_){
		_TaskPool* pool = (_TaskPool*)pool_;
		if (pool == 0) return;

		pool->flag_run = false;
		releaseSemaphore(pool->semaphoreWait, pool->count_worker);
		releaseSemaphore(pool->semaphoreGetResultFinish, pool->count_worker);
		while (!pool->flag_exit)
			sleep_cc(10);

		deleteSemaphore(&pool->semaphoreWait);
		for (int i = 0; i < pool->count_worker; ++i){
			deleteSemaphore((semaphore**)&pool->semaphoreGetResult[i]);
			deleteSemaphore((semaphore**)&pool->cacheSemaphoreGetResult[i]);
		}

		deleteSemaphore(&pool->semaphoreGetResultFinish);
		deleteCriticalSection(&pool->jobCS);

		delete[] pool->recImgs;
		delete[] pool->operType;
		delete[] pool->blobNames;

		delete[] pool->cacheImgs;
		delete[] pool->cacheOperType;
		delete[] pool->cacheBlobNames;

		delete[] pool->recDetection;
		delete[] pool->semaphoreGetResult;
		delete[] pool->cacheSemaphoreGetResult;
		delete[] pool->outBlobs;
		delete[] pool->cacheOutBlobs;
		delete pool;
	}

	CCAPI void CCCALL initializeCriticalSection(criticalsection* cs){
		cs->p = new std::mutex();
	}

	CCAPI void CCCALL enterCriticalSection(criticalsection* cs){
		((std::mutex*)cs->p)->lock();
	}

	CCAPI void CCCALL leaveCriticalSection(criticalsection* cs){
		((std::mutex*)cs->p)->unlock();
	}

	CCAPI void CCCALL deleteCriticalSection(criticalsection* cs){
		delete ((std::mutex*)cs->p);
		cs->p = 0;
	}

	CCAPI semaphore* CCCALL createSemaphore(int numInitialize, int maxSemaphore){
		semaphore* s = new semaphore();
		s->p1 = new std::condition_variable();
		s->p2 = new std::mutex();
		s->numFree = numInitialize;
		s->maxSemaphore = maxSemaphore;
		return s;
	}

	CCAPI void CCCALL deleteSemaphore(semaphore** pps){
		if (pps){
			semaphore* ps = *pps;
			if (ps){
				if (ps->p1) delete ((std::condition_variable*)(ps->p1));
				if (ps->p2) delete ((std::mutex*)(ps->p2));
				ps->p1 = 0;
				ps->p2 = 0;
			}
			*pps = 0;
		}
	}

	CCAPI void CCCALL waitSemaphore(semaphore* s){
		std::unique_lock<std::mutex> lock(*(std::mutex*)(s->p2));
		((std::condition_variable*)(s->p1))->wait(lock, [=]{return s->numFree > 0; });
		--s->numFree;
	}

	CCAPI void CCCALL releaseSemaphore(semaphore* s, int num){
		std::unique_lock<std::mutex> lock(*(std::mutex*)(s->p2));
		num = min(s->maxSemaphore - s->numFree, num);

		for (int i = 0; i < num; ++i){
			s->numFree++;
			((std::condition_variable*)(s->p1))->notify_one();
		}
	}

	CCAPI void CCCALL sleep_cc(int milliseconds){
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}

	CCAPI void CCCALL disableLogPrintToConsole(){
		static volatile int flag = 0;
		if (flag) return;

		flag = 1;
		google::InitGoogleLogging("cc");
	}

	CCAPI const char* CCCALL getCCVersionString(){
		return VersionStr __TIMESTAMP__;
	}

	CCAPI int CCCALL getCCVersionInt(){
		return VersionInt;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	CCAPI int CCCALL argmax(Blob* classification_blob, int numIndex, float* conf_ptr){
		if (conf_ptr) *conf_ptr = 0;
		if (!classification_blob) return -1;

		int planeSize = classification_blob->height() * classification_blob->width();
		return argmax(classification_blob->mutable_cpu_data() + classification_blob->offset(numIndex), 
			classification_blob->channel() * planeSize, conf_ptr);
	}

	CCAPI int CCCALL argmax(float* data_ptr, int num_data, float* conf_ptr){
		if (conf_ptr) *conf_ptr = 0;
		if (!data_ptr || num_data < 1) return -1;
		int label = std::max_element(data_ptr, data_ptr + num_data) - data_ptr;
		if (conf_ptr) *conf_ptr = data_ptr[label];
		return label;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
}