/*
	CC深度学习库（Caffe）V4.0
	
	能够训练、推理
	对模型可以加解密打包为一个文件
	训练可视化，网络可视化，RPC
	提供基于OpenCV Mat的接口，提供基于std C、Mat的接口
	能够支持x64-GPU、x64-CPU和win32
	能跨平台，支持linux、windows
	提供java、易语言、c、python、c#接口
	对目标检测有特殊支持，例如RPN、SSD等
	任务池
	CPP Layer

	//Solver
	//Layer
	//Blob
	//Net
*/

#ifndef CC_UTILS_H
#define CC_UTILS_H
#include "cc.h"
  
using namespace std;

namespace cc{

#define VersionStr		"CC4.0.3"
#define VersionInt		0x0403

	using cv::Mat;
	//这是一个智能指针类，负责自动释放一些指针	
	//例如 WPtr<Net> net = loadNetFromPrototxt("deploy.prototxt");
	template<typename Dtype>
	class WPtr{
		typedef Dtype* DtypePtr;

		template<typename T>
		struct ptrInfo{
			T ptr;
			int refCount;

			ptrInfo(T p) :ptr(p), refCount(1){}
			void addRef(){ refCount++; }
			bool releaseRef(){ return --refCount <= 0; }
		};

	public:
		WPtr() :ptr(0){};
		WPtr(DtypePtr p){
			ptr = new ptrInfo<DtypePtr>(p);
		}
		WPtr(const WPtr& other){
			ptr = 0;
			operator=(other);
		}
		~WPtr(){
			releaseRef();
		}

		void release(DtypePtr ptr);

		DtypePtr operator->(){
			return get();
		}

		operator DtypePtr(){
			return get();
		}

		operator DtypePtr() const{
			return get();
		}

		WPtr& operator=(const WPtr& other){
			releaseRef();

			this->ptr = other.ptr;
			addRef();
			return *this;
		}

		DtypePtr get(){
			if (this->ptr)
				return ptr->ptr;
			return 0;
		}

		void addRef(){
			if (this->ptr)
				this->ptr->addRef();
		}

		void releaseRef(){
			if (this->ptr && this->ptr->releaseRef()){
				release(this->ptr->ptr);
				delete ptr;
				ptr = 0;
			}
		}

	private:
		ptrInfo<DtypePtr>* ptr;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	inline map<string, string> parseParamStr(const char* str){
		map<string, string> o;
		if (str){
			char* prev = 0;
			char* p = (char*)str;
			int stage = 0;
			string name, value;

			while (*p){
				while (*p){ if (*p != ' ') break; p++; }
				prev = p;

				while (*p){ if (*p == ' ' || *p == ':') break; p++; }
				if (*p) name = string(prev, p);

				while (*p){ if (*p != ' ' && *p != ':' || *p == '\'') break; p++; }
				bool has_yh = *p == '\'';
				if (has_yh) p++;
				prev = p;

				while (*p){ if (has_yh && *p == '\'' || !has_yh && (*p == ' ' || *p == ';')) break; p++; }
				if (p != prev){
					value = string(prev, p);
					o[name] = value;

					p++;
					while (*p){ if (*p != ' ' && *p != ';' && *p != '\'') break; p++; }
				}
			}
		}
		return o;
	}

	inline float getParamFloat(map<string, string>& p, const string& key, float default_ = 0){
		if (p.find(key) == p.end())
			return default_;
		return atof(p[key].c_str());
	}

	inline int getParamInt(map<string, string>& p, const string& key, int default_ = 0){
		if (p.find(key) == p.end())
			return default_;
		return atoi(p[key].c_str());
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct ObjectInfo{
		float score;
		float xmin, ymin, xmax, ymax;
		int label, image_id;

		cv::Rect box(){
			return cv::Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
		}

		cv::Point2f center(){
			return cv::Point2f(xmin+(xmax-xmin)*0.5, ymin+(ymax-ymin)*0.5);
		}
	};

	struct ObjectDetectList{
		int count;
		ObjectInfo* list;
	};

	typedef struct{ 
		void* p;
	} criticalsection;

	typedef struct{
		void* p1;
		void* p2;
		volatile int numFree;
		int maxSemaphore;
	} semaphore;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
#define CLASSIFIER_INVALID_DEVICE_ID			-2
	class CCAPI Classifier{
	public:
		Classifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID);
		Classifier(const char* datmodel, float scale, int numMeans, float* meanValue, int gpuID);
		Classifier(const void* prototxt, int lenprototxt, const void* caffemodel, int lencaffemodel, float scale, int numMeans, float* meanValue, int gpuID);
		void initContext();
		bool isInitContext();
		virtual ~Classifier();
		void forward(const Mat& im);
		void forward(int num, const Mat* ims);
		void reshape2(int width, int height);
		void reshape(int num = -1, int channels = -1, int height = -1, int width = -1);
		Blob* getBlob(const char* name);
		void getBlob(const char* name, BlobData* data);
		Blob* inputBlob(int index);
		Blob* outputBlob(int index);
		
	private:
		float mean_[3];
		int num_mean_;
		float scale_;
		WPtr<Net> net_;
		int gpuID_;
		CCString prototxt_;
		CCString caffemodel_;
		CCString datmodel_;
		int modelFrom_;
		bool contextInited_;
		char* ptrPrototxt_;
		int lenPrototxt_;
		char* ptrCaffemodel_;
		int lenCaffemodel_;
	};

	CCAPI Classifier* CCCALL loadClassifier3(const void* prototxt, int lenprototxt, const void* caffemodel, int lencaffemodel, float scale, int numMeans, float* meanValue, int gpuID);
	CCAPI Classifier* CCCALL loadClassifier2(const char* datmodel, float scale, int numMeans, float* meanValue, int gpuID);
	CCAPI Classifier* CCCALL loadClassifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID);
	CCAPI void CCCALL releaseClassifier(Classifier* clas);

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	CCAPI int CCCALL argmax(Blob* classification_blob, int numIndex = 0, float* conf_ptr = 0);
	CCAPI int CCCALL argmax(float* data_ptr, int num_data, float* conf_ptr);

	///////////////////////////////////////////////////////////////////////////////////////////////////////


#define operType_Forward			2
#define operType_Detection			3

	CCAPI BlobData* CCCALL newBlobData(int num, int channels, int height, int width);
	CCAPI BlobData* CCCALL newBlobDataFromBlobShape(Blob* blob);
	CCAPI void CCCALL copyFromBlob(BlobData* dest, Blob* blob);
	CCAPI void CCCALL copyOneFromBlob(BlobData* dest, Blob* blob, int numIndex);
	CCAPI void CCCALL releaseBlobData(BlobData* ptr);
	CCAPI void CCCALL releaseObjectDetectList(ObjectDetectList* list);

	CCAPI void CCCALL initializeCriticalSection(criticalsection* cs);
	CCAPI void CCCALL  enterCriticalSection(criticalsection* cs);
	CCAPI void CCCALL  leaveCriticalSection(criticalsection* cs);
	CCAPI void CCCALL  deleteCriticalSection(criticalsection* cs);

	CCAPI semaphore* CCCALL createSemaphore(int numInitialize, int maxSemaphore);
	CCAPI void CCCALL deleteSemaphore(semaphore** pps);

	CCAPI void CCCALL waitSemaphore(semaphore* s);
	CCAPI void CCCALL releaseSemaphore(semaphore* s, int num);
	CCAPI void CCCALL sleep_cc(int milliseconds);
	 
	typedef void TaskPool;
	CCAPI TaskPool* CCCALL buildPool(Classifier* model, int gpu_id, int batch_size);
	CCAPI ObjectDetectList* CCCALL forwardSSDByTaskPool(TaskPool* pool, const Mat& img, const char* blob_name);
	CCAPI bool CCCALL forwardByTaskPool(TaskPool* pool, const Mat& img, const char* blob_name, BlobData* inplace_blobData);
	CCAPI void CCCALL releaseTaskPool(TaskPool* pool);
	CCAPI void CCCALL disableLogPrintToConsole();
	CCAPI const char* CCCALL getCCVersionString();
	CCAPI int CCCALL getCCVersionInt();

	////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEFWPTR_RELEASE_FUNC(type)		\
	inline void WPtr<type>::release(type* p){\
	if (p) release##type(p);}

	template<typename DtypePtr>
	inline void WPtr<DtypePtr>::release(DtypePtr p){
		if (p) delete p;
	}

	DEFWPTR_RELEASE_FUNC(Solver);
	DEFWPTR_RELEASE_FUNC(Net);
	DEFWPTR_RELEASE_FUNC(Blob);
	DEFWPTR_RELEASE_FUNC(Classifier);
	DEFWPTR_RELEASE_FUNC(BlobData);
	DEFWPTR_RELEASE_FUNC(ObjectDetectList);
	DEFWPTR_RELEASE_FUNC(TaskPool);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	typedef enum CC_CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 } CC_CBLAS_ORDER;
	typedef enum CC_CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114 } CC_CBLAS_TRANSPOSE;
	typedef enum CC_CBLAS_UPLO      { CblasUpper = 121, CblasLower = 122 } CC_CBLAS_UPLO;
	typedef enum CC_CBLAS_DIAG      { CblasNonUnit = 131, CblasUnit = 132 } CC_CBLAS_DIAG;
	typedef enum CC_CBLAS_SIDE      { CblasLeft = 141, CblasRight = 142 } CC_CBLAS_SIDE;

	template <typename Dtype>
	CCAPI void CCCALL caffe_cpu_gemm(const CC_CBLAS_TRANSPOSE TransA,
		const CC_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		Dtype* C);

	template <typename Dtype>
	CCAPI void CCCALL caffe_cpu_gemv(const CC_CBLAS_TRANSPOSE TransA, const int M, const int N,
		const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
		Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
		Dtype* Y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
		const Dtype beta, Dtype* Y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_copy(const int N, const Dtype *X, Dtype *Y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_set(const int N, const Dtype alpha, Dtype *X);

	CCAPI void CCCALL caffe_memset(const size_t N, const int alpha, void* X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_scal(const int N, const Dtype alpha, Dtype *X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_sqr(const int N, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

	CCAPI unsigned int CCCALL caffe_rng_rand();

	template <typename Dtype>
	CCAPI Dtype CCCALL caffe_nextafter(const Dtype b);

	template <typename Dtype>
	CCAPI void CCCALL caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
		Dtype* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_rng_bernoulli(const int n, const Dtype p, int* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_exp(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_log(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_abs(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI Dtype CCCALL caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

	template <typename Dtype>
	CCAPI Dtype CCCALL caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
		const Dtype* y, const int incy);

	// Returns the sum of the absolute values of the elements of vector x
	template <typename Dtype>
	CCAPI Dtype CCCALL caffe_cpu_asum(const int n, const Dtype* x);

	template <typename Dtype>
	CCAPI void CCCALL caffe_bound(const int n, const Dtype* a, const Dtype min,
		const Dtype max, Dtype* y);

	// the branchless, type-safe version from
	// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
	template<typename Dtype>
	inline char caffe_sign(Dtype val) {
		return (Dtype(0) < val) - (val < Dtype(0));
	}

	// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
	//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
	// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
	// Git cherry picking that commit caused a conflict hard to resolve and
	//   copying that file in convenient for code reviewing.
	// So they have to be pasted here temporarily.
	template<typename Dtype>
	CCAPI void CCCALL caffe_cpu_sign(const int n, const Dtype* x, Dtype* y);

	template<typename Dtype>
	CCAPI void CCCALL caffe_cpu_sgnbit(const int n, const Dtype* x, Dtype* y);

	template<typename Dtype>
	CCAPI void CCCALL caffe_cpu_fabs(const int n, const Dtype* x, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);


#ifndef CPU_ONLY
	// Decaf gpu gemm provides an interface that is almost the same as the cpu
	// gemm function - following the c convention and calling the fortran-order
	// gpu code under the hood.
	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_gemm(const CC_CBLAS_TRANSPOSE TransA,
		const CC_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		Dtype* C);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_gemv(const CC_CBLAS_TRANSPOSE TransA, const int M, const int N,
		const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
		Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
		Dtype* Y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
		const Dtype beta, Dtype* Y);

	CCAPI void CCCALL caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

	CCAPI void CCCALL caffe_gpu_memset(const size_t N, const int alpha, void* X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_bound(const int n, const Dtype* a, const Dtype min,
		const Dtype max, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

	// caffe_gpu_rng_uniform with two arguments generates integers in the range
	// [0, UINT_MAX].
	CCAPI void CCCALL caffe_gpu_rng_uniform(const int n, unsigned int* r);

	// caffe_gpu_rng_uniform with four arguments generates floats in the range
	// (a, b] (strictly greater than a, less than or equal to b) due to the
	// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
	// curandGenerateUniform; with other limits will shift and scale the outputs
	// appropriately after calling curandGenerateUniform.
	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
		Dtype* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

	template<typename Dtype>
	CCAPI void CCCALL caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

	template<typename Dtype>
	CCAPI void CCCALL caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

	template <typename Dtype>
	CCAPI void CCCALL caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif
};

#endif //CC_UTILS_H