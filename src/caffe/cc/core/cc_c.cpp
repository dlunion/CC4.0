
#include "caffe/net.hpp"
#include "caffe/cc/core/cc.h"
#include "caffe/cc/core/cc_c.h"
#include <cv.h>
#include <highgui.h>

using namespace cv;

//Net
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC_CAPI cc::Blob* CC_C_CALL net_blobWithName(cc::Net* net, const char* name){ return net->blob(name); }
CC_CAPI cc::Blob* CC_C_CALL net_blobWithIndex(cc::Net* net, int index){ return net->blob(index); }
CC_CAPI void CC_C_CALL net_Forward(cc::Net* net, float* loss){ net->Forward(loss); }
CC_CAPI void CC_C_CALL net_Reshape(cc::Net* net){ net->Reshape(); }
CC_CAPI void CC_C_CALL net_copyTrainedParamFromFile(cc::Net* net, const char* file){ net->copyTrainedParamFromFile(file); }
CC_CAPI void CC_C_CALL net_copyTrainedParamFromData(cc::Net* net, const void* data, int length){ net->copyTrainedParamFromData(data, length); }
CC_CAPI int CC_C_CALL net_num_input_blobs(cc::Net* net){ return net->num_input_blobs(); }
CC_CAPI int CC_C_CALL net_num_output_blobs(cc::Net* net){ return net->num_output_blobs(); }
CC_CAPI int CC_C_CALL net_num_blobs(cc::Net* net){ return net->num_blobs(); }
CC_CAPI cc::Blob* CC_C_CALL net_input_blob(cc::Net* net, int index){ return net->input_blob(index); }
CC_CAPI cc::Blob* CC_C_CALL net_output_blob(cc::Net* net, int index){ return net->output_blob(index); }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


//blob
/////////////////////////////////////////////////////////////////////////////////////////////////////////
CC_CAPI int CC_C_CALL blob_shape(cc::Blob* blob, int index){ return blob->shape(index); }
CC_CAPI int CC_C_CALL blob_num_axes(cc::Blob* blob){ return blob->num_axes(); }
CC_CAPI int CC_C_CALL blob_count(cc::Blob* blob){ return blob->count(); }
CC_CAPI int CC_C_CALL blob_count2(cc::Blob* blob, int start_axis){ return blob->count(start_axis); }
CC_CAPI int CC_C_CALL blob_height(cc::Blob* blob){ return blob->height(); }
CC_CAPI int CC_C_CALL blob_width(cc::Blob* blob){ return blob->width(); }
CC_CAPI int CC_C_CALL blob_channel(cc::Blob* blob){ return blob->channel(); }
CC_CAPI int CC_C_CALL blob_num(cc::Blob* blob){ return blob->num(); }
CC_CAPI int CC_C_CALL blob_offset(cc::Blob* blob, int n){ return blob->offset(n); }
CC_CAPI void blob_set_cpu_data(cc::Blob* blob, float* data){ blob->set_cpu_data(data); }
CC_CAPI const float* CC_C_CALL blob_cpu_data(cc::Blob* blob){ return blob->cpu_data(); }
CC_CAPI const float* CC_C_CALL blob_gpu_data(cc::Blob* blob){ return blob->gpu_data(); }
CC_CAPI float* CC_C_CALL blob_mutable_cpu_data(cc::Blob* blob){ return blob->mutable_cpu_data(); }
CC_CAPI float* CC_C_CALL blob_mutable_gpu_data(cc::Blob* blob){ return blob->mutable_gpu_data(); }
CC_CAPI const float* CC_C_CALL blob_cpu_diff(cc::Blob* blob){ return blob->cpu_diff(); }
CC_CAPI const float* CC_C_CALL blob_gpu_diff(cc::Blob* blob){ return blob->gpu_diff(); }
CC_CAPI float* CC_C_CALL blob_mutable_cpu_diff(cc::Blob* blob){ return blob->mutable_cpu_data(); }
CC_CAPI float* CC_C_CALL blob_mutable_gpu_diff(cc::Blob* blob){ return blob->mutable_gpu_data(); }
CC_CAPI void CC_C_CALL blob_Reshape(cc::Blob* blob, int num, int channels, int height, int width){ blob->Reshape(num, channels, height, width); }
CC_CAPI void CC_C_CALL blob_Reshape2(cc::Blob* blob, int numShape, int* shapeDims){ blob->Reshape(numShape, shapeDims); }
CC_CAPI void CC_C_CALL blob_ReshapeLike(cc::Blob* blob, const cc::Blob* other){ blob->ReshapeLike(*other); }
CC_CAPI void CC_C_CALL blob_copyFrom(cc::Blob* blob, const cc::Blob* other, bool copyDiff, bool reshape){ blob->copyFrom(*other, copyDiff, reshape); }
CC_CAPI void CC_C_CALL blob_shapeString(cc::Blob* blob, char* buffer){ strcpy(buffer, blob->shapeString().c_str()); }
CC_CAPI bool CC_C_CALL blob_setDataRGB(cc::Blob* blob, int numIndex, const void* imageData, int length, float alpha, float beta){
	Mat im;
	try{
		im = imdecode(Mat(1, length, CV_8U, (uchar*)imageData), 1);
	}
	catch (...){}

	if (im.empty()) return false;
	im.convertTo(im, CV_32F, alpha, beta);
	blob->setDataRGB(numIndex, im);
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


CC_CAPI cc::Blob* CC_C_CALL newBlob(){ return cc::newBlob(); }
CC_CAPI cc::Blob* CC_C_CALL newBlobByShape(int num, int channels, int height, int width){ return cc::newBlobByShape(num, channels, height, width); };
CC_CAPI cc::Blob* CC_C_CALL newBlobByShapes(int numShape, int* shapes){ return cc::newBlobByShapes(numShape, shapes); }
CC_CAPI void CC_C_CALL releaseBlob(cc::Blob* blob){ cc::releaseBlob(blob); }
CC_CAPI void CC_C_CALL releaseSolver(cc::Solver* solver){cc::releaseSolver(solver);};
CC_CAPI void CC_C_CALL releaseNet(cc::Net* net){ cc::releaseNet(net); }
CC_CAPI cc::Solver* CC_C_CALL loadSolverFromPrototxt(const char* solver_prototxt){ return cc::loadSolverFromPrototxt(solver_prototxt); }
CC_CAPI cc::Solver* CC_C_CALL loadSolverFromPrototxtString(const char* solver_prototxt_string){ return cc::loadSolverFromPrototxtString(solver_prototxt_string); }
CC_CAPI cc::Net* CC_C_CALL loadNetFromPrototxt(const char* net_prototxt, int phase){ return cc::loadNetFromPrototxt(net_prototxt, phase); }
CC_CAPI cc::Net* CC_C_CALL loadNetFromPrototxtString(const char* net_prototxt, int length, int phase){ return cc::loadNetFromPrototxtString(net_prototxt, length, phase); }
CC_CAPI void CC_C_CALL setGPU(int id){ cc::setGPU(id); }


CC_CAPI cc::Classifier* CC_C_CALL loadClassifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID){
	return cc::loadClassifier(prototxt, caffemodel, scale, numMeans, meanValue, gpuID);
}

CC_CAPI void CC_C_CALL releaseClassifier(cc::Classifier* clas){
	cc::releaseClassifier(clas);
}

CC_CAPI bool CC_C_CALL Classifier_forward(cc::Classifier* classifier, const void* imageData, int length){
	Mat im;
	try{
		im = imdecode(Mat(1, length, CV_8U, (uchar*)imageData), 1);
	}
	catch (...){}

	if (im.empty()) return false;

	classifier->forward(im);
	return true;
}

CC_CAPI void CC_C_CALL Classifier_reshape2(cc::Classifier* classifier, int width, int height){ classifier->reshape2(width, height); }
CC_CAPI void CC_C_CALL Classifier_reshape(cc::Classifier* classifier, int num, int channels, int height, int width){
	classifier->reshape(num, channels, height, width);
}
CC_CAPI cc::Blob* CC_C_CALL Classifier_getBlob(cc::Classifier* classifier, const char* name){
	return classifier->getBlob(name);
}