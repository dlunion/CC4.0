/*
CC…Ó∂»—ßœ∞ø‚£®Caffe£©V4.0
*/

#ifndef CC_C_H
#define CC_C_H
#include "cc.h"
#include "cc_utils.h"

#ifdef EXPORT_CC_DLL
#define CC_CAPI extern "C" __declspec(dllexport)  
#else
#define CC_CAPI __declspec(dllimport)  
#endif

#define CC_C_CALL __stdcall

//Net
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC_CAPI cc::Blob* CC_C_CALL net_blobWithName(cc::Net* net, const char* name);
CC_CAPI cc::Blob* CC_C_CALL net_blobWithIndex(cc::Net* net, int index);
CC_CAPI void CC_C_CALL net_Forward(cc::Net* net, float* loss = 0);
CC_CAPI void CC_C_CALL net_Reshape(cc::Net* net);
CC_CAPI void CC_C_CALL net_copyTrainedParamFromFile(cc::Net* net, const char* file);
CC_CAPI void CC_C_CALL net_copyTrainedParamFromData(cc::Net* net, const void* data, int length);
CC_CAPI int CC_C_CALL net_num_input_blobs(cc::Net* net);
CC_CAPI int CC_C_CALL net_num_output_blobs(cc::Net* net);
CC_CAPI int CC_C_CALL net_num_blobs(cc::Net* net);
CC_CAPI cc::Blob* CC_C_CALL net_input_blob(cc::Net* net, int index);
CC_CAPI cc::Blob* CC_C_CALL net_output_blob(cc::Net* net, int index);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

CC_CAPI cc::Blob* CC_C_CALL newBlob();
CC_CAPI cc::Blob* CC_C_CALL newBlobByShape(int num = 1, int channels = 1, int height = 1, int width = 1);
CC_CAPI cc::Blob* CC_C_CALL newBlobByShapes(int numShape, int* shapes);
CC_CAPI void CC_C_CALL releaseBlob(cc::Blob* blob);
CC_CAPI void CC_C_CALL releaseSolver(cc::Solver* solver);
CC_CAPI void CC_C_CALL releaseNet(cc::Net* net);
CC_CAPI cc::Solver* CC_C_CALL loadSolverFromPrototxt(const char* solver_prototxt);
CC_CAPI cc::Solver* CC_C_CALL loadSolverFromPrototxtString(const char* solver_prototxt_string);
CC_CAPI cc::Net* CC_C_CALL loadNetFromPrototxt(const char* net_prototxt, int phase = cc::PhaseTest);
CC_CAPI cc::Net* CC_C_CALL loadNetFromPrototxtString(const char* net_prototxt, int length = -1, int phase = cc::PhaseTest);
CC_CAPI void CC_C_CALL setGPU(int id);


CC_CAPI int CC_C_CALL blob_shape(cc::Blob* blob, int index);
CC_CAPI int CC_C_CALL blob_num_axes(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_count(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_count2(cc::Blob* blob, int start_axis);
CC_CAPI int CC_C_CALL blob_height(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_width(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_channel(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_num(cc::Blob* blob);
CC_CAPI int CC_C_CALL blob_offset(cc::Blob* blob, int n);;
CC_CAPI void blob_set_cpu_data(cc::Blob* blob, float* data);
CC_CAPI const float* CC_C_CALL blob_cpu_data(cc::Blob* blob);
CC_CAPI const float* CC_C_CALL blob_gpu_data(cc::Blob* blob);
CC_CAPI float* CC_C_CALL blob_mutable_cpu_data(cc::Blob* blob);
CC_CAPI float* CC_C_CALL blob_mutable_gpu_data(cc::Blob* blob);
CC_CAPI const float* CC_C_CALL blob_cpu_diff(cc::Blob* blob);
CC_CAPI const float* CC_C_CALL blob_gpu_diff(cc::Blob* blob);
CC_CAPI float* CC_C_CALL blob_mutable_cpu_diff(cc::Blob* blob);
CC_CAPI float* CC_C_CALL blob_mutable_gpu_diff(cc::Blob* blob);
CC_CAPI void CC_C_CALL blob_Reshape(cc::Blob* blob, int num = 1, int channels = 1, int height = 1, int width = 1);
CC_CAPI void CC_C_CALL blob_Reshape2(cc::Blob* blob, int numShape, int* shapeDims);
CC_CAPI void CC_C_CALL blob_ReshapeLike(cc::Blob* blob, const cc::Blob* other);
CC_CAPI void CC_C_CALL blob_copyFrom(cc::Blob* blob, const cc::Blob* other, bool copyDiff = false, bool reshape = false);
CC_CAPI void CC_C_CALL blob_shapeString(cc::Blob* blob, char* buffer);
CC_CAPI bool CC_C_CALL blob_setDataRGB(cc::Blob* blob, int numIndex, const void* imageData, int length, float alpha, float beta);


CC_CAPI cc::Classifier* CC_C_CALL loadClassifier(const char* prototxt, const char* caffemodel, float scale, int numMeans, float* meanValue, int gpuID);
CC_CAPI void CC_C_CALL releaseClassifier(cc::Classifier* clas);
CC_CAPI bool CC_C_CALL Classifier_forward(cc::Classifier* classifier, const void* imageData, int length);
CC_CAPI void CC_C_CALL Classifier_reshape2(cc::Classifier* classifier, int width, int height);
CC_CAPI void CC_C_CALL Classifier_reshape(cc::Classifier* classifier, int num = -1, int channels = -1, int height = -1, int width = -1);
CC_CAPI cc::Blob* CC_C_CALL Classifier_getBlob(cc::Classifier* classifier, const char* name);

#endif //CC_C_H