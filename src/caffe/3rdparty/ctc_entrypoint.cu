#include <cstddef>
#include <iostream>
#include <algorithm>

#define __CUDACC__
#include "caffe/3rdparty/ctc.h"
#include "caffe/3rdparty/detail/cpu_ctc.cuh"
#include "caffe/3rdparty/detail/gpu_ctc.cuh"


extern "C" {

ctcStatus_t compute_ctc_loss_gpu(const float* const activations,
                                 float* gradients,
                                 const int* const flat_labels,
                                 const int* const label_lengths,
                                 const int* const input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 float *costs,
                                 void *workspace,
                                 ctcOptions options) {
        GpuCTC<float> ctc(alphabet_size, minibatch, workspace, options.stream,
                          options.blank_label);

        if (gradients != NULL)
            return ctc.cost_and_grad(activations, gradients, costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
        else
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
        return CTC_STATUS_EXECUTION_FAILED;
}
}
