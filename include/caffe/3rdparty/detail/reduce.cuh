#pragma once

ctcStatus_t reduce_negate(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_exp(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_max(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);

ctcStatus_t reduce_negate(const double* input, double* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_exp(const double* input, double* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_max(const double* input, double* output, int rows, int cols, bool axis, cudaStream_t stream);