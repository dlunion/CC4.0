#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/cc/core/cc_utils.h"

using namespace caffe;
namespace cc {

#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
	template<typename Dtype> \
	CCAPI void CCCALL caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) {\
	for (int i = 0; i < n; ++i) {\
		operation; \
	} \
	}

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
	y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template CCAPI void CCCALL caffe_cpu_sign(const int n, const float* x, float* y);
template CCAPI void CCCALL caffe_cpu_sgnbit(const int n, const float* x, float* y);
template CCAPI void CCCALL caffe_cpu_fabs(const int n, const float* x, float* y);

template CCAPI void CCCALL caffe_cpu_sign(const int n, const double* x, double* y);
template CCAPI void CCCALL caffe_cpu_sgnbit(const int n, const double* x, double* y);
template CCAPI void CCCALL caffe_cpu_fabs(const int n, const double* x, double* y);

template<>
CCAPI void CCCALL caffe_cpu_gemm<float>(const CC_CBLAS_TRANSPOSE TransA,
	const CC_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == ::CblasNoTrans) ? K : M;
  int ldb = (TransB == ::CblasNoTrans) ? N : K;
  cblas_sgemm(::CblasRowMajor, (CBLAS_TRANSPOSE)TransA, (CBLAS_TRANSPOSE)TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

CCAPI void CCCALL caffe_memset(const size_t N, const int alpha, void* X) {
	memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template<>
CCAPI void CCCALL caffe_cpu_gemm<double>(const CC_CBLAS_TRANSPOSE TransA,
	const CC_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == ::CblasNoTrans) ? K : M;
  int ldb = (TransB == ::CblasNoTrans) ? N : K;
  cblas_dgemm(::CblasRowMajor, (CBLAS_TRANSPOSE)TransA, (CBLAS_TRANSPOSE)TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
CCAPI void CCCALL caffe_cpu_gemv<float>(const CC_CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
	cblas_sgemv(::CblasRowMajor, (CBLAS_TRANSPOSE)TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
CCAPI void CCCALL caffe_cpu_gemv<double>(const CC_CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
	cblas_dgemv(::CblasRowMajor, (CBLAS_TRANSPOSE)TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
CCAPI void CCCALL caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
CCAPI void CCCALL caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
CCAPI void CCCALL caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template CCAPI void CCCALL caffe_set<int>(const int N, const int alpha, int* Y);
template CCAPI void CCCALL caffe_set<float>(const int N, const float alpha, float* Y);
template CCAPI void CCCALL caffe_set<double>(const int N, const double alpha, double* Y);

template <>
CCAPI void CCCALL caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
CCAPI void CCCALL caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
CCAPI void CCCALL caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template CCAPI void CCCALL caffe_copy<bool>(const int N, const bool* X, bool* Y);
template CCAPI void CCCALL caffe_copy<int>(const int N, const int* X, int* Y);
template CCAPI void CCCALL caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template CCAPI void CCCALL caffe_copy<float>(const int N, const float* X, float* Y);
template CCAPI void CCCALL caffe_copy<double>(const int N, const double* X, double* Y);

template <>
CCAPI void CCCALL caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
CCAPI void CCCALL caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
CCAPI void CCCALL caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
CCAPI void CCCALL caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
CCAPI void CCCALL caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
CCAPI void CCCALL caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
CCAPI void CCCALL caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
CCAPI void CCCALL caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
CCAPI void CCCALL caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
CCAPI void CCCALL caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
CCAPI void CCCALL caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
CCAPI void CCCALL caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
CCAPI void CCCALL caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

CCAPI unsigned int CCCALL caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
CCAPI Dtype CCCALL caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
CCAPI float CCCALL caffe_nextafter(const float b);

template
CCAPI double CCCALL caffe_nextafter(const double b);

template <typename Dtype>
CCAPI void CCCALL caffe_rng_uniform(const int n, const Dtype a_, const Dtype b_, Dtype* r) {

	Dtype a = a_;
	Dtype b = b_;
	if (a > b) std::swap(a, b);
  
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
CCAPI void CCCALL caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
CCAPI void CCCALL caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
CCAPI void CCCALL caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
CCAPI void CCCALL caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
CCAPI void CCCALL caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
CCAPI void CCCALL caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
CCAPI void CCCALL caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
CCAPI void CCCALL caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
CCAPI void CCCALL caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
CCAPI void CCCALL caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
CCAPI void CCCALL caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
CCAPI float CCCALL caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
CCAPI double CCCALL caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
CCAPI Dtype CCCALL caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
CCAPI float CCCALL caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
CCAPI double CCCALL caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
CCAPI float CCCALL caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
CCAPI double CCCALL caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
CCAPI void CCCALL caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
CCAPI void CCCALL caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <>
CCAPI void CCCALL caffe_bound(const int N, const float* a, const float min,
	const float max, float* y) {
	for (int i = 0; i < N; ++i) {
		y[i] = std::min(std::max(a[i], min), max);
	}
}

template <>
CCAPI void CCCALL caffe_bound(const int N, const double* a, const double min,
	const double max, double* y) {
	for (int i = 0; i < N; ++i) {
		y[i] = std::min(std::max(a[i], min), max);
	}
}

}  // namespace caffe
