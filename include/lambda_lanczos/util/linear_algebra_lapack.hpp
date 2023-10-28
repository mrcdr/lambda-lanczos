#ifndef LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_LAPACK_H_
#define LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_LAPACK_H_

// clang-format off
#include "macro.hpp" // This processes and defines certain macros
// clang-format on

#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
#include <execution>
#endif

#if defined(LAMBDA_LANCZOS_USE_LAPACK)
#include <cblas.h>
#include <lapacke.h>
#elif defined(LAMBDA_LANCZOS_USE_MKL)
#include <mkl.h>
#endif

#include "common.hpp"

namespace lambda_lanczos {
namespace util {

inline float inner_prod(const std::vector<float>& v1, const std::vector<float>& v2) {
  assert(v1.size() == v2.size());

  lapack_int n = v1.size();
  return cblas_sdot(n, v1.data(), 1, v2.data(), 1);
}

inline double inner_prod(const std::vector<double>& v1, const std::vector<double>& v2) {
  assert(v1.size() == v2.size());

  lapack_int n = v1.size();
  return cblas_ddot(n, v1.data(), 1, v2.data(), 1);
}

inline std::complex<float> inner_prod(const std::vector<std::complex<float>>& v1,
                                      const std::vector<std::complex<float>>& v2) {
  assert(v1.size() == v2.size());

  lapack_int n = v1.size();
  std::complex<float> result;
  cblas_cdotc_sub(n, v1.data(), 1, v2.data(), 1, &result);
  return result;
}

inline std::complex<double> inner_prod(const std::vector<std::complex<double>>& v1,
                                       const std::vector<std::complex<double>>& v2) {
  assert(v1.size() == v2.size());

  lapack_int n = v1.size();
  std::complex<double> result;
  cblas_zdotc_sub(n, v1.data(), 1, v2.data(), 1, &result);
  return result;
}

inline float norm(const std::vector<float>& vec) {
  return cblas_snrm2(vec.size(), vec.data(), 1);
}

inline double norm(const std::vector<double>& vec) {
  return cblas_dnrm2(vec.size(), vec.data(), 1);
}

inline float norm(const std::vector<std::complex<float>>& vec) {
  return cblas_scnrm2(vec.size(), vec.data(), 1);
}

inline double norm(const std::vector<std::complex<double>>& vec) {
  return cblas_dznrm2(vec.size(), vec.data(), 1);
}

inline void scalar_mul(float a, std::vector<float>& vec) {
  cblas_sscal(vec.size(), a, vec.data(), 1);
}

inline void scalar_mul(double a, std::vector<double>& vec) {
  cblas_dscal(vec.size(), a, vec.data(), 1);
}

inline void scalar_mul(std::complex<float> a, std::vector<std::complex<float>>& vec) {
  cblas_cscal(vec.size(), &a, vec.data(), 1);
}

inline void scalar_mul(std::complex<double> a, std::vector<std::complex<double>>& vec) {
  cblas_zscal(vec.size(), &a, vec.data(), 1);
}

/* Special routine to multiply a real scalar to a complex vector */
inline void scalar_mul(float a, std::vector<std::complex<float>>& vec) {
  cblas_csscal(vec.size(), a, vec.data(), 1);
}

/* Special routine to multiply a real scalar to a complex vector */
inline void scalar_mul(double a, std::vector<std::complex<double>>& vec) {
  cblas_zdscal(vec.size(), a, vec.data(), 1);
}

template <typename T>
inline void normalize(std::vector<T>& vec) {
  scalar_mul(T(1) / norm(vec), vec);
}

inline float m_norm(const std::vector<float>& vec) {
  return cblas_sasum(vec.size(), vec.data(), 1);
}

inline double m_norm(const std::vector<double>& vec) {
  return cblas_dasum(vec.size(), vec.data(), 1);
}

inline float m_norm(const std::vector<std::complex<float>>& vec) {
  return cblas_scasum(vec.size(), vec.data(), 1);
}

inline double m_norm(const std::vector<std::complex<double>>& vec) {
  return cblas_dzasum(vec.size(), vec.data(), 1);
}

template <typename ForwardIterator>
inline void schmidt_orth(std::vector<float>& uorth, ForwardIterator first, ForwardIterator last) {
  const auto n = uorth.size();

  for (auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    float alpha = -util::inner_prod(uk, uorth);
    cblas_saxpy(n, alpha, uk.data(), 1, uorth.data(), 1);
  }
}

template <typename ForwardIterator>
inline void schmidt_orth(std::vector<double>& uorth, ForwardIterator first, ForwardIterator last) {
  const auto n = uorth.size();

  for (auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    double alpha = -util::inner_prod(uk, uorth);
    cblas_daxpy(n, alpha, uk.data(), 1, uorth.data(), 1);
  }
}

template <typename ForwardIterator>
inline void schmidt_orth(std::vector<std::complex<float>>& uorth, ForwardIterator first, ForwardIterator last) {
  const auto n = uorth.size();

  for (auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    std::complex<float> alpha = -util::inner_prod(uk, uorth);
    cblas_caxpy(n, &alpha, uk.data(), 1, uorth.data(), 1);
  }
}

template <typename ForwardIterator>
inline void schmidt_orth(std::vector<std::complex<double>>& uorth, ForwardIterator first, ForwardIterator last) {
  const auto n = uorth.size();

  for (auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    std::complex<double> alpha = -util::inner_prod(uk, uorth);
    cblas_zaxpy(n, &alpha, uk.data(), 1, uorth.data(), 1);
  }
}

template <typename T>
void initAsIdentity(std::vector<std::vector<T>>& a, size_t n) {
  a.resize(n);
  for (size_t i = 0; i < n; ++i) {
    a[i].resize(n);

#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
    std::fill(LAMBDA_LANCZOS_STDPAR_POLICY, a[i].begin(), a[i].end(), T());
#else
    std::fill(a[i].begin(), a[i].end(), T());
#endif

    a[i][i] = 1.0;
  }
}

} /* namespace util */
} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_LAPACK_H_ */