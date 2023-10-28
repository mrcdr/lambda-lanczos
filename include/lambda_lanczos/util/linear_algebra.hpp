#ifndef LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_H_
#define LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_H_

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

#include "common.hpp"

namespace lambda_lanczos {
namespace util {

/**
 * @brief Returns "mathematical" inner product of v1 and v2.
 *
 * This function is needed because
 * std::inner_product calculates transpose(v1)*v2 instead of dagger(v1)*v2 for complex type.
 *
 */
template <typename T>
inline T inner_prod(const std::vector<T>& v1, const std::vector<T>& v2) {
  assert(v1.size() == v2.size());

#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
  return std::transform_reduce(
      LAMBDA_LANCZOS_STDPAR_POLICY,
      std::begin(v1),
      std::end(v1),
      std::begin(v2),
      T(),
      [](const T& u, const T& v) -> T { return u + v; },
      [](const T& a, const T& b) -> T { return typed_conj(a) * b; });
#else
  return std::inner_product(
      std::begin(v1),
      std::end(v1),
      std::begin(v2),
      T(),
      [](const T& u, const T& v) -> T { return u + v; },
      [](const T& a, const T& b) -> T { return typed_conj(a) * b; });
#endif
}

/**
 * @brief Returns Euclidean norm of given vector.
 */
template <typename T>
inline real_t<T> norm(const std::vector<T>& vec) {
  return std::sqrt(std::real(inner_prod(vec, vec)));
  // The norm of any complex vector <v|v> is real by definition.
}

/**
 * @brief Multiplies each element of vec by a.
 */
template <typename T1, typename T2>
inline void scalar_mul(T1 a, std::vector<T2>& vec) {
#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
  std::for_each(LAMBDA_LANCZOS_STDPAR_POLICY, vec.begin(), vec.end(), [&a](T2& elem) { elem *= a; });
#else
  std::for_each(vec.begin(), vec.end(), [&a](T2& elem) { elem *= a; });
#endif
}

/**
 * @brief Normalizes given vector.
 */
template <typename T>
inline void normalize(std::vector<T>& vec) {
  scalar_mul(T(1) / norm(vec), vec);
}

template <typename T>
struct ManhattanNorm {
  static T invoke(const std::vector<T>& vec) {
#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
    return std::transform_reduce(
        LAMBDA_LANCZOS_STDPAR_POLICY, std::begin(vec), std::end(vec), T(), std::plus<T>(), std::abs);
#else
    return std::accumulate(
        std::begin(vec), std::end(vec), T(), [](const T& acc, const T& elem) -> T { return acc + std::abs(elem); });
#endif
  }
};

template <typename T>
struct ManhattanNorm<std::complex<T>> {
  static T invoke(const std::vector<std::complex<T>>& vec) {
#ifdef LAMBDA_LANCZOS_STDPAR_POLICY
    return std::transform_reduce(
        LAMBDA_LANCZOS_STDPAR_POLICY,
        std::begin(vec),
        std::end(vec),
        T(),
        std::plus<T>(),
        [](const std::complex<T>& elem) -> T { return std::abs(std::real(elem)) + std::abs(std::imag(elem)); });
#else
    return std::accumulate(std::begin(vec), std::end(vec), T(), [](const T& acc, const std::complex<T>& elem) -> T {
      return acc + std::abs(std::real(elem)) + std::abs(std::imag(elem));
    });
#endif
  }
};

/**
 * @brief Returns Manhattan-like norm of given vector.
 *
 * @note For a real vector {r_i}, returned value is sum_i |r_i|, i.e. the L1 norm in mathematical definition.
 * For a complex vector {c_i}, returned value is sum_i |Re(c_i)| + |Im(c_i)|,
 * instead of sum_i sqrt(Re(c_i)^2 + Im(c_i)^2) in mathematical definition.
 * This definition can avoid sqrt calculations and is also implemented as _ASUM routines in BLAS.
 */
template <typename T>
inline real_t<T> m_norm(const std::vector<T>& vec) {
  return ManhattanNorm<T>::invoke(vec);
}

/**
 * @brief Orthogonalizes vector `uorth` with respect to orthonormal vectors defined by given iterators.
 *
 * Vectors in `u` must be normalized, but uorth doesn't have to be.
 */
template <typename ForwardIterator, typename T>
inline void schmidt_orth(std::vector<T>& uorth, ForwardIterator first, ForwardIterator last) {
  const auto n = uorth.size();

  for (auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    T innprod = util::inner_prod(uk, uorth);

    for (size_t i = 0; i < n; ++i) {
      uorth[i] -= innprod * uk[i];
    }
  }
}

/**
 * @brief Initializes the given matrix `a` to an n by n identity matrix.
 */
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

#endif /* LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_H_ */