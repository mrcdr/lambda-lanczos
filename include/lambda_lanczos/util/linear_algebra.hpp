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
  return std::inner_product(
      std::begin(v1),
      std::end(v1),
      std::begin(v2),
      T(),
      [](const T& v, const T& u) -> T { return v + u; },
      [](const T& a, const T& b) -> T { return typed_conj(a) * b; });
  // T() means zero value of type T
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
  for (auto& elem : vec) {
    elem *= a;
  }
}

/**
 * @brief Normalizes given vector.
 */
template <typename T>
inline void normalize(std::vector<T>& vec) {
  scalar_mul(T(1) / norm(vec), vec);
}

/**
 * @brief Returns 1-norm of given vector.
 */
template <typename T>
inline real_t<T> l1_norm(const std::vector<T>& vec) {
  real_t<T> norm = real_t<T>();  // Zero initialization

  for (const T& element : vec) {
    norm += std::abs(element);
  }

  return norm;
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

} /* namespace util */
} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_UTIL_LINEAR_ALGEBRA_H_ */