#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>
#include <complex>
#include <limits>
#include <cmath>
#include <numeric>
#include <cassert>

namespace lambda_lanczos { namespace util {


/**
 * @brief Template class to map specific types. See #real_t<T> for usage.
 *
 */
template <typename T>
struct realTypeMap {
  typedef T type;
};

template <typename T>
struct realTypeMap<std::complex<T>> {
  typedef T type;
};

/**
 * @brief Type mapper from `T` to real type of `T`.
 *
 * By default, `real_t<T>` returns `T`.
 * However `real_t<complex<T>>` returns `T`.
 * Usage example: This function returns a real number even if `T` is complex:
 * @code
 * template <typename T>
 * inline real_t<T> norm(const std::vector<T>& vec);
 * @endcode
 */
template <typename T>
using real_t = typename realTypeMap<T>::type;


/**
 * @brief Complex conjugate template.
 *
 * This structure is required because partial specialization of
 * function template is not allowed in C++.
 *
 * Use #typed_conj in practice.
 */
template <typename T>
struct TypedConjugate {
  static T invoke(const T& val) {
    return val;
  }
};

template <typename T>
struct TypedConjugate<std::complex<T>> {
  static std::complex<T> invoke(const std::complex<T> val) {
    return std::conj(val);
  }
};


/**
 * @brief Complex conjugate with type.
 * This function returns the argument itself for real type,
 * and returns its complex conjugate for complex type.
 */
template <typename T>
inline T typed_conj(const T& val) {
  return TypedConjugate<T>::invoke(val);
}


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
  return std::inner_product(std::begin(v1), std::end(v1),
                            std::begin(v2), T(),
                            [](const T& v, const T& u) -> T { return v+u; },
                            [](const T& a, const T& b) -> T { return typed_conj(a)*b; });
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
  for(auto& elem : vec) {
    elem *= a;
  }
}

/**
 * @brief Normalizes given vector.
 */
template <typename T>
inline void normalize(std::vector<T>& vec) {
  scalar_mul(1.0/norm(vec), vec);
}


/**
 * @brief Returns 1-norm of given vector.
 */
template <typename T>
inline real_t<T> l1_norm(const std::vector<T>& vec) {
  real_t<T> norm = real_t<T>(); // Zero initialization

  for(const T& element : vec) {
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
inline void schmidt_orth(std::vector<T>& uorth,
                         ForwardIterator first,
                         ForwardIterator last) {
  const auto n = uorth.size();

  for(auto iter = first; iter != last; ++iter) {
    const auto& uk = *iter;
    T innprod = util::inner_prod(uk, uorth);

    for(size_t i = 0; i < n; ++i) {
      uorth[i] -= innprod * uk[i];
    }
  }
}


/**
 * @brief Returns the significant decimal digits of type T.
 *
 */
template <typename T>
inline constexpr int sig_decimal_digit() {
  return (int)(std::numeric_limits<T>::digits *
               log10(std::numeric_limits<T>::radix));
}


template <typename T>
inline constexpr T minimum_effective_decimal() {
  return pow(10, -sig_decimal_digit<T>());
}

}} /* namespace lambda_lanczos::util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
