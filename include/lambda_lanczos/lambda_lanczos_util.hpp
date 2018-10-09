#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>
#include <complex>
#include <limits>
#include <cmath>

namespace lambda_lanczos_util {
namespace {
template<typename T>
using vector = std::vector<T>;

using std::begin;
using std::end;
}


/*
 * Type to corresponding real type map,
 * used as realType<double>::type
 */
template <typename T>
struct realTypeMap;

template <>
struct realTypeMap<double> {
  typedef double type;
};

template <>
struct realTypeMap<float> {
  typedef float type;
};

template <>
struct realTypeMap<long double> {
  typedef long double type;
};

template <typename real_t>
struct realTypeMap<std::complex<real_t>> {
  typedef real_t type;
};

template <typename T>
using real_t = typename realTypeMap<T>::type;

template <typename T>
double norm(const vector<double>&);

template <typename T1, typename T2>
void scalar_mul(T1, vector<T2>&);

template <typename T>
void normalize(vector<T>&);


template <typename T>
constexpr int sig_decimal_digit();

template <typename T>
constexpr T minimum_effective_decimal();


/* Implementation */

template <typename T>
real_t<T> norm(const vector<T>& vec) {
  return std::sqrt(std::inner_product(begin(vec), end(vec), begin(vec), T()));
  // T() means zero value of type T
}

template <typename T1, typename T2>
void scalar_mul(T1 a, vector<T2>& vec) {
  int n = vec.size();
  for(int i = 0;i < n;i++) {
    vec[i] *= a;
  }
}

template <typename T>
void normalize(vector<T>& vec) {
  scalar_mul(1.0/norm(vec), vec);
}


/*
 * This returns the significant decimal digits of type T.
 * 
 */
template <typename T>
constexpr int sig_decimal_digit() {
  return (int)(std::numeric_limits<T>::digits *
	       log10(std::numeric_limits<T>::radix));
}

template <typename T>
constexpr T minimum_effective_decimal() {
  return pow(10, -sig_decimal_digit<T>());
}

} /* namespace lambda_lanczos_util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
