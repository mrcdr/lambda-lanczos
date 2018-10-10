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

template<typename T>
using complex = std::complex<T>;

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
struct realTypeMap<complex<real_t>> {
  typedef real_t type;
};

template <typename T>
using real_t = typename realTypeMap<T>::type;


template <typename T>
struct ConjugateProduct {
public:
  static T prod(T, T);
};

template <typename T>
struct ConjugateProduct<complex<T>> {
public:
  static complex<T> prod(complex<T>, complex<T>);
};



template <typename T>
T inner_prod(const vector<T>&, const vector<T>&);

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
inline T ConjugateProduct<T>::prod(T a, T b) {
  return a*b;
}

template <typename T>
inline complex<T> ConjugateProduct<complex<T>>::prod(complex<T> a, complex<T> b) {
  return conj(a)*b;
}


template <typename T>
inline T inner_prod(const vector<T>& v1, const vector<T>& v2) {
  return std::inner_product(begin(v1), end(v1),
			    begin(v2), T(),
			    [](T a, T b) -> T { return a+b; },
			    ConjugateProduct<T>::prod);
  // T() means zero value of type T
  // This spec is required because std::inner_product calculates
  // v1*v2 not conj(v1)*v2
}


template <typename T>
inline real_t<T> norm(const vector<T>& vec) {
  return std::sqrt(std::real(inner_prod(vec, vec)));
  // The norm of any complex vector <v|v> is real by definition.
}

template <typename T1, typename T2>
inline void scalar_mul(T1 a, vector<T2>& vec) {
  int n = vec.size();
  for(int i = 0;i < n;i++) {
    vec[i] *= a;
  }
}

template <typename T>
inline void normalize(vector<T>& vec) {
  scalar_mul(1.0/norm(vec), vec);
}


/*
 * This returns the significant decimal digits of type T.
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

} /* namespace lambda_lanczos_util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
