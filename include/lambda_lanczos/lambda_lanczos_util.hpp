#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>
#include <complex>
#include <limits>
#include <cmath>

namespace lambda_lanczos_util {

/*
 * real_t<T> is a type mapper.  It is defined below.
 * By default, real_t<T> returns T.  However real_t<complex<T>> returns T.
 * Usage example: This function returns a real number even if T is complex:
 * template <typename T>
 * inline real_t<T> norm(const std::vector<T>& vec);
 */
template <typename T>
struct realTypeMap {
  typedef T type;
};
template <typename T>
struct realTypeMap<std::complex<T>> {
  typedef T type;
};
template <typename T>
using real_t = typename realTypeMap<T>::type;


template <typename T>
T inner_prod(const vector<T>&, const vector<T>&);


template <typename T>
double norm(const vector<double>&);


template <typename T1, typename T2>
void scalar_mul(T1, vector<T2>&);


template <typename T>
void normalize(vector<T>&);


template <typename T>
real_t<T> l1_norm(const vector<T>&);


/* Implementation */
  
/*
 * ConjugateProduct::prod(a,b) is a function which returns a*b by default.
 * However if the arguments are complex numbers, it returns conj(a)*b instead.
 *
 * (Since it is a static function, we don't need to include it in the interface.
 *  We can hide it here in the implementation section instead.)
 */
template <typename T>
struct ConjugateProduct {
public:
  static T prod(T a, T b) { return a*b; }
};

template <typename T>
struct ConjugateProduct<std::complex<T>> {
public:
  static std::complex<T> prod(std::complex<T> a, std::complex<T> b) {
    return std::conj(a)*b;
  }
};


template <typename T>
inline T inner_prod(const vector<T>& v1, const vector<T>& v2) {
  return std::inner_product(std::begin(v1), std::end(v1),
			    std::begin(v2), T(),
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


template <typename T>
inline real_t<T> l1_norm(const vector<T>& vec) {
  real_t<T> norm = real_t<T>(); // Zero initialization

  for(const T& element : vec) {
    norm += std::abs(element);
  }

  return norm;
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
