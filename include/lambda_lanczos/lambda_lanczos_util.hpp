#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>

namespace lambda_lanczos_util {
namespace {
template<typename T>
using vector = std::vector<T>;

using std::begin;
using std::end;
}

double norm(const std::vector<double>&);
void scalar_mul(double, std::vector<double>&);
void normalize(std::vector<double>&);


/* Implementation */

double norm(const vector<double>& vec) {
  return std::sqrt(std::inner_product(begin(vec), end(vec), begin(vec), 0.0));
}

void scalar_mul(double a, vector<double>& vec) {
  int n = vec.size();
  for(int i = 0;i < n;i++) {
    vec[i] *= a;
  }
}

void normalize(vector<double>& vec) {
  scalar_mul(1.0/norm(vec), vec);
}

} /* namespace lambda_lanczos_util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
