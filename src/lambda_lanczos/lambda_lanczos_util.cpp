#include <vector>
#include <numeric>
#include <cmath>
#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos_util {

template<typename T>
using vector = std::vector<T>;

using std::begin;
using std::end;

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
