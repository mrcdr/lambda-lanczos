#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>

namespace lambda_lanczos_util {

double norm(const std::vector<double>&);
void scalar_mul(double, std::vector<double>&);
void normalize(std::vector<double>&);

} /* namespace lambda_lanczos_util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
