#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_H_

#if defined(LAMBDA_LANCZOS_USE_LAPACK) || defined(LAMBDA_LANCZOS_USE_MKL)
#include "lambda_lanczos_tridiagonal_lapack.hpp"
namespace lambda_lanczos {
namespace tridiagonal {
using namespace lambda_lanczos::tridiagonal_lapack;
} /* namespace tridiagonal */
} /* namespace lambda_lanczos */

#else
#include "lambda_lanczos_tridiagonal_impl.hpp"
namespace lambda_lanczos {
namespace tridiagonal {
using namespace lambda_lanczos::tridiagonal_impl;
} /* namespace tridiagonal */
} /* namespace lambda_lanczos */

#endif

#endif /* LAMBDA_LANCZOS_TRIDIAGONAL_H_ */
