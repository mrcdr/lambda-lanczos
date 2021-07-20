#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_H_

#if defined(LAMBDA_LANCZOS_USE_LAPACK) || defined(LAMBDA_LANCZOS_USE_MKL)
#include "lambda_lanczos_tridiagonal_lapack.hpp"
#else
#include "lambda_lanczos_tridiagonal_impl.hpp"
#endif

#endif /* LAMBDA_LANCZOS_TRIDIAGONAL_H_ */
