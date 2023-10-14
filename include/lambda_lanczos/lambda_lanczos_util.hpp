#include "util/common.hpp"

#if defined(LAMBDA_LANCZOS_USE_LAPACK) || defined(LAMBDA_LANCZOS_USE_MKL)
#include "util/linear_algebra_lapack.hpp"
#else
#include "util/linear_algebra.hpp"
#endif