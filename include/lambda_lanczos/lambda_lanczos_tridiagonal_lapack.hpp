#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_


/*
 * This file is used just for debug and benchmark.
 */

#include <vector>
#include <complex>
#include <lapacke.h>
#include "lambda_lanczos_util.hpp"


namespace lambda_lanczos { namespace tridiagonal_lapack {
/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha,
                             const std::vector<T>& beta,
                             const size_t m,
                             const T eps) {
  const size_t n = alpha.size();
  auto a = alpha;
  auto b = beta;
  auto z = std::vector<T>(1);


  int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', n, a.data(), b.data(), z.data(), 1);

  return a[m];
}


/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for given tri-diagonal matrix.
 */
template <typename T>
inline std::vector<T> tridiagonal_eigenvector(const std::vector<T>& alpha,
                                              const std::vector<T>& beta,
                                              const size_t m) {
  const size_t n = alpha.size();
  auto a = alpha;
  auto b = beta;
  auto z = std::vector<T>(n*n);

  int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, a.data(), b.data(), z.data(), n);

  std::vector<T> cv(n);
  for(size_t i = 0; i < n; ++i) {
    cv[i] = z[m*n + i];
  }

  util::normalize(cv);

  return cv;
}

}} // namespace lambda_lanczos::tridiagonal_lapack

#endif  /* LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_ */
