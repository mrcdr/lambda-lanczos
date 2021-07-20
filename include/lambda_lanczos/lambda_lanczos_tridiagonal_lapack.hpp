#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_


/*
 * This file is used just for debug and benchmark.
 */

#include <vector>
#include <complex>

#if defined(LAMBDA_LANCZOS_USE_LAPACK)
#include <lapacke.h>
#elif defined(LAMBDA_LANCZOS_USE_MKL)
#include <mkl.h>
#endif

#include "lambda_lanczos_util.hpp"


namespace lambda_lanczos { namespace tridiagonal {
/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha,
                             const std::vector<T>& beta,
                             const size_t index) {
  const size_t n = alpha.size();
  auto a = alpha; // copy required because the contents will be destroyed
  auto b = beta;  // copy required because the contents will be destroyed
  auto z = std::vector<T>(1);


  LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', n, a.data(), b.data(), z.data(), 1);

  return a[index];
}


/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for given tri-diagonal matrix.
 */
template <typename T>
inline std::vector<T> tridiagonal_eigenvector(const std::vector<T>& alpha,
                                              const std::vector<T>& beta,
                                              const size_t index,
                                              const T ev) {
  (void)ev; // Unused declaration for compiler

  const size_t n = alpha.size();
  auto a = alpha; // copy required because the contents will be destroyed
  auto b = beta;  // copy required because the contents will be destroyed
  auto z = std::vector<T>(n*n);

  LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, a.data(), b.data(), z.data(), n);

  std::vector<T> cv(n);
  for(size_t i = 0; i < n; ++i) {
    cv[i] = z[index*n + i];
  }

  return cv;
}


/**
 * @brief Computes all eigenpairs (eigenvalues and eigenvectors) for given tri-diagonal matrix.
 */
template <typename T>
inline void tridiagonal_eigenpairs(const std::vector<T>& alpha,
                                   const std::vector<T>& beta,
                                   std::vector<T>& eigenvalues,
                                   std::vector<std::vector<T>>& eigenvectors) {
  const size_t n = alpha.size();
  auto a = alpha;
  auto b = beta;
  auto z = std::vector<T>(n*n);

  LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, a.data(), b.data(), z.data(), n);


  eigenvalues = std::move(a);
  eigenvectors.resize(n);
  for(size_t k = 0; k < n; ++k) {  // k-th eigenvector
    eigenvectors[k] = std::vector<T>(n);
    for(size_t i = 0; i < n; ++i) {
      eigenvectors[k][i] = z[k*n + i];
    }
  }
}


}} // namespace lambda_lanczos::tridiagonal

#endif  /* LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_ */
