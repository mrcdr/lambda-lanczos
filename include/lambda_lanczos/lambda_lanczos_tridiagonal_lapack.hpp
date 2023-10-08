#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_

/*
 * This file is used just for debug and benchmark.
 */

#include <complex>
#include <vector>

#if defined(LAMBDA_LANCZOS_USE_LAPACK)
#include <lapacke.h>
#elif defined(LAMBDA_LANCZOS_USE_MKL)
#include <mkl.h>
#endif

#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos {
namespace tridiagonal_lapack {

inline lapack_int stev(int matrix_layout, char jobz, lapack_int n, float* d, float* e, float* z, lapack_int ldz) {
  return LAPACKE_sstev(matrix_layout, jobz, n, d, e, z, ldz);
}

inline lapack_int stev(int matrix_layout, char jobz, lapack_int n, double* d, double* e, double* z, lapack_int ldz) {
  return LAPACKE_dstev(matrix_layout, jobz, n, d, e, z, ldz);
}

/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha, const std::vector<T>& beta, const size_t index) {
  const size_t n = alpha.size();
  auto a = alpha;  // copy required because the contents will be destroyed
  auto b = beta;   // copy required because the contents will be destroyed
  auto z = std::vector<T>(1);

  stev(LAPACK_COL_MAJOR, 'N', n, a.data(), b.data(), z.data(), 1);

  return a[index];
}

/**
 * @brief Computes all eigenpairs (eigenvalues and eigenvectors) for given tri-diagonal matrix.
 */
template <typename T>
inline void tridiagonal_eigenpairs(const std::vector<T>& alpha,
                                   const std::vector<T>& beta,
                                   std::vector<T>& eigenvalues,
                                   std::vector<std::vector<T>>& eigenvectors,
                                   bool compute_eigenvector = true) {
  const size_t n = alpha.size();
  auto a = alpha;
  auto b = beta;
  auto z = std::vector<T>(n * n);
  char jobz = compute_eigenvector ? 'V' : 'N';

  stev(LAPACK_COL_MAJOR, jobz, n, a.data(), b.data(), z.data(), n);

  eigenvalues = std::move(a);
  eigenvectors.resize(n);
  for (size_t k = 0; k < n; ++k) {  // k-th eigenvector
    eigenvectors[k] = std::vector<T>(n);
    for (size_t i = 0; i < n; ++i) {
      eigenvectors[k][i] = z[k * n + i];
    }
  }
}

/**
 * @brief Computes all eigenvalues for given tri-diagonal matrix
 * using the Implicitly Shifted QR algorithm.
 *
 * @param [in] alpha Diagonal elements of the full tridiagonal matrix.
 * @param [in] beta Sub-diagonal elements of the full tridiagonal matrix.
 * @param [out] eigenvalues Eigenvalues.
 *
 * @return Count of forced breaks due to unconvergence.
 */
template <typename T>
inline void tridiagonal_eigenvalues(const std::vector<T>& alpha,
                                    const std::vector<T>& beta,
                                    std::vector<T>& eigenvalues) {
  std::vector<std::vector<T>> dummy_eigenvectors;
  return tridiagonal_eigenpairs(alpha, beta, eigenvalues, dummy_eigenvectors, false);
}

} /* namespace tridiagonal_lapack */
} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_TRIDIAGONAL_LAPACK_H_ */
