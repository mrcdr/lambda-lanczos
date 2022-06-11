#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <iterator>
#include "lambda_lanczos_util.hpp"


namespace lambda_lanczos { namespace tridiagonal_impl {
/**
 * @brief Finds the number of eigenvalues of given tridiagonal matrix smaller than `c`.
 *
 * Algorithm from
 * Peter Arbenz et al. / "High Performance Algorithms for Structured Matrix Problems" /
 * Nova Science Publishers, Inc.
 */
template <typename T>
inline size_t num_of_eigs_smaller_than(T c,
                                       const std::vector<T>& alpha,
                                       const std::vector<T>& beta) {
  T q_i = alpha[0] - c;
  size_t count = 0;
  size_t m = alpha.size();

  if(q_i < 0){
    ++count;
  }

  for(size_t i = 1; i < m; ++i){
    q_i = alpha[i] - c - beta[i-1]*beta[i-1]/q_i;
    if(q_i < 0){
      ++count;
    }
    if(q_i == 0){
      q_i = std::numeric_limits<T>::epsilon();
    }
  }

  return count;
}


/**
 * @brief Computes the upper bound of the absolute value of eigenvalues by Gerschgorin theorem.
 *
 * This routine gives a rough upper bound,
 * but it is sufficient because the bisection routine using
 * the upper bound converges exponentially.
 */
template <typename T>
inline T tridiagonal_eigen_limit(const std::vector<T>& alpha,
                                 const std::vector<T>& beta) {
  T r = util::l1_norm(alpha);
  r += 2*util::l1_norm(beta);

  return r;
}


/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha,
                             const std::vector<T>& beta,
                             const size_t m) {
  T mid;
  T pmid = std::numeric_limits<T>::max();
  T r = tridiagonal_eigen_limit(alpha, beta);
  T lower = -r;
  T upper = r;

  while(upper-lower > std::min(std::abs(lower), std::abs(upper))*std::numeric_limits<T>::epsilon()) {
    mid = (lower+upper)/2.0;

    if(num_of_eigs_smaller_than(mid, alpha, beta) >= m+1) {
      upper = mid;
    } else {
      lower = mid;
    }

    if(mid == pmid) {
      /* This avoids an infinite loop due to zero matrix */
      break;
    }
    pmid = mid;
  }

  return lower; // The "lower" almost equals the "upper" here.
}


/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for given tri-diagonal matrix.
 */
template <typename T>
inline std::vector<T> tridiagonal_eigenvector(const std::vector<T>& alpha,
                                              const std::vector<T>& beta,
                                              const size_t index,
                                              const T ev) {
  (void)index; // Unused declaration for compiler

  const auto m = alpha.size();
  std::vector<T> cv(m);

  cv[m-1] = 1.0;

  if(m >= 2) {
    cv[m-2] = (ev - alpha[m-1]) * cv[m-1] / beta[m-2];
    for (size_t k = m-2; k-- > 0;) {
      cv[k] = ((ev - alpha[k + 1]) * cv[k + 1] - beta[k + 1] * cv[k + 2]) / beta[k];
    }
  }

  util::normalize(cv);

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
  eigenvalues.resize(n);
  eigenvectors.resize(n);

  for(size_t j = 0; j < n; ++j) {
    eigenvalues[j] = lambda_lanczos::tridiagonal_impl::find_mth_eigenvalue(alpha, beta, j);
    eigenvectors[j] = lambda_lanczos::tridiagonal_impl::tridiagonal_eigenvector(alpha, beta, j, eigenvalues[j]);
  }
}



/**
 * @brief Calculates the cosine and sine of Givens rotation that eliminates a specific element (see detail).
 *
 * This function calculates the cosine and sine that eliminate the b element, i.e. that satisfy
 * @verbatim
   (c -s)(a) = (x)
   (s  c)(b) = (0)
 * @endverbatim
 */
template <typename T>
inline std::pair<T, T> calc_givens_cs(T a, T b) {
  T c = 1;
  T s = 0;

  if(b != 0) {
    if(abs(b) > abs(a)) {
      auto r = -a/b;
      s = 1/sqrt(1+r*r);
      c = s*r;
    } else {
      auto r = -b/a;
      c = 1/sqrt(1+r*r);
      s = c*r;
    }
  }

  return std::make_pair(c, s);
}


/**
 * @brief Compute Givens rotation against kth and k+1th space.
 *
 * This function is highly specialized for the implicitly shifted QR algorithm.
 */
template <typename RandomIterator, typename T>
inline T givens_rotation_tridiagonal(RandomIterator alpha,
                                     RandomIterator beta,
                                     size_t n,
                                     size_t k,
                                     T x,
                                     T z) {
  using namespace std;

  // Prepare Givens' cosine and sine
  auto cs = calc_givens_cs(x, z);
  auto c = cs.first;
  auto s = cs.second;

  // Rotate
  auto a00 = c*alpha[k] - s*beta[k];
  auto a01 = c*beta[k] - s*alpha[k+1];
  auto a10 = s*alpha[k] + c*beta[k];
  auto a11 = s*beta[k] + c*alpha[k+1];
  auto zp = -s*beta[k+1];

  if(k < n-1) {
    beta[k+1] = c*beta[k+1];
  }


  if(0 < k) {
    beta[k-1] = beta[k-1]*c - z*s;
  }

  alpha[k] = a00*c - a01*s;
  beta[k] = a00*s + a01*c;
  alpha[k+1] = a10*s + a11*c;

  return zp;
}


template <typename RandomIterator, typename T>
inline void isqr_step(RandomIterator alpha,
                      RandomIterator beta,
                      std::vector<std::vector<T>>& q,
                      size_t n) {
  auto d = (alpha[n-2] - alpha[n-1])/2;
  auto mu = alpha[n-1] - beta[n-2]*beta[n-2]/(d+lambda_lanczos::util::sgn(d)*std::sqrt(d*d+beta[n-2]*beta[n-2]));
  auto x = alpha[0] - mu;
  auto z = beta[0];

  for(size_t k = 0; k < n-1; ++k) {
    if(q.size() == n) { // If necessary, calculate the orthogonal matrix (eigenvectors)
      auto cs = calc_givens_cs(x, z);
      auto c = cs.first;
      auto s = cs.second;

      // Keep in mind that q[k][j] is the jth element of the kth eigenvector.
      // This means an eigenvector is stored as a ROW of the matrix q
      // in the sense of mathematical notation.
      for(size_t j = 0; j < n; ++j) {
        auto v0 = q[k][j];
        auto v1 = q[k+1][j];

        q[k][j] = c*v0 - s*v1;
        q[k+1][j] = s*v0 + c*v1;
      }
    }

    z = givens_rotation_tridiagonal(alpha, beta, n, k, x, z);
    x = beta[k];
  }
}


/**
 * @brief Computes all eigenpairs (eigenvalues and eigenvectors) for given tri-diagonal matrix
 * using the Implicitly Shifted QR algorithm.
 */
template <typename T>
inline void tridiagonal_eigenpairs_isqr(const std::vector<T>& alpha_org,
                                        const std::vector<T>& beta_org,
                                        std::vector<T>& eigenvalues,
                                        std::vector<std::vector<T>>& eigenvectors) {
  const T eps = 1e-15;
  const size_t n = alpha_org.size();

  auto alpha = alpha_org;
  auto beta = beta_org;

  eigenvectors.resize(n);
  for(size_t i = 0; i < n; ++i) {
    eigenvectors[i].resize(n);
    std::fill(eigenvectors[i].begin(), eigenvectors[i].end(), T());
    eigenvectors[i][i] = 1.0;
  }

  while(true) {
    for(size_t i = 0; i < n-1; ++i) {
      if(std::abs(beta[i]) < (std::abs(alpha[i]) + std::abs(alpha[i+1]))*eps) {
        beta[i] = 0;
      }
    }

    size_t qidx = n-1;
    while(qidx > 0 && beta[qidx-1] == 0) {
      qidx--;
    }
    size_t pidx = qidx;
    while(pidx > 0 && beta[pidx-1] != 0) {
      pidx--;
    }

    if(qidx > 0) {
      isqr_step(alpha.begin(), beta.begin(), eigenvectors, n);
    } else {
      break;
    }
  }

  eigenvalues = alpha;
}

}} // namespace lambda_lanczos::tridiagonal_impl

#endif  /* LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_ */
