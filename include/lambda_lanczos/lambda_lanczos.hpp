#ifndef LAMBDA_LANCZOS_H_
#define LAMBDA_LANCZOS_H_

#include <vector>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <lambda_lanczos_util.hpp>


namespace lambda_lanczos {


/*
 * "Partially specialization of function" is not allowed,
 * so here it is mimicked by wrapping the "init" function with a class template.
 */
template <typename T>
struct VectorRandomInitializer {
public:
  static void init(std::vector<T>& v) {
    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_real_distribution<T> rand((T)(-1.0), (T)(1.0));

    size_t n = v.size();
    for(size_t i = 0;i < n;i++) {
      v[i] = rand(mt);
    }
  }
};


template <typename T>
struct VectorRandomInitializer<std::complex<T>> {
public:
  static void init(std::vector<std::complex<T>>& v) {
    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_real_distribution<T> rand((T)(-1.0), (T)(1.0));

    size_t n = v.size();
    for(size_t i = 0;i < n;i++) {
      v[i] = std::complex<T>(rand(mt), rand(mt));
    }
  }
};


template <typename T>
class LambdaLanczos {
private:
  template <typename n_type>
  using real_t = util::real_t<n_type>;

public:
  std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul;
  std::function<void(std::vector<T>&)> init_vector = VectorRandomInitializer<T>::init;

  size_t matrix_size;
  int max_iteration;
  real_t<T> eps = util::minimum_effective_decimal<real_t<T>>() * 1e3;
  real_t<T> tridiag_eps_ratio = 1e-1;
  size_t initial_vector_size = 200;
  bool find_maximum;
  real_t<T> eigenvalue_offset = 0.0;

  LambdaLanczos(std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul, size_t matrix_size, bool find_maximum = false) :
    mv_mul(mv_mul), matrix_size(matrix_size), max_iteration(matrix_size), find_maximum(find_maximum) {}


  int run(real_t<T>& eigvalue, std::vector<T>& eigvec) const {
    assert(0 < this->tridiag_eps_ratio && this->tridiag_eps_ratio < 1);

    std::vector<std::vector<T>> u;     // Lanczos vectors
    std::vector<real_t<T>> alpha; // Diagonal elements of an approximated tridiagonal matrix
    std::vector<real_t<T>> beta;  // Subdiagonal elements of an approximated tridiagonal matrix

    const auto n = this->matrix_size;

    u.reserve(this->initial_vector_size);
    alpha.reserve(this->initial_vector_size);
    beta.reserve(this->initial_vector_size);

    u.emplace_back(n, 0.0); // Same as u.push_back(vector<T>(n, 0.0))

    std::vector<T> vk(n, 0.0);

    real_t<T> alphak = 0.0;
    alpha.push_back(alphak);
    real_t<T> betak = 0.0;
    beta.push_back(betak);

    std::vector<T> uk(n);
    this->init_vector(uk);
    util::normalize(uk);
    u.push_back(uk);

    real_t<T> ev, pev; // Calculated eigenvalue and previous one
    pev = std::numeric_limits<real_t<T>>::max();

    int itern = this->max_iteration;
    for(int k = 1;k <= this->max_iteration;k++) {
      /* vk = (A + offset*E)uk, here E is the identity matrix */
      std::fill(vk.begin(), vk.end(), 0.0);
      this->mv_mul(uk, vk);
      for(size_t i = 0;i < n;i++) {
	vk[i] += uk[i]*this->eigenvalue_offset;
      }

      alphak = std::real(util::inner_prod(u.back(), vk));

      /* The inner product <uk|vk> is real.
       * Proof:
       *     <uk|vk> = <uk|A|uk>
       *   On the other hand its complex conjugate is
       *     <uk|vk>^* = <vk|uk> = <uk|A^*|uk> = <uk|A|uk>
       *   here the condition that matrix A is a symmetric (Hermitian) is used.
       *   Therefore
       *     <uk|vk> = <vk|uk>^*
       *   <uk|vk> is real.
       */

      alpha.push_back(alphak);

      for(size_t i = 0;i < n; i++) {
	uk[i] = vk[i] - betak*u[k-1][i] - alphak*u[k][i];
      }

      schmidt_orth(uk, u);

      betak = util::norm(uk);
      beta.push_back(betak);

      ev = find_mth_eigenvalue(alpha, beta, this->find_maximum ? alpha.size()-2 : 0);
      // The first element of alpha is a dummy. Thus its size is alpha.size()-1

      const real_t<T> zero_threshold = util::minimum_effective_decimal<real_t<T>>()*1e-1;
      if(betak < zero_threshold) {
	u.push_back(uk);
	/* This element will never be accessed,
	 * but this "push" guarantees u to always have one more element than
	 * alpha and beta do.
	 */
	itern = k;
	break;
      }

      util::normalize(uk);
      u.push_back(uk);

      if(std::abs(ev-pev) < std::min(std::abs(ev), std::abs(pev))*this->eps) {
	itern = k;
	break;
      } else {
	pev = ev;
      }
    }

    eigvalue = ev - this->eigenvalue_offset;

    auto m = alpha.size();
    std::vector<T> cv(m+1);
    cv[0] = 0.0;
    cv[m] = 0.0;
    cv[m-1] = 1.0;

    beta[m-1] = 0.0;

    if(eigvec.size() < n) {
      eigvec.resize(n);
    }

    for(size_t i = 0;i < n;i++) {
      eigvec[i] = cv[m-1]*u[m-1][i];
    }

    for(size_t k = m-2;k >= 1;k--) {
      cv[k] = ((ev - alpha[k+1])*cv[k+1] - beta[k+1]*cv[k+2])/beta[k];

      for(size_t i = 0;i < n;i++) {
	eigvec[i] += cv[k]*u[k][i];
      }
    }

    util::normalize(eigvec);

    return itern;
  }

private:
  void schmidt_orth(std::vector<T>& uorth, const std::vector<std::vector<T>>& u) const {
    /* Vectors in u must be normalized, but uorth doesn't have to be. */

    auto n = uorth.size();

    for(size_t k = 0;k < u.size();k++) {
      T innprod = util::inner_prod(uorth, u[k]);

      for(size_t i = 0;i < n;i++) {
	uorth[i] -= innprod * u[k][i];
      }
    }
  }


  util::real_t<T> find_mth_eigenvalue(const std::vector<util::real_t<T>>& alpha,
				      const std::vector<util::real_t<T>>& beta,
				      const size_t m) const {
    real_t<T> eps = this->eps * this->tridiag_eps_ratio;
    real_t<T> pmid = std::numeric_limits<real_t<T>>::max();
    real_t<T> r = tridiagonal_eigen_limit(alpha, beta);
    real_t<T> lower = -r;
    real_t<T> upper = r;
    real_t<T> mid;
    unsigned int nmid; // Number of eigenvalues smaller than the "mid"

    while(upper-lower > std::min(std::abs(lower), std::abs(upper))*eps) {
      mid = (lower+upper)/2.0;
      nmid = num_of_eigs_smaller_than(mid, alpha, beta);
      if(nmid >= m+1) {
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


  /*
   * Compute the upper bound of the absolute value of eigenvalues
   * by Gerschgorin theorem. This routine gives a rough upper bound,
   * but it is sufficient because the bisection routine using
   * the upper bound converges exponentially.
   */
  util::real_t<T> tridiagonal_eigen_limit(const std::vector<real_t<T>>& alpha,
					  const std::vector<real_t<T>>& beta) const {
    real_t<T> r = util::l1_norm(alpha);
    r += 2*util::l1_norm(beta);

    return r;
  }


  /*
   * Algorithm from
   * Peter Arbenz et al. / "High Performance Algorithms for Structured Matrix Problems" /
   * Nova Science Publishers, Inc.
   */
  int num_of_eigs_smaller_than(real_t<T> c,
			       const std::vector<real_t<T>>& alpha,
			       const std::vector<real_t<T>>& beta) const {
    real_t<T> q_i = 1.0;
    int count = 0;
    size_t m = alpha.size();

    for(size_t i = 1;i < m;i++){
      q_i = alpha[i] - c - beta[i-1]*beta[i-1]/q_i;
      if(q_i < 0){
	count++;
      }
      if(q_i == 0){
	q_i = util::minimum_effective_decimal<real_t<T>>();
      }
    }

    return count;
  }
};


} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_H_ */
