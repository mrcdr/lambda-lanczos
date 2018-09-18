#ifndef LAMBDA_LANCZOS_H_
#define LAMBDA_LANCZOS_H_

#include <vector>
#include <functional>

namespace lambda_lanczos {

double lanczos(std::function<void(double*, double*)>,
		   int , double, double*, int&);
void schmidt_orth(std::vector<double>&, const std::vector<std::vector<double>>&);
double compute_eigenvalue(const std::vector<double>&,
			  const std::vector<double>&,
			  double);
double tridiagonal_eigen_limit(const std::vector<double>&,
			       const std::vector<double>&);
int num_of_eigs_smaller_than(double,
			     const std::vector<double>&,
			     const std::vector<double>&);
void mul_compressed_mat(double*, int*, int*, int, int, double*, double*);
void init_random(std::vector<double>&);

} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_H_ */
