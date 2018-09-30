# Lambda Lanczos

Adaptive Lanczos algorithm library

## Overview

**Lambda Lanczos** calculates the smallest or largest eigenvalue and
corresponding eigenvector of a matrix.

The characteristic feature is the matrix-vector multiplication routine used in
the Lanczos algorithm is adaptive:

```c++
#include <lambda_lanczos/lambda_lanczos.hpp>
using lambda_lanczos::LambdaLanczos;

/* Some include and using declarations are omitted */


void sample1() {
  const int n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
			  {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos engine(mv_mul, n, true); // true means to calculate the largest eigenvalue.
  double eigen_value;
  vector<double> eigen_vector(n);
  int itern  = engine.run(eigen_value, eigen_vecor);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigen_value << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigen_vector[i] << " ";
  }
  cout << endl;
}

```

This feature allows to use a matrix whose elements are partially given,
e.g. a sparse matrix whose non-zero elements are stored
as a list of {row-index, column-index, value} tuples.
For detailed specs, see [Details](https://github.com/mrcdr/lambda-lanczos#details)

## Requirement

C++11 compatible environment


## Installation

**Lambda Lanczos** is a header-only library.
So the installation step is as follows:

1. Clone or download the latest version from [Github](https://github.com/mrcdr/lambda-lanczos/).
2. Place the `include` directory anywhere your project can find.


## Details
### Constructors
1. `LambdaLanczos(function<void(const vector<double>&, vector<double>&)> mv_mul, int matrix_size)`
2. `LambdaLanczos(function<void(const vector<double>&, vector<double>&)> mv_mul, int matrix_size, bool find_maximum)`

The first one is equivalent to `LambdaLanczos(mv_mul, matrix_size, false)`, means to calculate the smallest eigenvalue.

### Member variables of LambdaLanzcos
#### `int max_iteration`
controls the limit of Lanczos iteration count.

- **Default value** : matrix_size

#### `double eps`
is the convergence threshold of Lanczos iteration.
"`eps` = 1e-12" means the eigenvalue will be calculated with
12 digits of precision.

- **Default value** : 1e-12

#### `std::function<void(vector<double>&)> init_vector`
is the function used to initialize the first Lanczos vector.

- **Default value** : a function to initialize a vector randomly in the range of [-1, 1].

#### (Not necessary to change) `double tridiag_eps_ratio`
controls the the convergence threshold of the "bisection routine" in
the Lanczos algorithm, 
which finds the eigenvalue of an approximated tridiagonal matrix.

- **Default value** : 1e-1

#### (Not necessary to change)  `int initial_vector_size`
controls the initial size of Lanczos vectors.

- **Default value** : 200

## Licence

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)
