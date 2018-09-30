# Lambda Lanczos

Adaptive Lanczos algorithm library

## Overview

**Lambda Lanczos** calculates the smallest or largest eigenvalue and
corresponding eigenvector of a matrix.
The characteristic feature is the matrix-vector multiplication routine used in
Lanczos algorithm is adaptive:

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

## Requirement

C++11 compatible environment

## Installation

**Lambda Lanczos** is a header-only library.
So the installation step is as follows:

1. Clone or download the latest version from [Github](https://github.com/mrcdr/lambda-lanczos/).
2. Place the `include` directory anywhere your project can find.

## Options

## Author

[mrcdr](https://github.com/mrcdr)
