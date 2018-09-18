#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <lambda_lanczos/lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::lanczos;

void test1() {
  const int n = 3;
  double matrix[n][n] = { {-2.0, -1.0, -1.0},
                          {-1.0, -2.0, -1.0},
			  {-1.0, -1.0, -2.0} };
  /* Its eigenvalues are {-4, -1, -1} */  

  auto matmul = [&](double *in, double*out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  int itern;
  double eigvec[n];
  double eigvalue = lanczos(matmul, n, 1e-10, eigvec, itern);

  cout << "Eigen value: " << setprecision(10) << eigvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << " ";
  }
  cout << endl;
}

void test2() {
  const int n = 100;

  double offset = -10.0;
  auto matmul = [&](double *in, double*out) {
    for(int i = 0;i < n;i++) {
      out[i] += offset*in[i];
    }
    
    for(int i = 0;i < n-1;i++) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }

//    out[0] += -1.0*in[n-1];
//    out[n-1] += -1.0*in[0];
  };
  /*
    This lambda is equivalent to applying following n by n matrix ("off" means "offset"):
    off  -1   0   0     ..    0
     -1 off  -1   0     ..    0
      0  -1 off         ..    0
      0   0             ..    0
      0     ..      off  -1   0
      0     ..       -1 off  -1
      0     ..        0  -1 off

      Its smallest eigenvalue is -2*cos(pi/(n+1)) + offset
   */

  int itern;
  double* eigvec = new double[n];
  double eigvalue = lanczos(matmul, n, 1e-10, eigvec, itern) - offset;

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(10) << eigvalue << endl;  
  cout << "Eigen vector:" << endl;
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << endl;
  }
  cout << endl;

  delete[] eigvec;
}

int main() {
  test1();
  test2();

  return EXIT_SUCCESS;
}
