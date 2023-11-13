#ifndef GRADIENT_FLOW_H
#define GRADIENT_FLOW_H
#include <complex>
#include <vector>
using namespace std;

class Flow {
public:
  using Complex = complex<double>;

  Flow();
  ~Flow();
  void Flow_update(Lattice &U);
  SU3Matrix ActionDerivative(SU3Matrix const Omega);
  double TopologicalCharge(Lattice &U);
  int epsilon(int mu, int nu, int rho, int sigma);
  int sign(int x);
  SU3Matrix Plaq(Lattice &U, int x, int y, int z, int t, int mu, int nu);
};

void write(vector<double> const &v, string const &filename);

#endif // GRADIENT_FLOW_H