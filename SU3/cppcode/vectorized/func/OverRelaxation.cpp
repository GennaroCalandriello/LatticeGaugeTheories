#include "OverRelaxation.h"
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "distributions.h"
#include "heatbath.h"
#include "lattice.h"
#include "su2.h"

using namespace std;

// g++ -g3 -Wall -I/eigen/Eigen OverRelaxation.cpp SU3Matrix.cpp lattice.cpp
// heatbath.cpp su2.cpp distributions.cpp -o OverRelaxation.exe

overrelaxation::overrelaxation() {}
overrelaxation::~overrelaxation(){};

SU3Matrix &overrelaxation::reflection(SU3Matrix &M) {

  int k = uniform_int_(1, 3);

  if (k == 1) {
    M(0, 1) = -M(0, 1);
    M(0, 2) = -M(0, 2);
    M(1, 0) = -M(1, 0);
    M(2, 0) = -M(2, 0);
  }

  if (k == 2) {
    M(0, 1) = -M(0, 1);
    M(1, 0) = -M(1, 0);
    M(1, 2) = -M(1, 2);
    M(2, 1) = -M(2, 1);
  }

  if (k == 3) {
    M(0, 2) = -M(0, 2);
    M(2, 0) = -M(2, 0);
    M(1, 2) = -M(1, 2);
    M(2, 1) = -M(2, 1);
  }
  return M;
}

void overrelaxation::OR_update(Lattice &U) {

  for (int idx = 0; idx < N; idx++) {

    int x = idx % Ns;
    int y = (idx / Ns) % Ns;
    int z = (idx / (Ns * Ns)) % Ns;
    int t = (idx / (Ns * Ns * Ns)) % Nt;
    int mu = idx / (Ns * Ns * Ns * Nt) % dir;

    SU3Matrix Utemp = U(U.index(x, y, z, t, mu));
    //---------------------staple-------------
    vector<int> a_mu(4, 0);
    a_mu[mu] = 1;
    SU3Matrix A;

    for (int nu = 0; nu < dir; nu++) {

      // SU3Matrix A1;
      // SU3Matrix A2;

      if (mu != nu) {

        vector<int> a_nu(4, 0);
        a_nu[nu] = 1;

        A += U(U.index((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns,
                       (z + a_mu[2]) % Ns, (t + a_mu[3]) % Nt, nu)) *
             U(U.index((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns,
                       (z + a_nu[2]) % Ns, (t + a_nu[3]) % Nt, mu))
                 .conjT() *
             U(U.index(x, y, z, t, nu)).conjT();

        A += U(U.index(posMod(x + a_mu[0] - a_nu[0], Ns),
                       posMod(y + a_mu[1] - a_nu[1], Ns),
                       posMod(z + a_mu[2] - a_nu[2], Ns),
                       posMod(t + a_mu[3] - a_nu[3], Nt), nu))
                 .conjT() *
             U(U.index(posMod(x - a_nu[0], Ns), posMod(y - a_nu[1], Ns),
                       posMod(z - a_nu[2], Ns), posMod(t - a_nu[3], Nt), mu))
                 .conjT() *
             U(U.index(posMod(x - a_nu[0], Ns), posMod(y - a_nu[1], Ns),
                       posMod(z - a_nu[2], Ns), posMod(t - a_nu[3], Nt), nu));

        // A += A1 + A2;
      }

      else {
        continue;
      }
    }

    Complex a = sqrt(A.det());
    A = A * (1.0 / a);
    SU3Matrix Adagger = A.conjT();
    SU3Matrix Atemp = Adagger * A;
    SU3Matrix H = Atemp.matrixSqrt();
    SU3Matrix O = A * H.inv();
    double det_O = (O.det()).real();
    int det_O_round = static_cast<int>(std::round(det_O));
    cout << "det_O_round " << det_O << endl;

    if (det_O_round == -1) {
      SU3Matrix I_alpha = IdentityMatrix() * (-1);
      O *= I_alpha;
    }

    SU3Matrix V = H.eigenvectors();

    SU3Matrix temp = V * Utemp * O * V.conjT();
    SU3Matrix Ureflected = reflection(temp);
    SU3Matrix Uprime = V.conjT() * Ureflected * V * O.conjT();
    U(U.index(x, y, z, t, mu)) = Uprime;
  }
}

int main() {

  Lattice U;
  U = fill();
  Heatbath HB;
  overrelaxation OR;

  double W1 = Wilson(U, 1, 1);
  cout << "Wilson action before update: " << W1 << endl;
  for (int i = 0; i < 20; i++) {

    HB.HB_update(U, 2);
    OR.OR_update(U);
    double Wil = Wilson(U, 1, 1);
    cout << "Wilson action after update: " << Wil << endl;
  }

  double W2 = Wilson(U, 1, 1);
  cout << "Wilson action after update: " << W2 << endl;

  return 0;
}
