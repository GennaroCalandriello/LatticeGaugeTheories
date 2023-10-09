#include "OverRelaxation.h"

using namespace std;

overrelaxation::overrelaxation() {}
overrelaxation::~overrelaxation(){};

void overrelaxation::OR_update(Lattice &U) {
  for (int idx = 0; idx < N; idx++) {

    x = idx % Ns;
    y = (idx / Ns) % Ns;
    z = (idx / (Ns * Ns)) % Ns;
    t = (idx / (Ns * Ns * Ns)) % Nt;
    mu = idx / (Ns * Ns * Ns * Nt) % dir;

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

    double a = sqrt(A.det()) A = A / a;
    SU3Matrix Adagger = A.conjT();
    SU3Matrix Atemp = Adagger * a;
    SU3Matrix H = 
    //---------------------staple-------------
  }
}
