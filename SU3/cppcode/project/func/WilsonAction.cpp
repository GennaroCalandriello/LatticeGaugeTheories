#include <array>
#include <complex>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "lattice.h"

// execute with command: g++ -g3 -Wall WilsonAction.cpp lattice.cpp
// SU3Matrix.cpp distributions.cpp su2.cpp -o WilsonAction.exe

vector<int> index(int x, int y, int z, int t, int direct, int l,
                  vector<int> &a_dir, string direction) {

  if (direction == "f") {
    int xp = (x + l * a_dir[direct]);
    int yp = (y + l * a_dir[direct]);
    int zp = (z + l * a_dir[direct]);
    int tp = (t + l * a_dir[direct]);

    return {xp, yp, zp, tp};
  }

  if (direction == "b") {
    int xm = (x - l * a_dir[direct]);
    int ym = (y - l * a_dir[direct]);
    int zm = (z - l * a_dir[direct]);
    int tm = (t - l * a_dir[direct]);

    return {xm, ym, zm, tm};
  }
}

void PBC(vector<int> &a_dir) {
  a_dir[0] = a_dir[0] % Ns;
  a_dir[1] = a_dir[1] % Ns;
  a_dir[2] = a_dir[2] % Ns;
  a_dir[3] = a_dir[3] % Nt;
}

SU3Matrix staple(Lattice U, int x, int y, int z, int t, int mu) {
  SU3Matrix Wf;

  vector<int> a_mu(4, 0);
  a_mu[mu] = 1;
  vector<int> pmu = index(x, y, z, t, mu, 1, a_mu, "f");
  vector<int> mmu = index(x, y, z, t, mu, 1, a_mu, "b");

  for (int nu = 0; nu < dir; nu++) {
    vector<int> a_nu(4, 0);
    a_nu[nu] = 1;
    vector<int> pnu = index(x, y, z, t, nu, 1, a_nu, "f");
    vector<int> mnu = index(x, y, z, t, nu, 1, a_nu, "b");

    if (nu == mu) {
      continue;
    }
    vector<int> pmu_m_nu =
        index(pmu[0], pmu[1], pmu[2], pmu[3], nu, 1, a_nu, "b");
    PBC(pmu);
    PBC(pmu_m_nu);
    PBC(mnu);
    PBC(pnu);

    SU3Matrix W1 = U(pmu[0], pmu[1], pmu[2], pmu[3], nu) *
                   U(pnu[0], pnu[1], pnu[2], pnu[3], mu) *
                   (U(x, y, z, t, nu).conjT());

    SU3Matrix W2 =
        U(pmu_m_nu[0], pmu_m_nu[1], pmu_m_nu[2], pmu_m_nu[3], nu).conjT() *
        U(mnu[0], mnu[1], mnu[2], mnu[3], mu).conjT() *
        U(mnu[0], mnu[1], mnu[2], mnu[3], nu);

    Wf += W1 + W2; // should perform Wf += W1+W2
  }

  return Wf;
}

double Wilson(Lattice &U, int R, int T) {

  double S = 0;
  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int nu = 0; nu < dir; nu++) {
            vector<int> a_nu(4, 0);
            a_nu[nu] = 1;

            for (int mu = nu + 1; mu < dir; mu++)

            {
              vector<int> a_mu(4, 0);
              a_mu[mu] = 1;

              SU3Matrix I = IdentityMatrix();
              for (int i = 0; i < R; i++) {

                // U[r+i*a_mu]
                vector<int> p = index(x, y, z, t, mu, i, a_mu, "f");
                PBC(p);
                I *= U(p[0], p[1], p[2], p[3], mu);
              }
              for (int j = 0; j < T; j++) {

                // U[r+j*a_nu+T*a_mu]
                vector<int> p = index(x, y, z, t, nu, j, a_nu, "f");
                vector<int> pT =
                    index(p[0], p[1], p[2], p[3], mu, T, a_mu, "f");
                PBC(pT);

                I *= U(pT[0], pT[1], pT[2], pT[3], nu);
              }

              for (int i = R - 1; i >= 0; i--) {

                // U[r+i*a_mu+R*a_nu]
                vector<int> p = index(x, y, z, t, mu, i, a_mu, "f");
                vector<int> pR =
                    index(p[0], p[1], p[2], p[3], nu, R, a_nu, "f");
                PBC(pR);

                I *= U(pR[0], pR[1], pR[2], pR[3], mu).conjT();
              }

              for (int j = T - 1; j >= 0; j--) {

                // U[r+j*a_nu]
                vector<int> p = index(x, y, z, t, nu, j, a_nu, "f");
                PBC(p);
                I *= U(p[0], p[1], p[2], p[3], nu).conjT();
              }

              S += I.reTr() / su3;
            }
          }
        }
      }
    }
  }
  return S / (6 * Ns * Ns * Ns * Nt);
}

double Plaquette(Lattice U) {
  double S = 0;

  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {

          for (int mu = 0; mu < dir; mu++) {

            vector<int> a_mu(4, 0);
            a_mu[mu] = 1;

            for (int nu = 0; nu < dir; nu++) {
              if (nu == mu) {
                continue;
              }

              SU3Matrix temp;
              vector<int> a_nu(4, 0);
              a_nu[nu] = 1;

              temp += U((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns,
                        (z + a_mu[2]) % Ns, (t + a_mu[3]) % Nt, nu) *
                      U((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns,
                        (z + a_nu[2]) % Ns, (t + a_nu[3]) % Nt, mu)
                          .conjT() *
                      U(x, y, z, t, nu).conjT();

              temp +=
                  U((x + a_mu[0] - a_nu[0]) % Ns, (y + a_mu[1] - a_nu[1]) % Ns,
                    (z + a_mu[2] - a_nu[2]) % Ns, (t + a_mu[3] - a_nu[3]) % Nt,
                    nu)
                      .conjT() *
                  U((x - a_nu[0]) % Ns, (y - a_nu[1]) % Ns, (z - a_nu[2]) % Ns,
                    (t - a_nu[3]) % Nt, mu)
                      .conjT() *
                  U((x - a_nu[0]) % Ns, (y - a_nu[1]) % Ns, (z - a_nu[2]) % Ns,
                    (t - a_nu[3]) % Nt, nu);

              SU3Matrix P = U(x, y, z, t, mu) * temp;

              S += P.reTr() / su3;
            }
          }
        }
      }
    }
  }
  return S / (6 * Ns * Ns * Ns * Nt);
}

void test(Lattice U) {
  for (int x = 0; x < 1000; x++) {
    SU3Matrix prova = su3_generator();
    cout << prova.reTr() << endl;
  }
}

// WilsonAction must be recontrolled!
int main() {
  std::cout << "first plaquette" << std::endl;
  Lattice U;
  U = fill();
  double cazz = Wilson(U, 1, 1);
  cout << "questa è la dimensione del tuo cazzo" << cazz << endl;
}