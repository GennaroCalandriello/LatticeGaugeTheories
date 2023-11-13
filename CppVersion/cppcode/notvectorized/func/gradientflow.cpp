// #include "OverRelaxation.h"
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "cooling.h"
#include "distributions.h"
#include "gradientflow.h"
#include "heatbath.h"
#include "lattice.h"
#include "su2.h"

using namespace std;
// g++ -fopenmp -g3 -Wall -I/eigen/Eigen gradientflow.cpp SU3Matrix.cpp
// cooling.cpp lattice.cpp WilsonAction.cpp heatbath.cpp su2.cpp
// distributions.cpp -o gradientflow.exe
// WITHOUT EIGEN LIBRARY g++ -fopenmp -g3 -Wall  gradientflow.cpp SU3Matrix.cpp
// cooling.cpp lattice.cpp WilsonAction.cpp heatbath.cpp su2.cpp
// distributions.cpp -o gradientflow.exe
Flow::Flow(){};
Flow::~Flow(){};
int topocharge = 0;

void Flow::Flow_update(Lattice &U) {
  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {

            SU3Matrix W;
            vector<int> a_mu(4, 0);
            a_mu[mu] = 1;

            for (int nu = 0; nu < dir; nu++) {
              vector<int> a_nu(4, 0);
              a_nu[nu] = 1;
              if (nu == mu) {
                continue;
              }
              W += U((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns, (z + a_mu[2]) % Ns,
                     (t + a_mu[3]) % Nt, nu) *
                   U((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns, (z + a_nu[2]) % Ns,
                     (t + a_nu[3]) % Nt, mu)
                       .conjT() *
                   U(x, y, z, t, nu).conjT();

              W += U((x + a_mu[0] - a_nu[0] + Ns) % Ns,
                     (y + a_mu[1] - a_nu[1] + Ns) % Ns,
                     (z + a_mu[2] - a_nu[2] + Ns) % Ns,
                     (t + a_mu[3] - a_nu[3] + Nt) % Nt, nu)
                       .conjT() *
                   U((x - a_nu[0] + Ns) % Ns, (y - a_nu[1] + Ns) % Ns,
                     (z - a_nu[2] + Ns) % Ns, (t - a_nu[3] + Nt) % Nt, mu)
                       .conjT() *
                   U((x - a_nu[0] + Ns) % Ns, (y - a_nu[1] + Ns) % Ns,
                     (z - a_nu[2] + Ns) % Ns, (t - a_nu[3] + Nt) % Nt, nu);

              // W += U((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns, (z + a_mu[2]) %
              // Ns,
              //        (t + a_mu[3]) % Nt, nu) *
              //      U((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns, (z + a_nu[2]) %
              //      Ns,
              //        (t + a_nu[3]) % Nt, mu)
              //          .conjT() *
              //      U(x, y, z, t, nu).conjT();

              // W += U(posMod(x + a_mu[0] - a_nu[0], Ns),
              //        posMod(y + a_mu[1] - a_nu[1], Ns),
              //        posMod(z + a_mu[2] - a_nu[2], Ns),
              //        posMod(t + a_mu[3] - a_nu[3], Nt), nu)
              //          .conjT() *
              //      U(posMod(x - a_nu[0], Ns), posMod(y - a_nu[1], Ns),
              //        posMod(z - a_nu[2], Ns), posMod(t - a_nu[3], Nt), mu)
              //          .conjT() *
              //      U(posMod(x - a_nu[0], Ns), posMod(y - a_nu[1], Ns),
              //        posMod(z - a_nu[2], Ns), posMod(t - a_nu[3], Nt), nu);
            }

            SU3Matrix Wdagger = W.conjT();
            SU3Matrix Omega = U(x, y, z, t, mu) * Wdagger;
            SU3Matrix Z0 = ActionDerivative(Omega);
            SU3Matrix Zp0 = Z0 * 0.25 * dtau;
            SU3Matrix W1 = LuscherExp(Zp0) * U(x, y, z, t, mu);

            Omega = W1 * Wdagger;
            SU3Matrix Z1 = ActionDerivative(Omega);
            SU3Matrix Zp1 = Z1 * (8.0 / 9.0) * dtau - Z0 * (17.0 / 36.0) * dtau;
            SU3Matrix W2 = LuscherExp(Zp1) * W1;

            Omega = W2 * Wdagger;
            SU3Matrix Z2 = ActionDerivative(Omega);
            SU3Matrix Zp2 = Z2 * (3.0 / 4.0) * dtau - Z1 * (8.0 / 9.0) * dtau +
                            Z0 * (17.0 / 36.0) * dtau;

            SU3Matrix W3 = LuscherExp(Zp2) * W2;
            complex<double> det_W3 = (W3.det());
            // W3.unitarize();
            U(x, y, z, t, mu) = W3 / det_W3;

            // I'm trying with a GS
            // U(x, y, z, t, mu).gramSchmidtQR();
            // double det_U = (U(x, y, z, t, mu).det()).real();
            // int det_U_round = static_cast<int>(std::round(det_U));

            // if (det_U_round == -1) {
            //   SU3Matrix I_alpha = IdentityMatrix() * (-1);
            //   U(x, y, z, t, mu) *= I_alpha;
            // }

            // SU3Matrix Ide = U(x, y, z, t, mu) * U(x, y, z, t, mu).conjT();
            // Ide.print();
          }
        }
      }
    }
  }
}

double Flow::TopologicalCharge(Lattice &U) {

  double Q = 0;

#pragma omp parallel for collapse(2) reduction(+ : Q)
  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {

            vector<int> a_mu(4, 0);
            a_mu[mu] = 1;

            for (int nu = 0; nu < dir; nu++) {

              vector<int> a_nu(4, 0);
              a_nu[nu] = 1;

              for (int rho = 0; rho < dir; rho++) {

                vector<int> a_rho(4, 0);
                a_rho[rho] = 1;

                for (int sigma = 0; sigma < dir; sigma++) {

                  vector<int> a_sigma(4, 0);
                  a_sigma[sigma] = 1;

                  int eps = epsilon(mu, nu, rho, sigma);
                  if (eps != 0) {
                    SU3Matrix Pmunu;
                    SU3Matrix Prhosigma;
                    Pmunu = U(x, y, z, t, mu) *
                            U((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns,
                              (z + a_mu[2]) % Ns, (t + a_mu[3]) % Nt, nu) *
                            U((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns,
                              (z + a_nu[2]) % Ns, (t + a_nu[3]) % Nt, mu)
                                .conjT() *
                            U(x, y, z, t, nu).conjT();

                    Prhosigma =
                        U(x, y, z, t, rho) *
                        U((x + a_rho[0]) % Ns, (y + a_rho[1]) % Ns,
                          (z + a_rho[2]) % Ns, (t + a_rho[3]) % Nt, sigma) *
                        U((x + a_sigma[0]) % Ns, (y + a_sigma[1]) % Ns,
                          (z + a_sigma[2]) % Ns, (t + a_sigma[3]) % Nt, rho)
                            .conjT() *
                        U(x, y, z, t, sigma).conjT();
                    double q = ((Pmunu * Prhosigma).tr() *
                                static_cast<double>(eps) * Ncharge)
                                   .imag();
                    Q += q;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return Q;
}

SU3Matrix Flow::Plaq(Lattice &U, int x, int y, int z, int t, int mu, int nu) {
  SU3Matrix P;
  vector<int> a_mu(4, 0);
  a_mu[mu] = 1;
  vector<int> a_nu(4, 0);
  a_nu[nu] = 1;

  P = U(x, y, z, t, mu) *
      U((x + a_mu[0]) % Ns, (y + a_mu[1]) % Ns, (z + a_mu[2]) % Ns,
        (t + a_mu[3]) % Nt, nu) *
      U((x + a_nu[0]) % Ns, (y + a_nu[1]) % Ns, (z + a_nu[2]) % Ns,
        (t + a_nu[3]) % Nt, mu)
          .conjT() *
      U(x, y, z, t, nu).conjT();

  return P;
}

int Flow::epsilon(int mu, int nu, int rho, int sigma) {
  int eps = 0;
  eps = sign(sigma - mu) * sign(rho - mu) * sign(nu - mu) * sign(sigma - nu) *
        sign(rho - nu) * sign(sigma - rho);
  return eps;
}

int Flow::sign(int x) {
  if (x >= 0) {
    return 1;
  } else {
    return -1;
  }
  return 0;
}

SU3Matrix Flow::ActionDerivative(SU3Matrix Omega) {
  SU3Matrix V;
  V = (Omega - Omega.conjT()) * (-0.5) +
      (Omega - Omega.conjT()).tr() * (0.16666666666666);
  // cout << "trace" << V.tr() << endl;

  return V;
}

void write(vector<double> const &v, string const &filename) {

  std::ofstream file(filename);

  if (!file) {
    std::cerr << "Error opening output file" << endl;
  }

  // write on file
  for (const auto &num : v) {
    file << num << endl;
  }

  file.close();
  cout << "Written" << endl;
}

int main() {

  bool execute = true;

  if (execute == true) {

    Lattice U = fill();
    // call classes
    Flow flow;
    Heatbath HB;
    Cooling cool;

    HB.HB_update(U, 6.7);
    vector<double> Actions(NstepFlow, 0);
    vector<double> Qarr(NstepFlow, 0);

    for (int nstep = 0; nstep < NstepFlow; nstep++) {

      cout << "-------step------" << nstep << endl;

      // auto start = std::chrono::high_resolution_clock::now();
      // flow.Flow_update(U);
      // auto stop = std::chrono::high_resolution_clock::now();
      // auto duration1 =
      //     std::chrono::duration_cast<std::chrono::seconds>(stop - start);
      // cout << "TIME ----> Flow Updating step " << duration1.count() <<
      //"seconds
      // "
      //      << endl;

      double Wbefore = Wilson(U, 1, 1);

      auto start4 = std::chrono::high_resolution_clock::now();
      cool.Cooling_update(U);
      auto stop4 = std::chrono::high_resolution_clock::now();
      auto duration4 =
          std::chrono::duration_cast<std::chrono::seconds>(stop4 - start4);
      cout << "TIME ----> Cooling Updating step " << duration4.count()
           << " seconds" << endl;

      auto start1 = std::chrono::high_resolution_clock::now();
      double W = Wilson(U, 1, 1);
      auto stop1 = std::chrono::high_resolution_clock::now();
      auto duration2 =
          std::chrono::duration_cast<std::chrono::seconds>(stop1 - start1);
      cout << "TIME ----> Wilson calculus " << duration2.count() << " seconds"
           << endl;

      auto start2 = std::chrono::high_resolution_clock::now();
      double Q = flow.TopologicalCharge(U);
      auto stop2 = std::chrono::high_resolution_clock::now();
      auto duration3 =
          std::chrono::duration_cast<std::chrono::seconds>(stop2 - start2);
      cout << "TIME ----> Topological Charge " << duration3.count() << "seconds"
           << endl;

      Actions[nstep] = W;

      cout << "OBS ----> Wilson Action before: " << Wbefore << endl;
      cout << "OBS ----> Wilson Action: " << W << endl;

      cout << "OBS ----> Topological Charge: " << Q << endl;
      Qarr[nstep] = Q;
    }

    write(Actions, "Wilson11.txt");
    write(Qarr, "TopologicalCharge.txt");

    return 0;
  }

  if (execute == false) {

    Cooling C;

    SU3Matrix U1 = su3_generator();
    SU3Matrix U2 = su3_generator();
    SU3Matrix U3 = su3_generator();
    SU3Matrix U4 = su3_generator();
    SU3Matrix U5 = su3_generator();
    SU3Matrix U6 = su3_generator();

    SU3Matrix W = U1 * U2 * U3 + U4 * U5 * U6;
    W = W.conjT();
    SU3Matrix R = C.CabibboMarinariProjection(W, "R");
    W *= R;

    SU3Matrix S = C.CabibboMarinariProjection(W, "S");
    W *= S;

    SU3Matrix T = C.CabibboMarinariProjection(W, "T");
    W *= T;
    W.unitarize();
    SU3Matrix Prod = W;
    (Prod.conjT() * Prod).print();
    // cout << "proddaggerprod" << endl;
    // (Prod.conjT() * Prod).print();
    // cout << "detProd" << Prod.det() << endl;

    // SU3Matrix ProdW = Prod * W;

    // cout << "prodWdaggerprodw" << endl;
    // (ProdW.conjT() * ProdW).print();
    // cout << "detProdW" << ProdW.det() << endl;
  }
}
