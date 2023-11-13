#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <thread>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "cooling.h"
#include "distributions.h"
#include "heatbath.h"
#include "lattice.h"
#include "su2.h"

using namespace std;

Cooling::Cooling() {}
Cooling::~Cooling() {}

// see M.D'Elia: https://arxiv.org/pdf/hep-lat/9605013.pdf
// see (maybe) Teper: https://arxiv.org/pdf/hep-lat/9909124.pdf

void Cooling::Cooling_update(Lattice &U) {
  Heatbath HB;
  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {

            SU3Matrix Utemp = U(x, y, z, t, mu);

            SU3Matrix W;
            W.zeros();
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
            }

            SU3Matrix R = CabibboMarinariProjection(W, "R");
            W = R * W;
            SU3Matrix S = CabibboMarinariProjection(W, "S");
            W = S * W;
            SU3Matrix T = CabibboMarinariProjection(W, "T");
            U(x, y, z, t, mu) = R * S * T * W;
            U(x, y, z, t, mu).unitarize();
          }
        }
      }
    }
  }
}
SU3Matrix Cooling::CabibboMarinariProjection(SU3Matrix W,
                                             const string subgroup) {
  // CHECKED: all matrices are SU(3)
  // this maximizes the trace of submatrices R S T to reproject them in SU(3)
  SU3Matrix SubMat;
  if (subgroup == "R") {

    double temp = pow(abs(conj(W(0, 0)) + W(1, 1)), 2) +
                  pow(abs(conj(W(1, 0)) - W(0, 1)), 2);
    double OneOverDetR = 1.0 / sqrt(temp);

    SU3Matrix::Matrix elem_R = {{{{(conj(W(0, 0)) + W(1, 1)) * OneOverDetR,
                                   (conj(W(1, 0)) - W(0, 1)) * OneOverDetR, 0}},
                                 {{(conj(W(0, 1)) - W(1, 0)) * OneOverDetR,
                                   (conj(W(1, 1)) + W(0, 0)) * OneOverDetR, 0}},
                                 {{0, 0, 1}}}};
    SubMat = SU3Matrix(elem_R);
  }

  if (subgroup == "S") {

    double temp = pow(abs(conj(W(0, 0)) + W(2, 2)), 2) +
                  pow(abs(conj(W(2, 0)) - W(0, 2)), 2);
    double OneOverDetS = 1.0 / sqrt(temp);

    SU3Matrix::Matrix elem_S = {{{{(conj(W(0, 0)) + W(2, 2)) * OneOverDetS, 0,
                                   (conj(W(2, 0)) - W(0, 2)) * OneOverDetS}},
                                 {{0, 1, 0}},
                                 {{(conj(W(0, 2)) - W(2, 0)) * OneOverDetS, 0,
                                   (conj(W(2, 2)) + W(0, 0)) * OneOverDetS}}}};
    SubMat = SU3Matrix(elem_S);
  }

  if (subgroup == "T") {
    double temp = pow(abs(conj(W(1, 1)) + W(2, 2)), 2) +
                  pow(abs(conj(W(2, 1)) - W(1, 2)), 2);
    double OneOverDetT = 1.0 / sqrt(temp);

    SU3Matrix::Matrix elem_T = {{{{1, 0, 0}},
                                 {{0, (conj(W(1, 1)) + W(2, 2)) * OneOverDetT,
                                   (conj(W(2, 1)) - W(1, 2)) * OneOverDetT}},
                                 {{0, (conj(W(1, 2)) - W(2, 1)) * OneOverDetT,
                                   (conj(W(2, 2)) + W(1, 1)) * OneOverDetT}}}};
    SubMat = SU3Matrix(elem_T);
  }

  return SubMat;
}

SU2Matrix Cooling::reconstructSU2(SU2Matrix M) {

  SU2Matrix::Matrix s_x_elem = {
      {{{Complex(0, 0), Complex(1, 0)}}, {{Complex(1, 0), Complex(0, 0)}}}};
  SU2Matrix::Matrix s_y_elem = {
      {{{Complex(0, 0), Complex(0, -1)}}, {{Complex(0, 1), Complex(0, 0)}}}};
  SU2Matrix::Matrix s_z_elem = {
      {{{Complex(1, 0), Complex(0, 0)}}, {{Complex(0, 0), Complex(-1, 0)}}}};
  SU2Matrix::Matrix Id_elem = {
      {{{Complex(1, 0), Complex(0, 0)}}, {{Complex(0, 0), Complex(1, 0)}}}};

  SU2Matrix sgm_x(s_x_elem);
  SU2Matrix sgm_y(s_y_elem);
  SU2Matrix sgm_z(s_z_elem);
  SU2Matrix Identita(Id_elem);

  double a = 0.5 * M.reTr();

  vector<double> b(3);
  b[0] = (M * sgm_x).reTr() * 0.5;
  b[1] = (M * sgm_y).reTr() * 0.5;
  b[2] = (M * sgm_z).reTr() * 0.5;

  // Normalize a and b
  double norm = std::sqrt(a * a + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  a /= norm;
  for (int i = 0; i < 3; ++i) {
    b[i] /= norm;
  }

  SU2Matrix normalized;

  for (int c = 0; c < 2; c++) {
    for (int d = 0; d < 2; d++) {
      complex<double> temp = a * Identita(c, d) + 1i * b[0] * sgm_x(c, d) +
                             1i * b[1] * sgm_y(c, d) + 1i * b[2] * sgm_z(c, d);

      double realpart = temp.real();
      double imagpart = temp.imag();

      normalized(c, d) = Complex(realpart, imagpart);
    }
  }
  return normalized;
}

// SU3Matrix Cooling::PolarDecomposition(SU3Matrix M) {

//   SU3Matrix H = (M.conjT() * M).matrixSqrt();
//   SU3Matrix U = M * H.inv();

//   return U;
// }

SU2Matrix Cooling::subgroup(SU3Matrix W, const string subgrp) {

  SU2Matrix Wsub;

  if (subgrp == "r") {
    SU2Matrix::Matrix elem_r = {{{{W(0, 0), W(0, 1)}}, {{W(1, 0), W(1, 1)}}}};
    Wsub = SU2Matrix(elem_r);
  }

  if (subgrp == "s") {
    SU2Matrix::Matrix elem_s = {{{{W(0, 0), W(0, 2)}}, {{W(2, 0), W(2, 2)}}}};
    Wsub = SU2Matrix(elem_s);
  }

  if (subgrp == "t") {
    SU2Matrix::Matrix elem_t = {{{{W(1, 1), W(1, 2)}}, {{W(2, 1), W(2, 2)}}}};
    Wsub = SU2Matrix(elem_t);
  }
  return Wsub;
}

SU3Matrix Cooling::reconstructSU3(SU2Matrix w, const string subgrp) {

  SU3Matrix W;
  if (subgrp == "R") {
    SU3Matrix::Matrix elem_R = {
        {{{w(0, 0), w(0, 1), 0}}, {{w(1, 0), w(1, 1), 0}}, {{0, 0, 1}}}};
    W = SU3Matrix(elem_R);
  }

  if (subgrp == "S") {
    SU3Matrix::Matrix elem_S = {
        {{{w(0, 0), 0, w(0, 1)}}, {{0, 1, 0}}, {{w(1, 0), 0, w(1, 1)}}}};
    W = SU3Matrix(elem_S);
  }

  if (subgrp == "T") {
    SU3Matrix::Matrix elem_T = {
        {{{1, 0, 0}}, {{0, w(0, 0), w(0, 1)}}, {{0, w(1, 0), w(1, 1)}}}};

    W = SU3Matrix(elem_T);
  }

  return W;
}