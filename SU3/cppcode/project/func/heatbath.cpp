#include <array>
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

// compile with g++ -g3 -Wall heatbath.cpp lattice.cpp SU3Matrix.cpp su2.cpp
// distributions.cpp -o heatbath.exe

using namespace std;
using Complex = complex<double>;

Heatbath::Heatbath() {}
Heatbath::~Heatbath() {}

void Heatbath::HB_update(Lattice &U, double beta) {
  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {

            SU3Matrix A = staple(U, x, y, z, t, mu);
            SU3Matrix Utemp = U(x, y, z, t, mu);
            SU3Matrix W = Utemp * A;

            double a = (W.det()).real();

            if (a != 0) {

              // Cabibbo-Marinari pseudo-heatbath

              SU2Matrix r_ = heatbath_subgroup(W, beta, "r");
              SU3Matrix::Matrix R_elems = {{{{r_(0, 0), r_(0, 1), 0}},
                                            {{r_(1, 0), r_(1, 1), 0}},
                                            {{0, 0, 1}}}};
              SU3Matrix R(R_elems);

              W *= R;
              SU2Matrix s_ = heatbath_subgroup(W, beta, "s");
              SU3Matrix::Matrix S_elems = {{{{s_(0, 0), 0, s_(0, 1)}},
                                            {{0, 1, 0}},
                                            {{s_(1, 0), 0, s_(1, 1)}}}};
              SU3Matrix S(S_elems);

              W *= S;
              SU2Matrix t_ = heatbath_subgroup(W, beta, "t");
              SU3Matrix::Matrix T_elems = {{{{1, 0, 0}},
                                            {{0, t_(0, 0), t_(0, 1)}},
                                            {{0, t_(1, 0), t_(1, 1)}}}};
              SU3Matrix T(T_elems);

              SU3Matrix Uprime = T * S * R * Utemp;
              U(x, y, z, t, mu) = Uprime;
            }

            else {
              U(x, y, z, t, mu) = su3_generator();
            }
          }
        }
      }
    }
  }
}

SU2Matrix Heatbath::heatbath_subgroup(SU3Matrix W, double beta,
                                      const string subgrp) {

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

  vector<double> w = getA(Wsub);
  double a = sqrt(abs(Wsub.det()));

  SU2Matrix wbar = quaternion(normalize(w));
  SU2Matrix return_matrix;

  if (a != 0) {
    vector<double> avec = sampleA(a, beta);
    SU2Matrix a_quat = quaternion(avec);
    SU2Matrix Wsub_new = a_quat * wbar.conjT();
    return_matrix = Wsub_new;
  }

  else {
    return_matrix = su2_generator();
  }

  return return_matrix;
}

vector<double> Heatbath::getA(SU2Matrix W) {

  double a0 = 0.5 * (W(0, 0) + W(1, 1)).real();
  double a1 = 0.5 * (W(0, 1) + W(1, 0)).imag();
  double a2 = 0.5 * (W(0, 1) - W(1, 0)).real();
  double a3 = 0.5 * (W(0, 0) - W(1, 1)).imag();

  vector<double> Avec = {a0, a1, a2, a3};

  return Avec;
}

SU2Matrix Heatbath::quaternion(vector<double> vec) {

  Complex a00 = Complex(vec[0], vec[3]);
  Complex a01 = Complex(vec[2], vec[1]);
  Complex a10 = Complex(-vec[2], vec[1]);
  Complex a11 = Complex(vec[0], -vec[3]);

  SU2Matrix::Matrix quat_elem = {{{{a00, a01}}, {{a10, a11}}}};
  SU2Matrix quat(quat_elem);

  return quat;
}

vector<double> Heatbath::normalize(vector<double> v) {

  double sum = 0;

  for (size_t i = 0; i < v.size(); i++) {
    sum += v[i] * v[i];
  }

  vector<double> v_norm(v.size(), 0);

  for (size_t i = 0; i < v.size(); i++) {
    v_norm[i] = v[i] / sqrt(sum);
  }

  return v_norm;
}

vector<double> Heatbath::sampleA(double a, double beta) {

  /**choose a0 with P(a0) \sim sqrt(1-a0^2)* exp(beta*k* a0)
   */

  double e = exp(-2 * beta * a);
  double xtrial = uniform_(0, 1) * (1 - e) + e;
  double a0 = 1 + log(xtrial) / (beta * a);

  while (sqrt(1 - a0 * a0) < uniform_(0, 1)) {
    double xtrial = uniform_(0, 1) * (1 - e) + e;
    double a0 = 1 + log(xtrial) / (beta * a);
  }

  double r = sqrt(1 - a0 * a0);
  double a1 = gauss_(0, 1);
  double a2 = gauss_(0, 1);
  double a3 = gauss_(0, 1);

  double norm = sqrt(a1 * a1 + a2 * a2 + a3 * a3);

  a1 = a1 * r / norm;
  a2 = a2 * r / norm;
  a3 = a3 * r / norm;

  vector<double> avec = {a0, a1, a2, a3};
  vector<double> avec_norm = normalize(avec);

  return avec_norm;
}

// testing
int main() {
  cout << "quest è nu testt" << endl;
  Lattice U;
  U = fill();
  double Wilson_prima = Wilson(U, 1, 1);
  HB_update(U, 3.9);
  double Wilson_dopo = Wilson(U, 1, 1);

  cout << "Wilson prima e dopo" << endl;
  cout << Wilson_prima << endl;
  cout << Wilson_dopo << endl;

  return 0;
}