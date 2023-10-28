#ifndef HEATBATH_H
#define HEATBATH_H
#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "lattice.h"
#include "su2.h"

// Lattice HB_update(Lattice &U, double beta);

class Heatbath {
public:
  using Complex = complex<double>;

  Heatbath();
  ~Heatbath();

  void HB_update(Lattice &U, double beta);
  SU2Matrix heatbath_subgroup(SU3Matrix W, double beta, const string subgrp);
  vector<double> getA(SU2Matrix W);
  SU3Matrix staple(Lattice U, int x, int y, int z, int t, int mu);
  SU2Matrix quaternion(vector<double> vec);
  vector<double> normalize(vector<double> v);
  vector<double> sampleA(double a, double beta); // this is the heart of
                                                 // sampling
};

#endif // HEATBATH_H