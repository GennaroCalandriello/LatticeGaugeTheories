#ifndef COOLING_H
#define COOLING_H
#include "SU3Matrix.h"
#include "su2.h"
#include <complex>
#include <vector>

class Cooling {
public:
  using Complex = complex<double>;

  Cooling();
  ~Cooling();

  void Cooling_update(Lattice &U);
  SU2Matrix subgroup(SU3Matrix W, const string subgrp);
  SU3Matrix reconstructSU3(SU2Matrix w, const string subgrp);
  SU2Matrix reconstructSU2(SU2Matrix M);
  SU3Matrix PolarDecomposition(SU3Matrix M);
  SU3Matrix CabibboMarinariProjection(SU3Matrix W, const string subgroup);
};

#endif // COOLING_H