#ifndef OVERRELAXATION_H
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

using namespace std;

class overrelaxation {
public:
  using Complex = complex<double>;

  overrelaxation();
  ~overrelaxation();
  void OR_update(Lattice &U);
  SU3Matrix &reflection(SU3Matrix &Ulink);
  void check_su3(SU3Matrix &Ulink);
};

#endif // OVERRELAXATION_H