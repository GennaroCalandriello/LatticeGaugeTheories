#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H
#include "func/paulimatrices.h"
#include "func/lattice.h"
#include <complex>
#include <array>
#include <iostream>

using namespace std;
using Lattice = array<array<array<array<array<array<array<complex<double>, 3>, 3>, dir>, Nt>, Ns>, Ns>, Ns>;

Lattice initialize_lattice();


#endif // DISTRIBUTIONS_H