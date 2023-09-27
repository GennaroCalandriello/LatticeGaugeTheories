// lattice.h
#ifndef LATTICE_H
#define LATTICE_H

#include <complex>
#include <array>

using namespace std;

const int Ns = 5;
const int Nt = 4;
const int dir = 4;
const int su3 = 3;
const int su2 = 2;

using Lattice = array<array<array<array<array<array<array<complex<double>, su3>, su3>, dir>, Nt>, Ns>, Ns>, Ns>;
using su2matr = array<array<complex<double>, 2>, 2>;





#endif // LATTICE_H