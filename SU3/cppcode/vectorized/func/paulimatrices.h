// pauli_matrices.h
#ifndef PAULI_MATRICES_H
#define PAULI_MATRICES_H

#include <array>
#include <complex>

using namespace std;

// Define the Pauli matrices
const array<array<complex<double>, 2>, 2> s_x = {{{{0, 1}}, {{1, 0}}}};
const array<array<complex<double>, 2>, 2> s_y = {{{{0, -1i}}, {{1i, 0}}}};
const array<array<complex<double>, 2>, 2> s_z = {{{{1, 0}}, {{0, -1}}}};
const array<array<complex<double>, 2>, 2> Id = {{{{1, 0}}, {{0, 1}}}};

#endif // PAULI_MATRICES_H