// Include guards to prevent double inclusion
#ifndef SU2_MATRIX_H
#define SU2_MATRIX_H

// Necessary includes
#include <complex>
#include <array>

// Use necessary namespaces or avoid it in header files for more encapsulation
using namespace std;
using Complex = complex<double>;

// Type definitions and constants
using ComplexMatrix2x2 = array<array<Complex, 2>, 2>;
const double epsilon = 0.1;

// Function prototypes
ComplexMatrix2x2 su2_matrix();

#endif // SU2_MATRIX_H