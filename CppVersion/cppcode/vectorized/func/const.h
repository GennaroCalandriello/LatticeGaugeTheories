#ifndef CONST_H
#define CONST_H

using namespace std;

const double epsilon = 0.1;
const int su3 = 3;
const int su2 = 2;
const int Ns = 10;
const int Nt = 4;
const int dir = 4;
const int R = 1;
const int T = 1;
const int N = Ns * Ns * Ns * Nt * dir; // for the vectorization of the lattice

#endif // CONST_H