#ifndef CONST_H
#define CONST_H

using namespace std;

const double epsilon = 0.1;
const double beta = 6.2;
const int su3 = 3;
const int su2 = 2;
const int Ns = 20;
const int Nt = 4;
const int dir = 4;
const int R = 1;
const int T = 1;
const int N = Ns * Ns * Ns * Nt * dir; // for the vectorization of the lattice
const int n_heat = 10;                 // number of heatbath updates
const double dtau = 0.02;
const int NstepFlow = 15;
const double pi = 3.14159265358979323846;
const double Ncharge = -(1.0 / (32.0 * pi * pi));
const double beta_array[1] = {6.2};

#endif // CONST_H