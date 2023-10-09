#ifndef WILSONACTION_H
#define WILSONACTION_H

#include "SU3Matrix.h"
#include "lattice.h"
#include "paulimatrices.h"
#include <array>
#include <complex>

using namespace std;

vector<int> index(int x, int y, int z, int t, int dir, int l,
                  vector<int> &a_dir, const string direction);
void PBC(vector<int> &a_dir);
double Wilson(Lattice &U, const int R, const int T);
// double Plaquette(Lattice U);
int posMod(int x, int N);

#endif // WILSONACTION_H