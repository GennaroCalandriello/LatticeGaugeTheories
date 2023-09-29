#ifndef WILSONACTION_H
#define WILSONACTION_H

#include <complex>
#include <array>
#include "paulimatrices.h"
#include "SU3Matrix.h"


using namespace std;

SU3Matrix staple(Lattice U, int x, int y, int z, int t, int dir);
vector<int> index(int x, int y, int z, int t, int dir, int l, const vector<int> a_dir, const string& direction);

#endif // WILSONACTION_H