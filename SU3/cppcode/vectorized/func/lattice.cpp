#include "lattice.h"
#include "SU3Matrix.h"
#include "const.h"
#include "distributions.h"
#include "su2.h"
#include <iostream>
#include <vector>

// execute with command: g++ -g3 -Wall lattice.cpp SU3Matrix.cpp
// distributions.cpp su2.cpp -o lattice.exe

using namespace std;

// fill U with su3 matrices:
Lattice fill() {
  Lattice U;
  for (int idx = 0; idx < N; idx++) {

    int x = idx % Ns;
    int y = (idx / Ns) % Ns;
    int z = (idx / (Ns * Ns)) % Ns;
    int t = (idx / (Ns * Ns * Ns)) % Nt;
    int mu = idx / (Ns * Ns * Ns * Nt) % dir;
    int i = U.index(x, y, z, t, mu);

    U(i) = su3_generator();
  }
  return U;
}

void printConfiguration(Lattice U) {

  Lattice instance;
  for (int idx = 0; idx < N; idx++) {

    int x = idx % Ns;
    int y = (idx / Ns) % Ns;
    int z = (idx / (Ns * Ns)) % Ns;
    int t = (idx / (Ns * Ns * Ns)) % Nt;
    int mu = idx / (Ns * Ns * Ns * Nt) % dir;
    int i = instance.index(x, y, z, t, mu);
    std::cout << "Matrix number " << idx << "direction " << mu << std::endl;
    cout << "det " << U(i).det() << endl;
    for (int a = 0; a < su3; a++) {
      for (int b = 0; b < su3; b++) {
        std::cout << U(i)(a, b) << " ";
      }
    }
  }
}

// Tested!!!! Works fine
// int main() {
//   Lattice U;
//   U = fill();
//   printConfiguration(U);
//   return 0;
// }
// Assuming Ns, Nt, and direction are defined constants

// std::vector<SU3Matrix> innermostVector(dir);
// std::vector<std::vector<SU3Matrix>> fourthDimension(Nt, innermostVector);
// std::vector<std::vector<std::vector<SU3Matrix>>> thirdDimension(
//     Ns, fourthDimension);
// std::vector<std::vector<std::vector<std::vector<SU3Matrix>>>>
// secondDimension(
//     Ns, thirdDimension);
// std::vector<std::vector<std::vector<std::vector<std::vector<SU3Matrix>>>>>
//     Conf(Ns, secondDimension);

// for (int x = 0; x < Ns; x++) {
//   for (int y = 0; y < Ns; y++) {
//     for (int z = 0; z < Ns; z++) {
//       for (int t = 0; t < Nt; t++) {
//         for (int mu = 0; mu < dir; mu++) {
//           cout << "indices" << x << y << z << t << mu << endl;
//           SU3Matrix temp = su3_generator();
//           for (int a = 0; a < su3; a++) {
//             for (int b = 0; b < su3; b++) {
//               Conf[x][y][z][t][mu](a, b) = temp(a, b);
//               cout << Conf[x][y][z][t][mu](a, b) << " ";
//             }
//           }
//         }
//       }
//     }
//   }
// }
// }
