#include "mpi.h"
#include <array>
#include <chrono>
#include <complex>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "SU3Matrix.h"
#include "WilsonAction.h"
#include "const.h"
#include "lattice.h"

// g++ -g3 -Wall WilsonActionCopy.cpp lattice.cpp SU3Matrix.cpp
// distributions.cpp su2.cpp -o WS
// C:\Users\vanho\Desktop\WorkingDirectory\C++\cppcode\notvectorized\func\SU3Matrix.cpp

// g++.exe -fdiagnostics-color=always -g WilsonActionCopy.cpp -I
// C:\Users\vanho\MPI\Include\ SU3Matrix.cpp lattice.cpp distributions.cpp
// su2.cpp -L C:\Users\vanho\MPI\Lib\x64\ -lmsmpi -o Wa
void timestamp() {

#define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time(NULL);
  tm_ptr = std::localtime(&now);
  std::strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);
  std::cout << time_buffer << "\n";
  return;
#undef TIME_SIZE
}

void PBC(vector<int> &a_dir) {
  a_dir[0] = posMod(a_dir[0], Ns);
  a_dir[1] = posMod(a_dir[1], Ns);
  a_dir[2] = posMod(a_dir[2], Ns);
  a_dir[3] = posMod(a_dir[3], Nt);
}

int posMod(int x, int N) {
  //*ensures that the modulus operation result is always positive.*/

  int x_pos = (x % N + N) % N;
  // int x_pos = x % N;
  return (x_pos);
}

double Wilson(Lattice &U, int R, int T) {

  double S = 0;
  double Stot = 0;

  int argc = 0;
  char **argv = NULL;
  MPI_Init(&argc, &argv);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // numproceses
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // id: 1, ..., numproceses-1

  int local_t_start = (Ns / world_size) * world_rank;
  int local_t_end =
      (world_rank == world_size - 1) ? Ns : local_t_start + (Ns / world_size);

  if (world_rank == 0) {
    timestamp();
    cout << "\n";
    cout << "WILSON ACTION\n";
    cout << "Using " << world_size << " processes.\n";
    cout << "local start-end" << local_t_start << " " << local_t_end << "\n";
  }

  double wtime;
  wtime = MPI_Wtime();

  for (int x = local_t_start; x < local_t_end; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {
            vector<int> a_mu(4, 0);
            a_mu[mu] = 1;

            for (int nu = mu + 1; nu < dir; nu++)

            {
              vector<int> a_nu(4, 0);
              a_nu[nu] = 1;

              SU3Matrix I = IdentityMatrix();
              for (int i = 0; i < R; i++) {

                // U[r+i*a_mu]
                I *= U(posMod(x + i * a_mu[0], Ns), posMod(y + i * a_mu[1], Ns),
                       posMod(z + i * a_mu[2], Ns), posMod(t + i * a_mu[3], Nt),
                       mu);
              }
              for (int j = 0; j < T; j++) {

                // U[r+j*a_nu+T*a_mu]
                I *= U(posMod(x + j * a_nu[0] + T * a_mu[0], Ns),
                       posMod(y + j * a_nu[1] + T * a_mu[1], Ns),
                       posMod(z + j * a_nu[2] + T * a_mu[2], Ns),
                       posMod(t + j * a_nu[3] + T * a_mu[3], Nt), nu);
              }

              for (int i = R - 1; i >= 0; i--) {

                // U[r+i*a_mu+R*a_nu].conjT()
                I *= U(posMod(x + i * a_mu[0] + R * a_nu[0], Ns),
                       posMod(y + i * a_mu[1] + R * a_nu[1], Ns),
                       posMod(z + i * a_mu[2] + R * a_nu[2], Ns),
                       posMod(t + i * a_mu[3] + R * a_nu[3], Nt), mu)
                         .conjT();
              }

              for (int j = T - 1; j >= 0; j--) {

                // U[r+j*a_nu].conjT()
                I *= U(posMod(x + j * a_nu[0], Ns), posMod(y + j * a_nu[1], Ns),
                       posMod(z + j * a_nu[2], Ns), posMod(t + j * a_nu[3], Nt),
                       nu)
                         .conjT();
              }

              S += I.reTr() / su3;
            }
          }
        }
      }
    }

    // MPI_Finalize();
  }
  S /= (6 * Ns * Ns * Ns * Nt);

  MPI_Allreduce(&S, &Stot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (world_rank == world_size - 1) {
    cout << " time elapsed: " << MPI_Wtime() - wtime << "\n";
    cout << "Wilson " << Stot << "\n";
    cout << "Wilson partial " << S << "\n";
    double Wil = WilsonNoMPI(U, 1, 1);
    cout << "WilsonNoMPI " << Wil << "\n";
  }
  MPI_Finalize();
  return Stot;
}

double WilsonNoMPI(Lattice &U, int R, int T) {

  double S = 0;

  for (int x = 0; x < Ns; x++) {
    for (int y = 0; y < Ns; y++) {
      for (int z = 0; z < Ns; z++) {
        for (int t = 0; t < Nt; t++) {
          for (int mu = 0; mu < dir; mu++) {
            vector<int> a_mu(4, 0);
            a_mu[mu] = 1;

            for (int nu = mu + 1; nu < dir; nu++)

            {
              vector<int> a_nu(4, 0);
              a_nu[nu] = 1;

              SU3Matrix I = IdentityMatrix();
              for (int i = 0; i < R; i++) {

                // U[r+i*a_mu]
                I *= U(posMod(x + i * a_mu[0], Ns), posMod(y + i * a_mu[1], Ns),
                       posMod(z + i * a_mu[2], Ns), posMod(t + i * a_mu[3], Nt),
                       mu);
              }
              for (int j = 0; j < T; j++) {

                // U[r+j*a_nu+T*a_mu]
                I *= U(posMod(x + j * a_nu[0] + T * a_mu[0], Ns),
                       posMod(y + j * a_nu[1] + T * a_mu[1], Ns),
                       posMod(z + j * a_nu[2] + T * a_mu[2], Ns),
                       posMod(t + j * a_nu[3] + T * a_mu[3], Nt), nu);
              }

              for (int i = R - 1; i >= 0; i--) {

                // U[r+i*a_mu+R*a_nu].conjT()
                I *= U(posMod(x + i * a_mu[0] + R * a_nu[0], Ns),
                       posMod(y + i * a_mu[1] + R * a_nu[1], Ns),
                       posMod(z + i * a_mu[2] + R * a_nu[2], Ns),
                       posMod(t + i * a_mu[3] + R * a_nu[3], Nt), mu)
                         .conjT();
              }

              for (int j = T - 1; j >= 0; j--) {

                // U[r+j*a_nu].conjT()
                I *= U(posMod(x + j * a_nu[0], Ns), posMod(y + j * a_nu[1], Ns),
                       posMod(z + j * a_nu[2], Ns), posMod(t + j * a_nu[3], Nt),
                       nu)
                         .conjT();
              }

              S += I.reTr() / su3;
            }
          }
        }
      }
    }
  }

  return S / (6 * Ns * Ns * Ns * Nt);
}

// WilsonAction must be recontrolled!
int main() {

  Lattice U = fill();

  // int argc = 0;
  // char **argv = NULL;
  // MPI_Init(&argc, &argv);

  // auto start4 = std::chrono::high_resolution_clock::now();
  double W = Wilson(U, 1, 1);
  // MPI_Finalize();

  // cout << W << endl;
  // auto stop4 = std::chrono::high_resolution_clock::now();
  // auto duration4 =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);
  // cout << "TIME ----> MPI step " << duration4.count() << " seconds" << endl;

  // auto start = std::chrono::high_resolution_clock::now();
  // double W1 = WilsonNoMPI(U, 1, 1);
  // cout << "gnogno " << W1 << endl;
  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  // cout << "TIME ----> NO MPI step " << duration.count() << " seconds" <<
  // endl;

  return 0;
}