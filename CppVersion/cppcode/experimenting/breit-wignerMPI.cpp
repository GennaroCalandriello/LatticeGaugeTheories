#include "mpi.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Function to calculate the Breit-Wigner probability density
double breitWigner(double E, double M, double Gamma) {
  const double k = (2 * sqrt(2) * M * Gamma * sqrt(M * M + Gamma * Gamma)) /
                   (M_PI * sqrt(M * M + (Gamma * Gamma) / 2));
  return k / ((E * E - M * M) * (E * E - M * M) + M * M * Gamma * Gamma);
}

// Function for rejection sampling from the Breit-Wigner distribution
void sampleBreitWigner(double M, double Gamma, double minE, double maxE,
                       int samples, std::vector<double> &results,
                       std::mt19937 &gen,
                       std::uniform_real_distribution<> &distE,
                       std::uniform_real_distribution<> &distP) {
  double maxP = breitWigner(M, M, Gamma);
  for (int i = 0; i < samples; ++i) {
    while (true) {
      double E = distE(gen);               // Sample energy
      double P = breitWigner(E, M, Gamma); // Calculate BW probability
      if (distP(gen) * maxP < P) {
        results[i] = E; // Accept sample
        break;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const double M = 91.1876;          // Z boson mass in GeV
  const double Gamma = 2.4952;       // Z boson width in GeV
  const int totalSamples = 30000000; // Total samples to generate
  int samplesPerProcess = totalSamples / world_size;

  // Initialize random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distE(M - 10 * Gamma, M + 10 * Gamma);
  std::uniform_real_distribution<> distP(0, 1);

  std::vector<double> localResults(samplesPerProcess);

  // Each process runs its own sampling
  sampleBreitWigner(M, Gamma, M - 10 * Gamma, M + 10 * Gamma, samplesPerProcess,
                    localResults, gen, distE, distP);

  // Gather results at root process
  std::vector<double> allResults;
  if (world_rank == 0) {
    allResults.resize(totalSamples);
  }

  MPI_Gather(localResults.data(), samplesPerProcess, MPI_DOUBLE,
             allResults.data(), samplesPerProcess, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  // Root process saves all results to a file
  if (world_rank == 0) {
    std::ofstream outFile("breit_wigner_samples_mpi.txt");
    for (double energy : allResults) {
      outFile << energy << std::endl;
    }
    outFile.close();
  }
  std::cout << "Process " << world_rank << " started." << std::endl;

  MPI_Finalize();
  return 0;
}
