#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std;
// Function to calculate the Breit-Wigner probability density
double breitWigner(double E, double M, double Gamma) {
  const double k = (2 * sqrt(2) * M * Gamma * sqrt(M * M + Gamma * Gamma)) /
                   (M_PI * sqrt(M * M + (Gamma * Gamma) / 2));
  return k / ((E * E - M * M) * (E * E - M * M) + M * M * Gamma * Gamma);
}

// Function for rejection sampling from the Breit-Wigner distribution
void sampleBreitWigner(double M, double Gamma, double minE, double maxE,
                       int samples, int threadNum) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> distE(minE, maxE);
  std::uniform_real_distribution<> distP(0, 1);

  double maxP = breitWigner(M, M, Gamma);
  std::vector<double> results(samples);

  for (int i = 0; i < samples; ++i) {
    while (true) {
      double E = distE(gen);
      double P = breitWigner(E, M, Gamma);
      if (distP(gen) * maxP < P) {
        results[i] = E;
        break;
      }
    }
  }

  // Save the samples to a file for this thread
  std::cout << "Threads number: " << threadNum << std::endl;
  std::ofstream outFile("breit_wigner" + std::to_string(threadNum) + ".txt");
  for (double energy : results) {
    outFile << energy << std::endl;
  }
  outFile.close();
}

int main() {
  const double M = 11.1876;             // Z boson mass in GeV
  const double Gamma = 2.4952;          // Z boson width in GeV
  const int samplesPerThread = 1000000; // Number of samples per thread
  vector<double> energies = {10, 13, 70};
  const int numThreads = 5; // Total number of threads

  // Create and run threads

  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    double minE = M - (10 - i * 5) * Gamma;
    double maxE = M + (10 - i * 5) * Gamma;
    threads.push_back(std::thread(sampleBreitWigner, M, Gamma, minE, maxE,
                                  samplesPerThread, i));
  }
  // for loop on elements of energies
  //   for (size_t i = 0; i < energies.size(); ++i) {
  //     double minE = M - energies[i] * Gamma;
  //     double maxE = M + energies[i] * Gamma;
  //     threads.push_back(std::thread(sampleBreitWigner, M, Gamma, minE, maxE,
  //                                   samplesPerThread, i));
  //   }

  // Join threads
  for (auto &t : threads) {
    t.join();
  }

  std::cout << "Saved samples to separate files for each thread" << std::endl;

  return 0;
}
