#include "distributions.h"
#include <complex>
#include <cstdlib> // for srand
#include <ctime>
#include <iostream>
#include <random>

using namespace std;

double gauss_(double mean, double sigma) {
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<double> d(mean, sigma);

  return d(gen);
}

double uniform_(double low, double high) {
  // These static variables are initialized only once and retain their values
  // between function calls
  static random_device rd;
  static mt19937 gen(rd());
  uniform_real_distribution<double> d(low, high);

  return d(gen);
}

int uniform_int_(int lower, int upper) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> d(lower, upper);

  return d(gen);
}

// TESTED
// int main()
// {
//     cout << "quest è nu testt" << endl;
//     cout << gauss_() << endl;
//     cout << uniform_() << endl;
//     cout << uniform_int_(0, 2) << endl;

// }
// #include <random>
// #include <iostream>

// int main() {
//     // Create a random device
//     std::random_device rd;

//     // Create a generator, seeded with the random device
//     std::mt19937 gen(rd());

//     // Create a distribution, e.g. a uniform distribution between 0 and 100
//     std::uniform_int_distribution<> dist(0, 100);

//     // Generate a random number
//     int randomValue = dist(gen);

//     std::cout << randomValue << std::endl;

//     // ... rest of your code ...
// }