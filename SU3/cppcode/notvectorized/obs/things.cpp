
#include <cmath>
#include <iostream>

int posMod(int x, int N) {
  //*ensures that the modulus operation result is always positive.*/

  int x_pos = (x % N + N) % N;
  // int x_pos = x % N;
  return (x_pos);
}

int main() {
  int x = 0;
  int y = 8;
  int Nn = 10;
  int res = posMod(x - y, Nn);
  std::cout << res << std::endl;
  std::cout << (x - y + Nn) % Nn << std::endl;
  return 1;
};