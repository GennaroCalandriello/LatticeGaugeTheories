#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv) {
  int err, size, rank;
  MPI_Status status;
  err = MPI_Init(&argc, &argv);
  /* error handling: */
  if (err != MPI_SUCCESS) {
    printf("Initialization of MPI failed!\n");
    return 1;
  }
  /* from now on the return values of MPI functions will be
 ignored */
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello World from process %d of %d processes\n", rank, size);
  err = MPI_Finalize();
  return 0;
}