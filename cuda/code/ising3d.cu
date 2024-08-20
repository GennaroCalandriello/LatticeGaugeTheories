#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#define CUB_CHUNK_SIZE ((1ll << 31) - (1ll << 28))
// #define TCRIT 2.26918531421f
#define TCRIT 4.5115f
#define THREADS 256

#include "cudamacro.h"

// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                           const float* __restrict__ randvals,
                           const long long nx,
                           const long long ny,
                           const long long nz) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= nx * ny * nz) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}

template<bool is_black>
__global__ void update_lattice(signed char* lattice,
                               const signed char* __restrict__ op_lattice,
                               const float* __restrict__ randvals,
                               const float inv_temp,
                               const long long nx,
                               const long long ny,
                               const long long nz) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = (tid / (ny * nz)) % nx;
  const int j = (tid / nz) % ny;
  const int k = tid % nz;

  if (i >= nx || j >= ny || k >= nz) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;
  int kpp = (k + 1 < nz) ? k + 1 : 0;
  int knn = (k - 1 >= 0) ? k - 1 : nz - 1;

  // Compute sum of nearest neighbor spins
  signed char nn_sum = op_lattice[inn * ny * nz + j * nz + k] +
                       op_lattice[ipp * ny * nz + j * nz + k] +
                       op_lattice[i * ny * nz + jnn * nz + k] +
                       op_lattice[i * ny * nz + jpp * nz + k] +
                       op_lattice[i * ny * nz + j * nz + knn] +
                       op_lattice[i * ny * nz + j * nz + kpp];

  // Determine whether to flip spin
  signed char lij = lattice[i * ny * nz + j * nz + k];
  float acceptance_ratio = exp(-2.0f * inv_temp * nn_sum * lij);
  if (randvals[tid] < acceptance_ratio) {
    lattice[tid] = -lij;
  }
}

// Write lattice configuration to file
void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny, long long nz) {
  printf("Writing lattice to %s...\n", filename.c_str());
  signed char *lattice_h, *lattice_b_h, *lattice_w_h;
  lattice_h = (signed char*) malloc(nx * ny * nz * sizeof(*lattice_h));
  lattice_b_h = (signed char*) malloc(nx * ny * nz / 2 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny * nz / 2 * sizeof(*lattice_w_h));

  CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx * ny * nz / 2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_b, nx * ny * nz / 2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz / 2; k++) {
        if (i % 2) {
          lattice_h[i * ny * nz + j * nz + 2 * k + 1] = lattice_b_h[i * ny * nz / 2 + j * nz / 2 + k];
          lattice_h[i * ny * nz + j * nz + 2 * k] = lattice_w_h[i * ny * nz / 2 + j * nz / 2 + k];
        } else {
          lattice_h[i * ny * nz + j * nz + 2 * k] = lattice_b_h[i * ny * nz / 2 + j * nz / 2 + k];
          lattice_h[i * ny * nz + j * nz + 2 * k + 1] = lattice_w_h[i * ny * nz / 2 + j * nz / 2 + k];
        }
      }
    }
  }

  std::ofstream f;
  f.open(filename);
  if (f.is_open()) {
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        for (int k = 0; k < nz; k++) {
          f << (int)lattice_h[i * ny * nz + j * nz + k] << " ";
        }
        f << std::endl;
      }
    }
  }
  f.close();

  free(lattice_h);
  free(lattice_b_h);
  free(lattice_w_h);
}

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, float inv_temp, long long nx, long long ny, long long nz) {

  // Setup CUDA launch configuration
  int blocks = (nx * ny * nz / 2 + THREADS - 1) / THREADS;

  // Update black
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx * ny * nz / 2));
  update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals, inv_temp, nx, ny, nz);

  // Update white
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx * ny * nz / 2));
  update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals, inv_temp, nx, ny, nz);
}

static void usage(const char *pname) {
  const char *bname = strrchr(pname, '/');
  if (!bname) {bname = pname;}
  else        {bname++;}

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--lattice-n <LATTICE_N>\n"
          "\t\tnumber of lattice rows\n"
          "\n"
          "\t-y|--lattice_m <LATTICE_M>\n"
          "\t\tnumber of lattice columns\n"
          "\n"
          "\t-z|--lattice-p <LATTICE_P>\n"
          "\t\tnumber of lattice depth\n"
          "\n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tnumber of warmup iterations\n"
          "\n"
          "\t-n|--niters <NITERS>\n"
          "\t\tnumber of trial iterations\n"
          "\n"
          "\t-a|--alpha <ALPHA>\n"
          "\t\tcoefficient of critical temperature\n"
          "\n"
          "\t-s|--seed <SEED>\n"
          "\t\tseed for random number generation\n"
          "\n"
          "\t-o|--write-lattice\n"
          "\t\twrite final lattice configuration to file\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
  // Defaults
  long long lattdim = 200;
  long long nx = lattdim;
  long long ny = lattdim;
  long long nz = lattdim;
  float alpha = 0.1f;
  int nwarmup = 1000;
  int niters = 2000;
  bool write = true;
  unsigned long long seed = 1234ULL;

  // Check arguments
  if (nx % 2 != 0 || ny % 2 != 0 || nz % 2 != 0) {
    fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
    exit(EXIT_FAILURE);
  }

  float inv_temp = 1.0f / (alpha * TCRIT);

  // Setup cuRAND generator
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
  float *randvals;
  CHECK_CUDA(cudaMalloc(&randvals, nx * ny * nz / 2 * sizeof(*randvals)));

  // Setup black and white lattice arrays on device
  signed char *lattice_b, *lattice_w;
  CHECK_CUDA(cudaMalloc(&lattice_b, nx * ny * nz / 2 * sizeof(*lattice_b)));
  CHECK_CUDA(cudaMalloc(&lattice_w, nx * ny * nz / 2 * sizeof(*lattice_w)));

  int blocks = (nx * ny * nz / 2 + THREADS - 1) / THREADS;
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx * ny * nz / 2));
  init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny, nz);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx * ny * nz / 2));
  init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny, nz);

  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny, nz);
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update(lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny, nz);
    if (i % 1000 == 0) printf("Completed %d/%d iterations...\n", i + 1, niters);
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", 1);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niters);
  printf("\tlattice dimensions: %lld x %lld x %lld\n", nx, ny, nz);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double)(nx * ny * nz) * niters / duration * 1e-3);

  // Reduce
  double* devsum;
  int nchunks = (nx * ny * nz / 2 + CUB_CHUNK_SIZE - 1) / CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&devsum, 2 * nchunks * sizeof(*devsum)));
  size_t cub_workspace_bytes = 0;
  void* workspace = NULL;
  CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice_b, devsum, CUB_CHUNK_SIZE));
  CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_b[i * CUB_CHUNK_SIZE], devsum + 2 * i,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny * nz / 2 - i * CUB_CHUNK_SIZE)));
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_w[i * CUB_CHUNK_SIZE], devsum + 2 * i + 1,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny * nz / 2 - i * CUB_CHUNK_SIZE)));
  }

  double* hostsum;
  hostsum = (double*)malloc(2 * nchunks * sizeof(*hostsum));
  CHECK_CUDA(cudaMemcpy(hostsum, devsum, 2 * nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
  double fullsum = 0.0;
  for (int i = 0; i < 2 * nchunks; i++) {
    fullsum += hostsum[i];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny * nz)) << std::endl;

  if (write) write_lattice(lattice_b, lattice_w, "final.txt", nx, ny, nz);

  return 0;
}
