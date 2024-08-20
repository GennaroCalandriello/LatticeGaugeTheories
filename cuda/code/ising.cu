
#include <chrono>
#include <fstream>
// #include "include/getopt.h"
#include <iostream>
#include <string>
#include "cudamacro.h"

#include <cuda_fp16.h> // utilizzata per gestire operazioni su numeri a 16-bit, precisione ridotta
#include <curand.h>
#include <cublas_v2.h> //Basic Linear Algebra Subprograms

#include <cub/cub.cuh> //Fa parte della libreria CUB, che è una libreria di primitive CUDA per la
// gestione efficiente di dati, come le operazioni di riduzione, scansione, ricerca e ordinamento.
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#define TCRIT 2.269185314213022 // Temperatura critica per il modello di Ising 2D
#define THREADS 128

//initialize lattice spins
__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
                             const long long nx,    
                             const long long ny) {
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x+threadIdx.x; // tid è l'indice unico del thread corrente, 
    //calcolato combinando l'indice del blocco (blockIdx.x) e l'indice del thread all'interno del blocco (threadIdx.x).
    if (tid >= nx*ny) return;

    float randval = randvals[tid];
    signed char val = (randval<0.5f) ? -1:1;
    lattice[tid] = val;

                             }
template<bool is_black>
__global__ void update_lattice(signed char* lattice, 
                                const signed char* __restrict__ op_lattice,
                                const float* __restrict__ randvals,
                                const float inv_temp,
                                const long long nx,
                                const long long ny) {
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x+threadIdx.x;
    const int i = tid/ny;
    const int j = tid%ny;


    if (i>=nx || j>=ny) return;

    //set stencil indices and PBC
    int ipp = (i+1<nx) ? i+1:0;
    int inn = (i-1>=0) ? i-1:nx-1;
    int jpp = (j+1<ny) ? j+1:0;
    int jnn = (j-1>=0) ? j-1:ny-1;

    // select off-column index based on color and row index parity
    int joff;
    if (is_black) {
        joff = (i%2) ? jpp: jnn;
    } else {
        joff = (i%2) ? jnn: jpp;
    }

    //compute sum of nn interactions
    signed char nn_sum = op_lattice[inn*ny+j] +op_lattice[i*ny+j] +op_lattice[ipp*ny+j] +op_lattice[i*ny+joff];

    // determine whether to flip spin Metropolis step
    signed char lij = lattice[i*ny+j];
    float acceptance_ratio = exp(-2.0f*inv_temp*nn_sum*lij);
    if (randvals[i*ny+j] < acceptance_ratio) {
        lattice[i*ny+j] = -lij;
             }
     }

     //write lattice conf on file
     void write_lattice(signed char* lattice_b, signed char* lattice_w, std:: string filename, long long nx, long long ny) {
        printf("Writing lattice configuration to file %s\n", filename.c_str());
        signed char *lattice_h, *lattice_b_h, *lattice_w_h;
        lattice_h = (signed char*)malloc(nx*ny*sizeof(*lattice_h));
        lattice_b_h = (signed char*)malloc(nx*ny/2*sizeof(*lattice_b_h));
        lattice_w_h = (signed char*)malloc(nx*ny/2*sizeof(*lattice_w_h));

        CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx*ny/2*sizeof(*lattice_b), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_b, nx*ny/2*sizeof(*lattice_w), cudaMemcpyDeviceToHost));

        for (int i =0; i<nx; i++){
            for (int j =0; j <ny/2; j++){
                if (i%2) {
                    lattice_h[i*ny+2*j+1] = lattice_b_h[i*ny/2+j];
                    lattice_h[i*ny+2*j] = lattice_w_h[i*ny/2+j];
                } else {
                    lattice_h[i*ny+2*j] = lattice_b_h[i*ny/2+j];
                    lattice_h[i*ny+2*j+1] = lattice_w_h[i*ny/2+j];
                }
            }
        }
        std::ofstream f;
        f.open(filename);
        if (f.is_open()) {
            for (int i =0; i<nx; i++){
                for (int j =0; j<ny; j++){
                    f <<(int)lattice_h[i*ny+j] << " ";
                }
                f<<std::endl;
            }
        }
        f.close();

        free(lattice_h);
        free(lattice_b_h);
        free(lattice_w_h);

     }

     void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, float inv_temp, long long nx, long long ny){
        //setup CUDA launch configuration
        int blocks = (nx*ny/2 +THREADS-1) /THREADS;

        //update black
        CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
        update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals, inv_temp, nx, ny/2);

        //update white
        CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
        update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals, inv_temp, nx, ny/2);
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

int main(int argc, char **argv){
    //defaults
    long long nx = 10120;
    long long ny = 10120;
    float alpha = 0.5f;
    int nwarmup = 100;
    int niters = 1000;
    bool write = true;
    unsigned long long seed = 1234ULL;

  
    //check arguments
    if (nx%2!=0 || ny%2 !=0) {
        fprintf(stderr, "ERRORE SCEMO: Lattice dimensions must be even values.\n");
        exit(EXIT_FAILURE);
    }

    float inv_temp = 1.0f/(alpha*TCRIT); //inverso della temperatura

    //setup cuRAND generator
    curandGenerator_t rng;
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
    float *randvals;
    CHECK_CUDA(cudaMalloc(&randvals, nx*ny/2 * sizeof(*randvals)));


    //setup black and w lattic arrais on device
    signed char *lattice_b, *lattice_w;
    CHECK_CUDA(cudaMalloc(&lattice_b, nx*ny/2*sizeof(*lattice_b)));
    CHECK_CUDA(cudaMalloc(&lattice_w, nx*ny/2*sizeof(*lattice_w)));

    int blocks = (nx*ny/2+THREADS-1)/THREADS;
    CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
    init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2);

    //warmup iterations --> termalizzazione
    printf("Starting termalizzazione...\n");
    for (int i =0; i<nwarmup; i++){
        update(lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    printf("starting trial iterations...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
     
    for (int i = 0; i< niters; i++){
        update(lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny);
        if (i%10==0) printf("Completed %d/%d iterations... \n", i+1, niters);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("REPORT:\n");
    printf("\tnGPUs: %d\n", 1);
    printf("\ttemperature: %f*%f\n", alpha, TCRIT);
    printf("\telapsed time: %f sec\n", duration*1e-6);


   // Reduce
  double* devsum;
  int nchunks = (nx * ny/2 + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&devsum, 2 * nchunks * sizeof(*devsum)));
  size_t cub_workspace_bytes = 0;
  void* workspace = NULL;
  CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice_b, devsum, CUB_CHUNK_SIZE));
  CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_b[i*CUB_CHUNK_SIZE], devsum + 2*i,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_w[i*CUB_CHUNK_SIZE], devsum + 2*i + 1,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
  }

  double* hostsum;
  hostsum = (double*)malloc(2 * nchunks * sizeof(*hostsum));
  CHECK_CUDA(cudaMemcpy(hostsum, devsum, 2 * nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
  double fullsum = 0.0;
  for (int i = 0; i < 2 * nchunks; i++) {
    fullsum += hostsum[i];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny)) << std::endl;

  if (write) write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);

  return 0;
}