#include <iostream>
// #include <fstream>
// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <cmath>

// #define THREADS_X 16
// #define THREADS_Y 16
// #define TCRIT 2.269185314213022f
// #define BLOCK_SIZE 16  // Assume a square block size

// // Kernel per inizializzare gli stati random per ogni thread
// __global__ void init_rng(unsigned int seed, curandState* state, long long nx, long long ny) {
//     long long idx = blockIdx.y * blockDim.y + threadIdx.y;
//     long long idy = blockIdx.x * blockDim.x + threadIdx.x;
//     long long id = idx * ny + idy;
//     if (idx < nx && idy < ny) {
//         curand_init(seed, id, 0, &state[id]);
//     }
// }

// // Funzione per inizializzare il reticolo utilizzando random generator nel kernel
// __global__ void init_spins(signed char* lattice, curandState* state, const long long nx, const long long ny) {
//     const long long i = blockIdx.y * blockDim.y + threadIdx.y;
//     const long long j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= nx || j >= ny) return;

//     long long idx = i * ny + j;
//     curandState localState = state[idx];
//     lattice[idx] = (curand_uniform(&localState) < 0.5f) ? -1 : 1;
//     state[idx] = localState;
// }

// __global__ void update_lattice(signed char* lattice, curandState* state, const long long nx, const long long ny, float inv_temp, int* flipped_spin_count) {
//     const long long i = blockIdx.y * blockDim.y + threadIdx.y;
//     const long long j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= nx || j >= ny) return;

//     long long idx = i * ny + j;
//     curandState localState = state[idx];
//     signed char spin = lattice[idx];

//     // Calcolo dell'energia del sistema considerando i vicini con condizioni al contorno periodiche
//     signed char left = lattice[i * ny + (j - 1 + ny) % ny];
//     signed char right = lattice[i * ny + (j + 1) % ny];
//     signed char up = lattice[((i - 1 + nx) % nx) * ny + j];
//     signed char down = lattice[((i + 1) % nx) * ny + j];
//     float delta_energy = inv_temp * 2 * spin * (left + right + up + down);

//     // Calcolo della probabilità di accettazione
//     float prob = expf(-delta_energy);

//     // Accettazione della mossa e uso di operazione atomica
//     if (curand_uniform(&localState) < prob) {
//         lattice[idx] = -spin;  // Inverti lo spin
//         atomicAdd(flipped_spin_count, 1);  // Incrementa il contatore degli spin flippati in modo atomico
//     }
//     state[idx] = localState;  // Salva lo stato aggiornato
//     __syncthreads();
// }

// // Funzione per scrivere il reticolo su file
// void write_lattice(signed char* lattice, int nx, int ny, const std::string& filename) {
//     signed char* lattice_h = (signed char*)malloc(nx * ny * sizeof(*lattice_h));
//     cudaMemcpy(lattice_h, lattice, nx * ny * sizeof(*lattice_h), cudaMemcpyDeviceToHost);

//     std::ofstream f(filename);
//     if (f.is_open()) {
//         for (int i = 0; i < nx; i++) {
//             for (int j = 0; j < ny; j++) {
//                 f << (int)lattice_h[i * ny + j] << " ";
//             }
//             f << "\n";
//         }
//         f.close();
//         std::cout << "Lattice written to " << filename << std::endl;
//     } else {
//         std::cerr << "Error opening file " << filename << std::endl;
//     }

//     free(lattice_h);
// }

// int main() {
//     long long nx = 1024;
//     long long ny = 1024;
//     long long grid_size = nx * ny;
//     unsigned int seed = 12345;
//     float alpha = 0.1f;
//     float inv_temp = 1.0f / (TCRIT * alpha);

//     signed char* lattice;
//     cudaMalloc(&lattice, grid_size * sizeof(signed char));

//     // Allocazione della memoria per curandState
//     curandState* d_state;
//     cudaMalloc(&d_state, grid_size * sizeof(curandState));

//     // Allocazione della memoria per il contatore degli spin flippati
//     int* d_flipped_spin_count;
//     cudaMalloc(&d_flipped_spin_count, sizeof(int));
//     cudaMemset(d_flipped_spin_count, 0, sizeof(int));  // Inizializzazione a 0

//     dim3 threadsPerBlock(THREADS_X, THREADS_Y);
//     dim3 numBlocks((ny + THREADS_X - 1) / THREADS_X, (nx + THREADS_Y - 1) / THREADS_Y);

//     // Inizializzazione degli stati random per ogni thread
//     init_rng<<<numBlocks, threadsPerBlock>>>(seed, d_state, nx, ny);
//     cudaDeviceSynchronize();

//     // Inizializzazione del reticolo
//     init_spins<<<numBlocks, threadsPerBlock>>>(lattice, d_state, nx, ny);
//     cudaDeviceSynchronize();

//     for (int i = 0; i < 10000; i++) {
//         if (i % 1000 == 0) {
//             std::cout << "Step " << i << std::endl;
//         }
//         update_lattice<<<numBlocks, threadsPerBlock>>>(lattice, d_state, nx, ny, inv_temp, d_flipped_spin_count);
//         cudaDeviceSynchronize();
//     }

//     // Scarica il valore del contatore degli spin flippati
//     int h_flipped_spin_count;
//     cudaMemcpy(&h_flipped_spin_count, d_flipped_spin_count, sizeof(int), cudaMemcpyDeviceToHost);
//     std::cout << "Total flipped spins: " << h_flipped_spin_count << std::endl;
    
//     // Scrittura del reticolo finale su file
//     write_lattice(lattice, nx, ny, "final.txt");

//     // Pulizia della memoria
//     cudaFree(lattice);
//     cudaFree(d_state);
//     cudaFree(d_flipped_spin_count);

//     return 0;
// }
