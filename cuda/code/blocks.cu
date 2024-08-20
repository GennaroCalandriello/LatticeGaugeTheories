#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define THREADS_X 16
#define THREADS_Y 16

__global__ void init_spins(signed char* lattice, const long long nx, const long long ny, const long long offset) {
    const long long i = blockIdx.y * blockDim.y + threadIdx.y;
    const long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx || j >= ny) return;

    bool isBoundary = (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 || 
                       threadIdx.y == 0 || threadIdx.y == blockDim.y - 1);

    lattice[offset + i * ny + j] = isBoundary ? 0 : 1;
}

__global__ void exchange_boundaries(signed char* lattice, signed char* shared_memory, const long long nx, const long long ny, const long long offset) {
    const long long i = blockIdx.y * blockDim.y + threadIdx.y;
    const long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx || j >= ny) return;

    // Memoria condivisa per i bordi
    extern __shared__ signed char shared[];

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // Copia i valori del bordo nella memoria condivisa
    if (local_i == 0) shared[local_j] = lattice[offset + i * ny + j];
    if (local_j == 0) shared[blockDim.x + local_i] = lattice[offset + i * ny + j];

    if (local_i == blockDim.y - 1) shared[2 * blockDim.x + local_j] = lattice[offset + i * ny + j];
    if (local_j == blockDim.x - 1) shared[3 * blockDim.x + local_i] = lattice[offset + i * ny + j];

    // Sincronizza tutti i thread nel blocco
    __syncthreads();

    // Usa i dati della memoria condivisa per aggiornare i valori dei confini
    // (Esempio: scambio tra bordi)
    if (local_i == 0) lattice[offset + (i - 1) * ny + j] = shared[local_j];
    if (local_j == 0) lattice[offset + i * ny + j - 1] = shared[blockDim.x + local_i];

    if (local_i == blockDim.y - 1) lattice[offset + (i + 1) * ny + j] = shared[2 * blockDim.x + local_j];
    if (local_j == blockDim.x - 1) lattice[offset + i * ny + j + 1] = shared[3 * blockDim.x + local_i];
}

void write_lattice(signed char* lattice, int nx, int ny, const std::string& filename) {
    signed char* lattice_h = (signed char*)malloc(nx * ny * sizeof(*lattice_h));
    cudaMemcpy(lattice_h, lattice, nx * ny * sizeof(*lattice_h), cudaMemcpyDeviceToHost);

    std::ofstream f(filename);
    if (f.is_open()) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                f << (int)lattice_h[i * ny + j] << " ";
            }
            f << "\n";
        }
        f.close();
        std::cout << "Lattice written to " << filename << std::endl;
    } else {
        std::cerr << "Error opening file " << filename << std::endl;
    }

    free(lattice_h);
}

int main() {
    long long nx = 64;
    long long ny = 64;

    long long grid_size = nx * ny;
    signed char* lattice;
    cudaMalloc(&lattice, grid_size * sizeof(signed char));

    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 numBlocks((ny + THREADS_X - 1) / THREADS_X, (nx + THREADS_Y - 1) / THREADS_Y);

    init_spins<<<numBlocks, threadsPerBlock>>>(lattice, nx, ny, 0);
    cudaDeviceSynchronize();

    // Esegui l'exchange dei bordi utilizzando la memoria condivisa
    exchange_boundaries<<<numBlocks, threadsPerBlock, 4 * THREADS_X * sizeof(signed char)>>>(lattice, nullptr, nx, ny, 0);
    cudaDeviceSynchronize();

    write_lattice(lattice, nx, ny, "init.txt");

    cudaFree(lattice);

    return 0;
}
