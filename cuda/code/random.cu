#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

//! Define threads per block as a power of 2
//! TILE_DIM must be an integral multiple of BLOCK_ROWS
#define TILE_DIM 32
#define BLOCK_ROWS 4
#define DEPTH 2
#define TCRIT 2.269185314213022f

__global__ void init_spins(signed char* lattice, curandState* state, const long long nx, const long long ny) {
    const long long i = blockIdx.y * blockDim.y + threadIdx.y;
    const long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx || j >= ny) return;

    long long idx = i * ny + j;
    curandState localState = state[idx];
    lattice[idx] = (curand_uniform(&localState) < 0.5f) ? -1 : 1;
    state[idx] = localState;
}

__global__ void init_rng(unsigned int seed, curandState* state, long long nx, long long ny) {
    long long idx = blockIdx.y * blockDim.y + threadIdx.y;
    long long idy = blockIdx.x * blockDim.x + threadIdx.x;
    long long id = idx * ny + idy;
    if (idx < nx && idy < ny) {
        curand_init(seed, id, 0, &state[id]);
    }
}
__global__ void kernel(int n, int* gpu_G, int* gpu_G_new, int* flag_changes_made, curandState* state, float inv_temp)
{
    //! Array in shared memory to store the examined moments
    //! (with their neighbors), every each iteration (part of gpu_G)
    __shared__ int sh_G[(TILE_DIM+2*DEPTH) * (BLOCK_ROWS+2*DEPTH)];

    //! Number of rows and cols of sh_G
    int sh_cols = (blockDim.x + 2*DEPTH);

    //! Moment's coordinates
    int mom_row = blockDim.y*blockIdx.y + threadIdx.y;
    int mom_col = blockDim.x*blockIdx.x + threadIdx.x;

    //! Moment's coordinates in shared array
    int sh_row = threadIdx.y + DEPTH;
    int sh_col = threadIdx.x + DEPTH;

    //The step of each thread
    int stepRow = blockDim.y *gridDim.y;
    int stepCol = blockDim.x *gridDim.x;

    //! The indices of the examined neighbors
    int idx_row, idx_col;

    //! Coordinates of the neighbors in shared array
    int neigh_row, neigh_col;

    //! Accessing the spins in the global lattice and "transfer" them in the shared matrix.
    for(int i=mom_row; i<n+DEPTH ;i+=stepRow)
    {
        for(int j=mom_col; j<n+DEPTH;j+=stepCol)
        {
            //! Every thread read its own element in shared memory
            sh_G[sh_row*sh_cols+sh_col] = gpu_G[((i + n)%n)*n + ( (j + n)%n )];

            //! Add left and right neighbors
            if(threadIdx.x < DEPTH)
            {
                neigh_row = sh_row;
                idx_row = (i + n)%n;

                for(int p=0; p<2; p++)
                {
                    int adder = (p-1)*DEPTH + p*blockDim.x;
                    neigh_col = sh_col + adder;
                    idx_col = (j + adder + n)%n;
                    sh_G[neigh_row*sh_cols + neigh_col] = gpu_G[idx_row*n + idx_col];
                }
            }

            //! Add top and bottom neighbors
            if(threadIdx.y < DEPTH)
            {
                neigh_col = sh_col;
                idx_col = (j + n)%n;

                for(int p=0; p<2; p++)
                {
                    int adder = (p-1)*DEPTH + p*blockDim.y;
                    neigh_row = sh_row + adder;
                    idx_row = (i + adder + n)%n;
                    sh_G[neigh_row*sh_cols + neigh_col] = gpu_G[idx_row*n + idx_col];
                }
            }

            //! Add corner neighbors
            if( (threadIdx.x < DEPTH) && (threadIdx.y<DEPTH) )
            {
                for(int p=0; p<4; p++)
                {
                    int adder_row = (p%2 - 1)*DEPTH + (p%2)*blockDim.y;
                    neigh_row = sh_row + adder_row;
                    idx_row = (i + adder_row + n)%n;

                    int adder_col = ((p+3)%(p+1)/2 - 1)*DEPTH + ((p+3)%(p+1)/2)*blockDim.x;
                    neigh_col = sh_col + adder_col;
                    idx_col = (j + adder_col + n)%n;

                    sh_G[neigh_row*sh_cols + neigh_col] = gpu_G[idx_row*n + idx_col];
                }
            }

            //! Synchronize to make sure all threads have added what they were supposed to
            __syncthreads();

            if((i<n)&&(j<n))
            {
                //! Calculate energy using Metropolis-Hastings
                signed char spin = sh_G[sh_row*sh_cols + sh_col];
                signed char left = sh_G[sh_row*sh_cols + (sh_col - 1)];
                signed char right = sh_G[sh_row*sh_cols + (sh_col + 1)];
                signed char up = sh_G[(sh_row - 1)*sh_cols + sh_col];
                signed char down = sh_G[(sh_row + 1)*sh_cols + sh_col];

                float delta_energy = inv_temp * 2 * spin * (left + right + up + down);
                float prob = expf(-delta_energy);

                curandState localState = state[i * n + j];
                if (curand_uniform(&localState) < prob) {
                    gpu_G_new[i*n + j] = -spin;
                    atomicAdd(flag_changes_made, 1);
                } else {
                    gpu_G_new[i*n + j] = spin;
                }
                state[i * n + j] = localState;
            }

            //! Synchronize to make sure no thread adds next iteration's values
            __syncthreads();
        }
    }
}

void ising(int *G, int k, int n)
{
    //! Store G array to GPU
    int *gpu_G;
    cudaMalloc(&gpu_G, n*n*sizeof(int));
    cudaMemcpy(gpu_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);

    //! GPU array to store the updated values
    int *gpu_G_new;
    cudaMalloc(&gpu_G_new, n*n*sizeof(int));

    //! Temp pointer to swap gpu_G and gpu_G_new
    int *temp;

    //! Allocate memory for curandState
    curandState* d_state;
    cudaMalloc(&d_state, n*n*sizeof(curandState));

    //! Blocks & Threads
    dim3 threads( TILE_DIM, BLOCK_ROWS );
    dim3 blocks( n/threads.x, n/threads.x );

    //! Flag to see if changes were made (also store it to gpu to pass it to the kernel)
    int changes_made;
    int *gpu_changes_made;
    cudaMalloc(&gpu_changes_made, (size_t)sizeof(int));

    //! Initialize RNG states
    init_rng<<<blocks, threads>>>(12345, d_state, n, n);
    cudaDeviceSynchronize();

    //! Calculate inverse temperature
    float inv_temp = 1.0f / TCRIT;

    //! Implement the process for k iterations
    for(int i = 0; i < k; i++)
    {
        //! Initialize changes_made as zero
        changes_made = 0;
        cudaMemcpy(gpu_changes_made, &changes_made, (size_t)sizeof(int), cudaMemcpyHostToDevice);

        kernel<<< blocks , threads >>>(n, gpu_G, gpu_G_new, gpu_changes_made, d_state, inv_temp);

        //! Synchronize threads before swapping pointers
        cudaDeviceSynchronize();

        //! Swap pointers for next iteration
        temp = gpu_G;
        gpu_G = gpu_G_new;
        gpu_G_new = temp;

        //! Terminate if no changes were made
        cudaMemcpy(&changes_made, gpu_changes_made,  (size_t)sizeof(int), cudaMemcpyDeviceToHost);
        if(changes_made == 0)
            break;
    }

    //! Copy GPU final data to CPU memory
    cudaMemcpy(G, gpu_G, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    //! Free allocated GPU memory
    cudaFree(gpu_G);
    cudaFree(gpu_G_new);
    cudaFree(d_state);
    cudaFree(gpu_changes_made);
}

void write_lattice(int* lattice, int nx, int ny, const std::string& filename) {
    int* lattice_h = (int*)malloc(nx * ny * sizeof(*lattice_h));
    cudaMemcpy(lattice_h, lattice, nx * ny * sizeof(*lattice_h), cudaMemcpyDeviceToHost);

    std::ofstream f(filename);
    if (f.is_open()) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                f << lattice_h[i * ny + j] << " ";
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
    long long n= 1024;
    int k = 10000;
    long long grid_size = nx * ny;
    unsigned int seed = 12345;
    float alpha = 0.1f;
    float inv_temp = 1.0f / (TCRIT * alpha);

    signed char* lattice;
    cudaMalloc(&lattice, grid_size * sizeof(signed char));

    // Allocazione della memoria per curandState
    curandState* d_state;
    cudaMalloc(&d_state, grid_size * sizeof(curandState));

    // Allocazione della memoria per il contatore degli spin flippati
    int* d_flipped_spin_count;
    cudaMalloc(&d_flipped_spin_count, sizeof(int));
    cudaMemset(d_flipped_spin_count, 0, sizeof(int));  // Inizializzazione a 0

    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 numBlocks((ny + THREADS_X - 1) / THREADS_X, (nx + THREADS_Y - 1) / THREADS_Y);

    // Inizializzazione degli stati random per ogni thread
    init_rng<<<numBlocks, threadsPerBlock>>>(seed, d_state, nx, ny);
    cudaDeviceSynchronize();

    // Inizializzazione del reticolo
    init_spins<<<numBlocks, threadsPerBlock>>>(lattice, d_state, nx, ny);
    cudaDeviceSynchronize();


    // Esegui l'algoritmo di Ising su GPU
    // ising(lattice, k, n);

    // Scrivi il reticolo finale su file
    write_lattice(lattice, n, n, "final_lattice.txt");

    // Libera la memoria
    free(lattice);

    return 0;
}