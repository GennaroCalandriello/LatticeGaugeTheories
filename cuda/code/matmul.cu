#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Get matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is located col
// sub-matrices to the right and row sub-matrices down from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    //_____________________________________TIME START_______________________________
      // Creazione e inizio degli eventi per misurare il tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio della registrazione del tempo
    cudaEventRecord(start, 0);
    //_____________________________________TIME START_______________________________
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

     // Fine della registrazione del tempo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    //________________________________TIME STOP____________________________________
    // Calcolo del tempo trascorso
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo di esecuzione del kernel %f millisecondi:", milliseconds);
    //________________________________TIME STOP____________________________________
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices of A and B together and accumulate the result
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int k = 0; k < (A.width / BLOCK_SIZE); k++) {
        // Get sub-matrices Asub and Bsub
        Matrix Asub = GetSubMatrix(A, blockRow, k);
        Matrix Bsub = GetSubMatrix(B, k, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; j++) {
            Cvalue += As[row][j] * Bs[j][col];
        }

        // Synchronize to make sure that the preceding computation
        // is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

//main function
int main() {
    int width = 20000;
    int height = 20000;
    //alloca le matrici sull'host 
    Matrix A, B, C;
    A.width = B.width = C.width = width;
    A.height = B.height= C.height = height;
    A.stride = B.stride = C.stride = width;

    size_t size = width *height * sizeof(float);
    A.elements = (float*)malloc(size);
    B.elements = (float*)malloc(size);
    C.elements = (float*)malloc(size);

    //inizializzo con valori casuali
    for (int i = 0; i < width * height; i++) {
        // A.elements[i] = rand() / (float)RAND_MAX;
        // B.elements[i] = rand() / (float)RAND_MAX;
        A.elements[i] = 1.0;
        B.elements[i] = 1.0;
    }

    //moltiplico le matrici
    MatMul(A, B, C);
    //stampa i primi 5x5 elementi della matrice C
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", C.elements[i * width + j]);
        }
        printf("\n");
    }
    //libera la memoria dell'host
    free(A.elements);
    free(B.elements);
    free(C.elements);
    return 0;
}