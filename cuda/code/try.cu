#include <iostream>
#include <cuda_runtime.h>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    // Size of vectors
    int N = 256;

    // Host input vectors
    float *h_A = new float[N];
    float *h_B = new float[N];
    // Host output vector
    float *h_C = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Device input vectors
    float *d_A, *d_B, *d_C;

    // Allocate memory for each vector on the GPU
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy array back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // Clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
