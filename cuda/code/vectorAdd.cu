#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
// #include <helper_cuda.h>
// the kernels execute on a GPU and the rest of the C++ program executes on a CPU.

__global__ void vectorAdd(const float *A, const float *B, float *c, int numElem) {
    int i = blockDim.x * blockIdx.x +threadIdx.x;

    if (i < numElem) {
        c[i] = A[i] + B[i] +0.0f;
    }
}

int main(void) {
    int numElem = 5000000;
    size_t size = numElem *sizeof(float);
    printf("Summing vectors of %d elements\n", numElem);
    cudaError_t err = cudaSuccess;

    //allocations
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    // verify allocations
     if(h_A ==NULL || h_B == NULL || h_C == NULL){
        fprintf(stderr, "Fallations");
        exit(EXIT_FAILURE);
     }

    //initialize host data
    for (int i =0; i< numElem; i++){
        h_A[i]= rand() /(float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    //allocate the device input vector A and B
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err!= cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err!=cudaSuccess) {
        fprintf(stderr, "Fallitoooo (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // launch the vector add CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElem +threadsPerBlock -1) /threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElem);
    err = cudaGetLastError();

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("Time elapsed: %f ms\n", duration.count());

    if (err !=cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //copy the result vector in device memory to the host result vector in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // verify the correctness of the result
    for (int i =0; i <numElem; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED mannaggia cristo \n");

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     // Print the results (for demonstration purposes, print only the first 10 elements)
    printf("Printing first 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }


    //free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}