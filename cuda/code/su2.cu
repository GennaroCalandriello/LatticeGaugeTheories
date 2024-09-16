#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>  // For time()

// Define the complex number structure
struct Complex {
    float real;
    float imag;

    __device__ __host__ Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}

    __device__ __host__ Complex operator+(const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }

    __device__ __host__ Complex operator-(const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }

    __device__ __host__ Complex operator*(const Complex& b) const {
        return Complex(real * b.real - imag * b.imag, real * b.imag + imag * b.real);
    }

    __device__ __host__ Complex conj() const {
        return Complex(real, -imag);
    }

    __device__ __host__ float normSquared() const {
        return real * real + imag * imag;
    }

    __device__ __host__ float realPart() const {
        return real;
    }
};

// Constants for the lattice size
const int Nx = 32;
const int Ny = 32;
const int Nz = 32;
const int Nt = 16;
const int lattice_size = Nx * Ny * Nz * Nt;  // 4D lattice total size

// CUDA Kernel to initialize SU(2) matrices for each direction using SoA format
__global__ void initialize_lattice(Complex *r0c0, Complex *r0c1, Complex *r1c0, Complex *r1c1, curandState *state, int seed, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global 1D index

    if (idx >= lattice_size * 4) return;  // Ensure idx is within bounds for 4 directions

    // Initialize random number generator
    curand_init(seed, idx, 0, &state[idx]);

    // Generate r0
    float r0 = curand_uniform(&state[idx]) - 0.5f;
    int sign = (r0 >= 0) ? 1 : -1;
    float x0 = sign * sqrt(1 - epsilon * epsilon);

    // Generate random vector r with 3 components
    float r[3];
    for (int i = 0; i < 3; ++i) {
        r[i] = curand_uniform(&state[idx]) - 0.5f;
    }

    // Normalize vector r
    float norm_r = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    float r_norm[3];
    for (int i = 0; i < 3; ++i) {
        r_norm[i] = epsilon * r[i] / norm_r;
    }

    // Define Pauli Matrices in Complex form inside the kernel
    Complex s_x[2][2] = { { Complex(0, 0), Complex(1, 0) }, { Complex(1, 0), Complex(0, 0) } };
    Complex s_y[2][2] = { { Complex(0, 0), Complex(0, -1) }, { Complex(0, 1), Complex(0, 0) } };
    Complex s_z[2][2] = { { Complex(1, 0), Complex(0, 0) }, { Complex(0, 0), Complex(-1, 0) } };
    Complex I[2][2]   = { { Complex(1, 0), Complex(0, 0) }, { Complex(0, 0), Complex(1, 0) } };

    // Construct SU(2) matrix based on x0 and normalized r
    Complex su2_matrix[2][2];
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            su2_matrix[a][b] = I[a][b] * x0
                             + s_x[a][b] * r_norm[0]
                             + s_y[a][b] * r_norm[1]
                             + s_z[a][b] * r_norm[2];
        }
    }

    // Store the SU(2) matrix elements in the SoA arrays for 4 directions
    r0c0[idx] = su2_matrix[0][0];  // First row, first column
    r0c1[idx] = su2_matrix[0][1];  // First row, second column
    r1c0[idx] = su2_matrix[1][0];  // Second row, first column
    r1c1[idx] = su2_matrix[1][1];  // Second row, second column
}

// CUDA Kernel to calculate the Wilson action including mu and nu directions and periodic boundary conditions
__global__ void calculate_wilson_action(Complex *r0c0, Complex *r0c1, Complex *r1c0, Complex *r1c1, float *action, int Nx, int Ny, int Nz, int Nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;

    int t = idx / (Nx * Ny * Nz);
    int z = (idx / (Nx * Ny)) % Nz;
    int y = (idx / Nx) % Ny;
    int x = idx % Nx;

    // Periodic boundary conditions
    int xp = (x + 1) % Nx;  // x + 1 with periodic boundary
    int yp = (y + 1) % Ny;  // y + 1 with periodic boundary
    int zp = (z + 1) % Nz;  // z + 1 with periodic boundary
    int tp = (t + 1) % Nt;  // t + 1 with periodic boundary

    // Loop over directions mu and nu for the plaquette calculation
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = mu + 1; nu < 4; ++nu) {
            // Compute index offsets for different directions
            int mu_offset = mu * lattice_size;
            int nu_offset = nu * lattice_size;

            // Load SU(2) matrices for the link in the mu direction at site idx
            Complex U_mu_0 = r0c0[idx + mu_offset];  // U_mu(0,0)
            Complex U_mu_1 = r0c1[idx + mu_offset];  // U_mu(0,1)
            Complex U_mu_2 = r1c0[idx + mu_offset];  // U_mu(1,0)
            Complex U_mu_3 = r1c1[idx + mu_offset];  // U_mu(1,1)

            // Load SU(2) matrices for the link in the nu direction at site idx + mu (shifted site)
            Complex U_nu_0 = r0c0[idx + nu_offset];  // U_nu(0,0)
            Complex U_nu_1 = r0c1[idx + nu_offset];  // U_nu(0,1)
            Complex U_nu_2 = r1c0[idx + nu_offset];  // U_nu(1,0)
            Complex U_nu_3 = r1c1[idx + nu_offset];  // U_nu(1,1)

            // Plaquette multiplication: U_mu * U_nu * U_mu_dagger * U_nu_dagger
            // (You need to perform SU(2) matrix multiplication here)

            // For simplicity, we'll calculate the trace (just adding up the real parts)
            float real_trace = U_mu_0.realPart() + U_mu_3.realPart();

            // Add to the Wilson action, subtracting from 1
            atomicAdd(action, 1.0f - real_trace);
        }
    }
}

// CUDA Kernel to calculate the determinant of each SU(2) matrix
__global__ void calculate_determinant(Complex *r0c0, Complex *r0c1, Complex *r1c0, Complex *r1c1, float *determinants) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;

    // Calculate the determinant of the SU(2) matrix at this lattice site
    // det(U) = (a0^2 + a3^2) - (a2^2 + a1^2)
    float det_real = r0c0[idx].normSquared() - r0c1[idx].normSquared();
    
    // Store the determinant value in the determinants array
    determinants[idx] = det_real;
}

// Host function to print the determinants
void print_determinants(float *determinants, int lattice_size) {
    for (int i = 0; i < lattice_size; ++i) {
        printf("Determinant of SU(2) Matrix at site %d: %f\n", i, determinants[i]);
    }
}

int main() {
    // Allocate memory for each SU(2) matrix component for each direction (SoA format)
    Complex *d_r0c0, *d_r0c1, *d_r1c0, *d_r1c1;
    float *d_action;
    curandState *d_rand_states;

    // Allocate GPU memory for the lattice components and random states
    // Multiply by 4 to account for 4 directions (mu = 0, 1, 2, 3)
    cudaMalloc(&d_r0c0, 4 * lattice_size * sizeof(Complex));  // Allocate memory for r0c0
    cudaMalloc(&d_r0c1, 4 * lattice_size * sizeof(Complex));  // Allocate memory for r0c1
    cudaMalloc(&d_r1c0, 4 * lattice_size * sizeof(Complex));  // Allocate memory for r1c0
    cudaMalloc(&d_r1c1, 4 * lattice_size * sizeof(Complex));  // Allocate memory for r1c1
    cudaMalloc(&d_rand_states, lattice_size * sizeof(curandState));  // For random number states
    cudaMalloc(&d_action, sizeof(float));  // For total Wilson action

    // Initialize the Wilson action to 0
    cudaMemset(d_action, 0, sizeof(float));

    // Define block and grid size for the kernel launch
    int blockSize = 256;
    int gridSize = (lattice_size + blockSize - 1) / blockSize;

    // Launch the kernel to initialize the lattice with SU(2) matrices in SoA format
    float epsilon = 0.0006f;  // Example epsilon value
    initialize_lattice<<<gridSize, blockSize>>>(d_r0c0, d_r0c1, d_r1c0, d_r1c1, d_rand_states, time(NULL), epsilon);

    // Synchronize to ensure kernel execution is finished
    cudaDeviceSynchronize();

    // Launch the kernel to calculate the Wilson action
    calculate_wilson_action<<<gridSize, blockSize>>>(d_r0c0, d_r0c1, d_r1c0, d_r1c1, d_action, Nx, Ny, Nz, Nt);

    // Allocate memory on the host (CPU) to copy the Wilson action
    float h_action = 0.0f;

    // Copy the Wilson action from GPU to CPU
    cudaMemcpy(&h_action, d_action, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the total Wilson action
    printf("Total Wilson Action: %f\n", h_action);

    // Free GPU memory
    cudaFree(d_r0c0);
    cudaFree(d_r0c1);
    cudaFree(d_r1c0);
    cudaFree(d_r1c1);
    cudaFree(d_rand_states);
    cudaFree(d_action);

    return 0;
}
