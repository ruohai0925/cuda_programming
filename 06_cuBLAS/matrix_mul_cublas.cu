// CUDA Matrix Multiplication using cuBLAS & cuRAND
// Integrates: cuBLAS (Math), cuRAND (Data Gen), Column-Major Logic
// Original By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

// nvcc matrix_mul_cublas.cu -o matrix_mul_cublas -lcublas -lcurand

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// [1] Library Headers
// We need cuBLAS for math and cuRAND for generating data on the GPU.
#include <cublas_v2.h>
#include <curand.h>

// [2] Verification Function (CPU)
// CRITICAL: This function assumes COLUMN-MAJOR order to match cuBLAS.
// In C++ (Row-Major), index = i * N + j.
// In cuBLAS (Col-Major), index = j * N + i.
void verify_solution(float *a, float *b, float *c, int n) {
    float epsilon = 0.001f; // Tolerance for floating point errors
    for (int i = 0; i < n; i++) { // Row index
        for (int j = 0; j < n; j++) { // Col index
            float temp = 0.0f;
            for (int k = 0; k < n; k++) {
                // Standard Logic: A[i][k] * B[k][j]
                // Column-Major Indexing: 
                // A[i][k] -> a[k * n + i]  (Note: k is col, i is row)
                // B[k][j] -> b[j * n + k]  (Note: j is col, k is row)
                temp += a[k * n + i] * b[j * n + k];
            }
            // Result C[i][j] -> c[j * n + i]
            float gpu_val = c[j * n + i];
            // Compare absolute difference against epsilon
            assert(fabs(temp - gpu_val) < epsilon);
        }
    }
}

int main() {
    // 1. Setup Parameters
    int n = 1 << 10; // 1024
    size_t bytes = n * n * sizeof(float);

    // 2. Memory Allocation
    // Host pointers (only used for verification)
    float *h_a, *h_b, *h_c;
    // Device pointers
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. cuRAND Setup (Generate Data on GPU)
    // --------------------------------------
    // Instead of copying from CPU, we create a generator on the GPU.
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set seed (using system time or fixed number)
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Generate Uniform Data (0.0, 1.0] directly into Device Memory
    // No cudaMemcpy needed!
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    // 4. cuBLAS Setup
    // ---------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 5. Scaling Factors (Alpha & Beta)
    // Formula: C = alpha * (A x B) + beta * C
    // We want C = A x B, so alpha=1, beta=0.
    float alpha = 1.0f;
    float beta = 0.0f;

    // 6. The Heavy Lifter: cublasSgemm
    // --------------------------------
    // Parameters:
    // handle: cuBLAS context
    // op(A), op(B): CUBLAS_OP_N (Normal), CUBLAS_OP_T (Transpose).
    // m, n, k: Dimensions. For A(m x k) * B(k x n) = C(m x n).
    //          Here matrices are square, so m=n=k=n.
    // &alpha: Scaling factor.
    // A, lda: Pointer to A, and "Leading Dimension of A" (stride between columns).
    //         Since it's tightly packed column-major, lda = n.
    // B, ldb: Pointer to B, ldb = n.
    // &beta: Scaling factor for C.
    // C, ldc: Pointer to C, ldc = n.
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, d_a, n, d_b, n, 
                &beta, d_c, n);

    // 7. Retrieve Results
    // Copy data back to Host for verification
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 8. Verify
    printf("Verifying solution...\n");
    verify_solution(h_a, h_b, h_c, n);
    printf("COMPLETED SUCCESSFULLY\n");

    // 9. Cleanup
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}