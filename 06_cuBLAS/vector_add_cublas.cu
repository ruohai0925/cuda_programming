// This program demonstrates Vector Addition using the cuBLAS library.
// Operation: Y = alpha * X + Y (SAXPY)
// Original By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

// nvcc vector_add_cublas.cu -o vector_add_cublas -lcublas

// Standard CUDA runtime headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// [Critical] Include the cuBLAS v2 header.
// This provides access to the BLAS API functions like cublasCreate, 
// cublasSaxpy, etc.
// Note: You must link against 'cublas.lib' (Windows) or 
// '-lcublas' (Linux) during compilation.
#include <cublas_v2.h>

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>

// Helper: Initialize a vector with random float values 0-99
void vector_init(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 100);
    }
}

// Helper: Verify the result on CPU
// We check if c[i] == factor * a[i] + b[i]
void verify_result(float *a, float *b, float *c, float factor, int n) {
    for (int i = 0; i < n; i++) {
        // Floating point comparison often uses epsilon, but for simple integers
        // cast to float, exact equality usually works in these toy examples.
        // In production: use abs(c - expected) < 1e-5
        assert(c[i] == factor * a[i] + b[i]);
    }
}

int main() {
    // 1. Setup Problem Size
    // ---------------------
    // n = 4 (1 << 2). In real apps, this would be large (e.g., 1 << 20).
    int n = 1 << 2;
    size_t bytes = n * sizeof(float);

    // 2. Memory Allocation
    // --------------------
    // Host pointers
    float *h_a, *h_b, *h_c;
    // Device pointers (Standard CUDA malloc)
    float *d_a, *d_b;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Allocate device memory. Even with cuBLAS, we use standard cudaMalloc.
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // Initialize vectors on Host
    vector_init(h_a, n);
    vector_init(h_b, n);

    // 3. Context Setup (The "Handle")
    // -------------------------------
    // Before calling any cuBLAS function, we must create a handle.
    // The handle stores the cuBLAS library context (internal state, streams, etc.).
    // Creating a handle is expensive! Do this once at startup, not inside a loop.
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // 4. Data Transfer: Host -> Device
    // --------------------------------
    // cublasSetVector is a helper function to copy data to the device.
    // Unlike cudaMemcpy, it supports "Strided Access" (skipping elements).
    //
    // Parameters:
    // 1. n          : Number of elements to copy.
    // 2. elemSize   : Size of each element (sizeof(float)).
    // 3. source     : Host pointer (h_a).
    // 4. incx       : Stride for source. 1 = contiguous (read every element).
    //                 If 2, we would read every other element.
    // 5. dest       : Device pointer (d_a).
    // 6. incy       : Stride for destination. 1 = contiguous.
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    // 5. Computation: SAXPY
    // ---------------------
    // We perform the operation: Y = alpha * X + Y
    // In our case: d_b = 2.0 * d_a + d_b
    // This effectively performs vector addition with scaling.
    //
    // Parameters:
    // 1. handle : The context we created earlier.
    // 2. n      : Number of elements.
    // 3. &alpha : POINTER to the scalar factor (on Host or Device).
    //             Here, we pass a pointer to host memory (&scale).
    // 4. x      : Input vector X (d_a).
    // 5. incx   : Stride for X (1 = contiguous).
    // 6. y      : Input/Output vector Y (d_b). The result overwrites d_b.
    // 7. incy   : Stride for Y (1 = contiguous).
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

    // 6. Data Transfer: Device -> Host
    // --------------------------------
    // Copy the result (d_b) back to host (h_c).
    // Parameters follow the same logic as SetVector.
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    // Verify
    verify_result(h_a, h_b, h_c, scale, n);
    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    // 7. Cleanup
    // ----------
    // Destroy the handle to release internal resources.
    cublasDestroy(handle);

    // Free standard memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}