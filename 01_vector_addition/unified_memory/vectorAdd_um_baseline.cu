// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch
//
// UNIFIED MEMORY INTRODUCTION:
// ============================
// CUDA Unified Memory provides a single memory space accessible from both CPU and GPU.
// Key benefits:
// 1. Simplified programming: No need for separate host/device memory allocations
// 2. No manual memory transfers: CUDA runtime automatically migrates data between
//    CPU and GPU as needed (on-demand page migration)
// 3. Single pointer: Same pointer works on both CPU and GPU code
// 4. Automatic synchronization: Memory is kept consistent between CPU and GPU
//
// How it works:
// - cudaMallocManaged() allocates memory that can be accessed by both CPU and GPU
// - The CUDA driver automatically migrates memory pages between CPU and GPU
//   when accessed by either processor (unified memory page faulting)
// - This happens transparently to the programmer
//
// Comparison with baseline version:
// - Baseline: Separate host (CPU) and device (GPU) memory + manual cudaMemcpy()
// - Unified Memory: Single memory space + automatic data migration

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

// CUDA kernel for vector addition
// No change when using CUDA unified memory - the kernel code is identical!
// The unified memory pointers can be used directly in GPU kernels.
// This is one of the key advantages: same kernel code works with unified memory.
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  // Calculate global thread ID
  // Note: Order is (blockDim.x * blockIdx.x) + threadIdx.x
  //       This is equivalent to (blockIdx.x * blockDim.x) + threadIdx.x
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {

  const int N = 1 << 26;
  size_t bytes = N * sizeof(int);

  // Declare unified memory pointers
  // These pointers will be accessible from both CPU and GPU code
  int *a, *b, *c;

  // Allocate unified memory using cudaMallocManaged()
  // cudaMallocManaged(): Allocates memory that can be accessed by both CPU and GPU.
  //                      This is the key difference from cudaMalloc() which only
  //                      allocates device (GPU) memory.
  //                      The memory is initially allocated but not yet migrated to
  //                      either CPU or GPU - migration happens on first access.
  //                      Returns cudaError_t (should check for errors in production code).
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  
  // Initialize vectors on the CPU
  // Note: We can directly access unified memory from CPU code!
  //       When we write to a[i] and b[i], the CUDA runtime automatically
  //       ensures the data is accessible to the CPU. No cudaMemcpy needed!
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  
  // Threads per CTA (1024 threads per CTA)
  int BLOCK_SIZE = 1 << 10;

  // CTAs per Grid
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Call CUDA kernel
  // Note: We pass the unified memory pointers directly to the kernel.
  //       When the GPU accesses these pointers, the CUDA runtime automatically
  //       migrates the memory pages to the GPU if needed.
  //       No cudaMemcpy() needed! This is a major simplification.
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  for (int i = 0; i < 100; i++) {
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
  }

  // CRITICAL: cudaDeviceSynchronize() - Wait for GPU kernel to complete
  // ====================================================================
  // Unlike the baseline version which uses cudaMemcpy() (which implicitly
  // synchronizes), unified memory requires explicit synchronization.
  //
  // Why is this needed?
  // - Kernel launches are asynchronous (CPU continues immediately)
  // - Without synchronization, CPU might read results before GPU finishes
  // - cudaMemcpy() in baseline version acts as implicit synchronization barrier
  // - With unified memory, we don't have cudaMemcpy, so we need explicit sync
  //
  // cudaDeviceSynchronize(): Blocks CPU execution until all previously issued
  //                          CUDA operations on the device are complete.
  //                          This ensures the kernel has finished and results
  //                          are available before we verify them on CPU.
  cudaDeviceSynchronize();

  // Verify the result on the CPU
  // Note: We can directly read unified memory from CPU after synchronization!
  //       The CUDA runtime automatically migrates the memory pages back to CPU
  //       when we access c[i] here. Again, no cudaMemcpy() needed.
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
  
  // Free unified memory
  // cudaFree(): Works the same way as freeing regular device memory.
  //             Frees the unified memory allocated with cudaMallocManaged().
  //             After this call, the pointers a, b, c are invalid.
  //             Always free allocated memory to prevent memory leaks!
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;
}