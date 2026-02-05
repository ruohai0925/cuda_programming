// This program computes the sum of two N-element vectors using unified memory
// Optimized Logic for Data Migration and Hints
// By: Nick from CoffeeBeforeArch & Gem_GPU
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

// CUDA kernel for vector addition
__global__ void vectorAdd(const int *a, const int *b, int *c, int N) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  // 1. Setup Parameters
  const int N = 1 << 16;
  size_t bytes = N * sizeof(int);
  int device_id = 0;
  
  // Get the actual GPU device ID
  // It's good practice not to assume ID is 0, though it usually is.
  cudaGetDevice(&device_id);

  // 2. Allocate Unified Memory
  // Pointers a, b, c are valid on both Host and Device.
  int *a, *b, *c;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  // =============================================================
  // PHASE 1: Initialization (CPU Side)
  // =============================================================
  // Concept: "First Touch". When CPU writes to these addresses, 
  // the driver creates the physical pages in CPU RAM (System Memory).
  // No prefetch needed here because we are already on the CPU.
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    // c[i] doesn't need init, it will be overwritten.
  }

  // =============================================================
  // PHASE 2: Optimization for GPU Execution (The "Handoff")
  // =============================================================
  
  // LOGIC 1: Data Migration (Prefetch) -> "Move data WHERE it is needed"
  // We are about to launch a kernel on the GPU.
  // 'a' and 'b' are INPUTS: GPU will read them heavily.
  // 'c' is OUTPUT: GPU will write to it.
  // Therefore, we move ALL of them to the GPU memory (VRAM) to maximize bandwidth.
  // If we don't do this, the GPU will trigger "Page Faults" one by one, which is slow.
  cudaMemPrefetchAsync(a, bytes, device_id);
  cudaMemPrefetchAsync(b, bytes, device_id);
  cudaMemPrefetchAsync(c, bytes, device_id);

  // LOGIC 2: Memory Advice (Hints) -> "Tell driver HOW data is used"
  // 'a' and 'b' are Read-Only for the Kernel.
  // 'SetReadMostly' tells the driver: "This data won't be modified soon".
  // Benefit: On some architectures (like Pascal+), this allows the driver to 
  // duplicate the data (keep a copy on CPU and create a copy on GPU), 
  // preventing thrashing if both try to read it later.
  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, device_id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, device_id);

  // Note: We DO NOT set 'PreferredLocation' to CPU here, because we WANT 
  // the data to migrate to the GPU for high-speed access.

  // 3. Launch Kernel
  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  // Launch occurs on the default stream.
  // The prefetches above were also on the default stream, so they are guaranteed
  // to finish BEFORE the kernel starts.
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  // 4. Synchronization
  // CPU must wait for GPU to finish.
  cudaDeviceSynchronize();

  // =============================================================
  // PHASE 3: Optimization for Verification (CPU Side)
  // =============================================================
  
  // Logic: Now we need the data back on the CPU for the 'assert' loop.
  // 1. We need 'c' (the result).
  // 2. We need 'a' and 'b' (to check the math).
  
  // Prefetch everything back to CPU device (cudaCpuDeviceId).
  // If we skip this, the CPU will trigger page faults when reading c[i], a[i], b[i].
  // Bulk prefetching is faster than individual page faults.
  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  // 5. Verification
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }

  // 6. Cleanup
  // Important: Remove advice (optional but clean) and Free
  cudaMemAdvise(a, bytes, cudaMemAdviseUnsetReadMostly, device_id);
  cudaMemAdvise(b, bytes, cudaMemAdviseUnsetReadMostly, device_id);
  
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;
}