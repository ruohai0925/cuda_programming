// This program computes the sum of two vectors of length N using pinned memory
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::generate;
using std::vector;

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    // Each thread adds a single element
    c[tid] = a[tid] + b[tid];
  }
}

// Check vector add result
void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  // Array size of 2^26 (approx. 67 million elements)
  // CRITICAL: We increased the size significantly compared to previous examples.
  // Pinned memory benefits are most visible with large data transfers.
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  // Note: We are using raw pointers instead of std::vector because std::vector
  // allocates 'pageable' memory by default.
  int *h_a, *h_b, *h_c;

  // Allocate pinned memory (Page-Locked Memory)
  // cudaMallocHost: Allocates memory on the host that is accessible to the device.
  // Key Feature 1: The OS cannot swap this memory out to disk (it is "pinned" in physical RAM).
  // Key Feature 2: Enables higher bandwidth for cudaMemcpy via DMA (Direct Memory Access).
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);

  // Initialize random numbers in each array
  // We access pinned memory just like standard C-style arrays on the CPU.
  for(int i = 0; i < N; i++){
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }
  
  // Allocate memory on the device (Standard VRAM allocation)
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  // Since h_a and h_b are pinned, the GPU DMA engine can read them directly.
  // This avoids the overhead of copying data to a temporary staging buffer first.
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024 threads per CTA)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  std::cout << "NUM_BLOCKS: " << NUM_BLOCKS << std::endl;
  std::cout << "NUM_THREADS: " << NUM_THREADS << std::endl;

  // Launch the kernel on the GPU
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  for (int i = 0; i < 100; i++) {
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  }

  // Copy sum vector from device to host
  // Again, copying back to pinned memory (h_c) is faster than pageable memory.
  // This is a synchronous call (blocking), acting as a barrier.
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(h_a, h_b, h_c, N);

  // Free pinned memory
  // CRITICAL: Must use cudaFreeHost, not free() or delete.
  // This releases the page lock and returns memory to the OS.
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}