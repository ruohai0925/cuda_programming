# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

No build system (Makefile/CMake) exists. Each directory contains self-contained `.cu` files compiled individually with nvcc:

```bash
# Standard compilation
nvcc source.cu -o executable

# For cuBLAS projects (requires linking)
nvcc vector_add_cublas.cu -o vector_add -lcublas
nvcc matrix_mul_cublas.cu -o matrix_mul -lcublas -lcurand
```

## Repository Structure

CUDA learning repository based on CoffeeBeforeArch's "CUDA Crash Course (v3)" series. Environment: Ubuntu 20.04, CUDA 11, GTX 2060.

Each numbered directory explores one optimization topic with multiple variants showing progressive optimization:

- **01_vector_addition/**: baseline → grid_stride_and_vectorized_memory_access → pinned_memory → unified_memory
- **02_matrix_mul/**: baseline → alignment → noAliasing(restrict, tmp) → rectangular → nonMultiple → tiled(1D, 2D)
- **03_sum_reduction/**: diverged → bank_conflicts → reduce_idle → no_conflicts → device_function → cooperative_groups
- **04_histogram/**: global_atomic → shmem_atomic
- **05_convolution/**: 1d_naive → 1d_cache → 1d_constant_memory → 1d_tiled → 2d_constant_memory
- **06_cuBLAS/**: SAXPY (`vector_add_cublas.cu`) and batched matrix multiplication (`matrix_mul_cublas.cu`) using cuBLAS v2 API

### Additional Directories

- **cuda_timing_Nick/**: Performance benchmarking framework with timing harness for matrix multiplication (global/shmem variants: naive, coalesced_access, prefetch, tmp_var, unroll) and sum reduction
- **cuda_timing_ZDSJTU/**: Subset of main examples (02_matrix_mul, 03_sum_reduction) with timing instrumentation for performance comparison study
- **learning_notes**: Profiling notes with `nsys profile` commands and example output

## Code Patterns

**Naming conventions**: `d_` prefix for device pointers, `h_` prefix for host pointers.

**Thread indexing**:
```cpp
// 1D kernel
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D kernel
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Grid configuration** (ceiling division for non-multiples):
```cpp
int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
```

**Memory pattern**: Host uses `std::vector`, device uses `cudaMalloc`/`cudaFree`, unified memory uses `cudaMallocManaged`.

**Verification**: Every kernel has a corresponding CPU `verify_result()` function using assertions.

## Key Concepts by Directory

- **Grid-stride loops** (`01_vector_addition/grid_stride_and_vectorized_memory_access/`): Decouples grid size from data size; uses `int4` for 128-bit vectorized loads
- **Pinned memory** (`01_vector_addition/pinned_memory/`): Eliminates double-copy penalty, enables DMA direct access
- **Unified memory** (`01_vector_addition/unified_memory/`): Single address space with automatic migration; performance degrades when working set exceeds VRAM
- **Tiling** (`02_matrix_mul/tiled/`): Shared memory (`__shared__`) for data reuse within thread blocks; both 1D-array and 2D-array variants
- **Rectangular/nonMultiple** (`02_matrix_mul/`): Handling non-square matrices and dimensions that aren't multiples of block size
- **Bank conflicts** (`03_sum_reduction/`): Sequential vs strided shared memory access patterns
- **Warp-level optimization** (`03_sum_reduction/device_function/`): Unrolled last warp with `warpReduce()` device function
- **Cooperative groups** (`03_sum_reduction/cooperative_groups/`): Modern sync API replacing `__syncthreads()`, with `int4` vectorized loads and `atomicAdd`
- **Constant memory** (`05_convolution/`): Read-only mask/coefficient data in `__constant__` memory
- **cuBLAS** (`06_cuBLAS/`): Column-major layout, `cublasSetVector`, cuRAND for data generation
