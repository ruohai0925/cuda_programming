# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

No build system (Makefile/CMake) exists. Each directory contains self-contained `.cu` files compiled individually with nvcc:

```bash
# Standard compilation
nvcc source.cu -o executable

# For cuBLAS projects (requires linking)
nvcc file.cu -o executable -lcublas
```

## Repository Structure

CUDA learning repository based on CoffeeBeforeArch's "CUDA Crash Course (v3)" series. Environment: Ubuntu 20.04, CUDA 11, GTX 2060.

Each numbered directory explores one optimization topic with multiple variants showing progressive optimization:

- **01_vector_addition/**: baseline → pinned_memory → unified_memory
- **02_matrix_mul/**: baseline → alignment → noAliasing → tiled
- **03_sum_reduction/**: bank_conflicts → diverged → reduce_idle → no_conflicts
- **04_histogram/**: global_atomic → shmem_atomic
- **05_convolution/**: 1d_naive → 1d_cache → 1d_constant_memory → 1d_tiled → 2d_constant_memory
- **06_cuBLAS/**: SAXPY and batched matrix multiplication using NVIDIA cuBLAS

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

- **Pinned memory** (`01_vector_addition/pinned_memory/`): Eliminates double-copy penalty, enables DMA direct access
- **Unified memory** (`01_vector_addition/unified_memory/`): Single address space with automatic migration; performance degrades when working set exceeds VRAM
- **Tiling** (`02_matrix_mul/tiled/`): Shared memory (`__shared__`) for data reuse within thread blocks
- **Bank conflicts** (`03_sum_reduction/`): Sequential vs strided shared memory access patterns
- **Constant memory** (`05_convolution/`): Read-only mask/coefficient data in `__constant__` memory
