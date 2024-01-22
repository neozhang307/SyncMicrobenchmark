/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/

#include <stdio.h>
#include <cooperative_groups.h>
#include "repeat.h"
namespace cg = cooperative_groups;

template <class T>
__device__ __forceinline__ T
warp_reduce(T mySum, cg::thread_block_tile<32> group) {
  mySum += group.shfl_down(mySum, 16);
  mySum += group.shfl_down(mySum, 8);
  mySum += group.shfl_down(mySum, 4);
  mySum += group.shfl_down(mySum, 2);
  mySum += group.shfl_down(mySum, 1);
  return mySum;
}

template <class T, unsigned int blockSize>
__device__ __forceinline__ T
block_reduce_cuda_sample_opt(T mySum, T *sdata, cg::thread_block cta) {
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  unsigned int tid = cta.thread_rank();
  if (blockSize >= 64) {
    sdata[tid] = mySum;
    cg::sync(cta);
  }
  // do reduction in shared mem
  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] = mySum = mySum + sdata[tid + 512];
    }
    cg::sync(cta);
  }

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
    }
    cg::sync(cta);
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
    }
    cg::sync(cta);
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = mySum = mySum + sdata[tid + 64];
    }
    cg::sync(cta);
  }

  if (blockSize >= 64) {
    if (tid < 32) {
      // Fetch final intermediate sum from 2nd warp
      mySum += sdata[tid + 32];
      // Reduce final warp using shuffle
      mySum = warp_reduce(mySum, tile32);
    }
  }
  if (blockSize == 32) {
    mySum = warp_reduce(mySum, tile32);
  }
  return mySum;
}

template <class T, unsigned int blockSize>
__device__ __forceinline__ T
block_reduce_warpserial(T mySum, T *sdata, cg::thread_block cta) {
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  unsigned int tid = cta.thread_rank();

  if (blockSize >= 64) {
    sdata[tid] = mySum;
    cg::sync(cta);
    if (tid < 32) {
      if (blockSize >= 1024) {
        mySum += sdata[tid + 512];
        mySum += sdata[tid + 544];
        mySum += sdata[tid + 576];
        mySum += sdata[tid + 608];
        mySum += sdata[tid + 640];
        mySum += sdata[tid + 672];
        mySum += sdata[tid + 704];
        mySum += sdata[tid + 736];
        mySum += sdata[tid + 768];
        mySum += sdata[tid + 800];
        mySum += sdata[tid + 832];
        mySum += sdata[tid + 864];
        mySum += sdata[tid + 896];
        mySum += sdata[tid + 928];
        mySum += sdata[tid + 960];
        mySum += sdata[tid + 992];
      }
      if (blockSize >= 512) {
        mySum += sdata[tid + 256];
        mySum += sdata[tid + 288];
        mySum += sdata[tid + 320];
        mySum += sdata[tid + 352];
        mySum += sdata[tid + 384];
        mySum += sdata[tid + 416];
        mySum += sdata[tid + 448];
        mySum += sdata[tid + 480];
      }
      if (blockSize >= 256) {
        mySum += sdata[tid + 128];
        mySum += sdata[tid + 160];
        mySum += sdata[tid + 192];
        mySum += sdata[tid + 224];
      }
      if (blockSize >= 128) {
        mySum += sdata[tid + 64];
        mySum += sdata[tid + 96];
      }
      if (blockSize >= 64)
        mySum += sdata[tid + 32];
      mySum = warp_reduce(mySum, tile32);
    }
  }
  if (blockSize == 32) {
    mySum = warp_reduce(mySum, tile32);
  }

  return mySum;
}

template <class T, bool nIsPow2>
__device__ __forceinline__ T
serial_reduce(unsigned int n, unsigned int threadId, unsigned int blockId,
              unsigned int blockSize, unsigned int gridSizeMul2, T *idata)
    /*
    n is the size of array
    blockSize is the size of a block
    */
{
  unsigned int i = blockId * blockSize * 2 + threadId;
  T mySum = 0;
  while (i < n) {
    mySum += idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += idata[i + blockSize];

    i += gridSizeMul2;
  }
  return mySum;
}

template <class T, unsigned int BlockSize>
__device__ __forceinline__ T
serial_reduce_final(unsigned int n, unsigned int threadId, T *idata)
    /*
    n is the size of array
    blockSize is the size of a block
    */
{
  unsigned int i = threadId;
  T mySum = 0;
  while (i < n) {
    if (BlockSize <= 128) {
      repeat16(mySum += idata[i]; i += BlockSize;);
    }
    if (BlockSize == 256) {
      repeat8(mySum += idata[i]; i += BlockSize;);
    }
    if (BlockSize == 512) {
      repeat4(mySum += idata[i]; i += BlockSize;);
    }
    if (BlockSize == 1024) {
      repeat2(mySum += idata[i]; i += BlockSize;);
    }
  }
  return mySum;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};
#include <iostream>

template <typename T, unsigned int blockSize, bool nIsPow2, bool useSM,
          bool useWarpSerial>
__global__ void reduce_grid_sb(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group gg = cg::this_grid();
  T (*block_reduction)(T, T *, cg::thread_block cta);
  if (useWarpSerial == true) {
    block_reduction = block_reduce_warpserial<T, blockSize>;
  } else {
    block_reduction = block_reduce_cuda_sample_opt<T, blockSize>;
  }
  T *sdata;
  if (useSM == true)
    sdata = SharedMemory<T>();
  else
    sdata = g_odata;
  T mySum = 0;
  // if constexpr (singleBlock)
  {
    // use fewer threads is more profitable
    if (blockIdx.x == 0) {
      mySum = 0;
      mySum = serial_reduce<T, nIsPow2>(n, threadIdx.x, 0, blockDim.x,
                                        blockDim.x * 1 * 2, g_idata);

      mySum = block_reduction(mySum, sdata, cta);
      if (threadIdx.x == 0)
        g_odata[blockIdx.x] = mySum;
    }
  }
}
template <typename T, unsigned int blockSize, bool nIsPow2, bool useSM,
          bool useWarpSerial>
__global__ void reduce_grid(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group gg = cg::this_grid();
  T (*block_reduction)(T, T *, cg::thread_block cta);
  if (useWarpSerial == true) {
    block_reduction = block_reduce_warpserial<T, blockSize>;
  } else {
    block_reduction = block_reduce_cuda_sample_opt<T, blockSize>;
  }
  T *sdata;
  if (useSM == true)
    sdata = SharedMemory<T>();
  else
    sdata = g_odata;
  T mySum = 0;

  {
    mySum = serial_reduce<T, nIsPow2>(n, threadIdx.x, blockIdx.x, blockDim.x,
                                      blockDim.x * gridDim.x * 2, g_idata);
    g_odata[gg.thread_rank()] = mySum;
    cg::sync(gg);

    // write result for this block to global mem
    if (blockIdx.x == 0) {
      mySum = 0;
      mySum = serial_reduce_final<T, blockSize>(blockSize * gridDim.x,
                                                threadIdx.x, g_odata);

      mySum = block_reduction(mySum, sdata, cta);
      if (threadIdx.x == 0)
        g_odata[blockIdx.x] = mySum;
    }
  }
}

template <class T, bool nIsPow2>
__global__ void reduce_kernel1(T *g_idata, T *g_odata, unsigned int n)
    /*
    size of g_odata no smaller than n; equal to the multiply of blockSize;
    value index larger than n should be setted to 0 in advance;
    */
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // double mySum = g_idata[i];
  T mySum = 0;

  mySum = serial_reduce<T, nIsPow2>(n, threadIdx.x, blockIdx.x, blockDim.x,
                                    blockDim.x * gridDim.x * 2, g_idata);
  g_odata[blockIdx.x * blockDim.x + threadIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool useSM, bool useWarpSerial>
__global__ void reduce_kernel2(T *g_idata, T *g_odata, unsigned int n)
    /*
    size of g_odata no smaller than n; equal to the multiply of blockSize;
    value index larger than n should be setted to 0 in advance;
    */
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  T *sdata;
  if (useSM == true)
    sdata = SharedMemory<T>();
  else
    sdata = g_odata;

  T (*block_reduction)(T, T *, cg::thread_block cta);
  if (useWarpSerial == true) {
    block_reduction = block_reduce_warpserial<T, blockSize>;
  } else {
    block_reduction = block_reduce_cuda_sample_opt<T, blockSize>;
  }

  unsigned int tid = threadIdx.x;

  double mySum = 0;

  if (blockIdx.x == 0) {
    mySum = 0;
    mySum = serial_reduce_final<T, blockSize>(n, tid, g_idata);

    mySum = block_reduction(mySum, sdata, cta);
    if (tid == 0)
      g_odata[blockIdx.x] = mySum;
  }
}
template <class T> T cpu_reduce(T *array, unsigned int array_size) {
  T sum = 0;
  for (int i = 0; i < array_size; i++) {
    sum += array[i];
  }
  return sum;
}

template <class T, unsigned int blockSize, bool nIsPow2, bool useSM,
          bool useWarpSerial, bool singleBlock>
void __forceinline__
launchKernelBasedReduction(T *g_idata, T *g_odata, unsigned int gridSize,
                           unsigned int n) {
  if (singleBlock == false) {
    reduce_kernel1<T, nIsPow2> << <gridSize, blockSize>>> (g_idata, g_odata, n);
    if (useSM == true)
      reduce_kernel2<T, blockSize, true, useWarpSerial> << <1, blockSize,
                                                            blockSize *
                                                                sizeof(T)>>>
          (g_odata, g_odata, blockSize * gridSize);
    else
      reduce_kernel2<T, blockSize, false, useWarpSerial> << <1, blockSize>>>
          (g_odata, g_odata, blockSize * gridSize);
  } else {
    if (useSM == true)
      reduce_kernel2<T, blockSize, true, useWarpSerial> << <1, blockSize,
                                                            blockSize *
                                                                sizeof(T)>>>
          (g_idata, g_odata, n);
    else
      reduce_kernel2<T, blockSize, false, useWarpSerial> << <1, blockSize>>>
          (g_idata, g_odata, n);
  }
}
template <class T, unsigned int blockSize, bool nIsPow2, bool useSM,
          bool useWarpSerial, bool singleBlock>
void __forceinline__ gridBasedReduction(T *g_idata, T *g_odata,
                                        unsigned int gridSize, unsigned int n) {
  void *KernelArgs[] = {(void **)&g_idata, (void **)&g_odata, (void *)&n };
  if (singleBlock == false) {
    if (useSM == true) {
      cudaLaunchCooperativeKernel(
          (void *)reduce_grid<T, blockSize, nIsPow2, true, useWarpSerial>,
          gridSize, blockSize, KernelArgs, blockSize * sizeof(T), 0);
    } else {
      cudaLaunchCooperativeKernel(
          (void *)reduce_grid<T, blockSize, nIsPow2, false, useWarpSerial>,
          gridSize, blockSize, KernelArgs, 0, 0);
    }
  } else {
    if (useSM == true) {
      cudaLaunchCooperativeKernel(
          (void *)reduce_grid_sb<T, blockSize, nIsPow2, true, useWarpSerial>, 1,
          blockSize, KernelArgs, blockSize * sizeof(T), 0);
    } else {
      cudaLaunchCooperativeKernel(
          (void *)reduce_grid_sb<T, blockSize, nIsPow2, false, useWarpSerial>,
          1, blockSize, KernelArgs, 0, 0);
    }
  }
}

template <class T, unsigned int blockSize, bool nIsPow2, bool useSM,
          bool useWarpSerial, bool useKernelLaunch, bool singleBlock>
void single_test(float &millisecond, T &gpu_result, unsigned int grid_size,
                 unsigned int array_size, T *h_input) {
  T *h_output = (T *)malloc(sizeof(T) * array_size);
  T *d_input;
  T *d_output;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaMalloc((void **)&d_input, sizeof(T) * array_size);
  cudaMalloc((void **)&d_output, sizeof(T) * array_size);
  cudaMemcpy(d_input, h_input, sizeof(T) * array_size, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  if (useKernelLaunch == true) {
    launchKernelBasedReduction<T, blockSize, true, useSM, useWarpSerial,
                               singleBlock>(d_input, d_output, grid_size,
                                            array_size);
    launchKernelBasedReduction<T, blockSize, true, useSM, useWarpSerial,
                               singleBlock>(d_input, d_output, grid_size,
                                            array_size);
    launchKernelBasedReduction<T, blockSize, true, useSM, useWarpSerial,
                               singleBlock>(d_input, d_output, grid_size,
                                            array_size);
    launchKernelBasedReduction<T, blockSize, true, useSM, useWarpSerial,
                               singleBlock>(d_input, d_output, grid_size,
                                            array_size);
    launchKernelBasedReduction<T, blockSize, true, useSM, useWarpSerial,
                               singleBlock>(d_input, d_output, grid_size,
                                            array_size);
  } else {
    gridBasedReduction<T, blockSize, true, useSM, useWarpSerial, singleBlock>(
        d_input, d_output, grid_size, array_size);
    gridBasedReduction<T, blockSize, true, useSM, useWarpSerial, singleBlock>(
        d_input, d_output, grid_size, array_size);
    gridBasedReduction<T, blockSize, true, useSM, useWarpSerial, singleBlock>(
        d_input, d_output, grid_size, array_size);
    gridBasedReduction<T, blockSize, true, useSM, useWarpSerial, singleBlock>(
        d_input, d_output, grid_size, array_size);
    gridBasedReduction<T, blockSize, true, useSM, useWarpSerial, singleBlock>(
        d_input, d_output, grid_size, array_size);
  }
  cudaEventRecord(end);
  cudaMemcpy(h_output, d_output, sizeof(T) * array_size,
             cudaMemcpyDeviceToHost);
  gpu_result = h_output[0];
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,
           cudaGetErrorString(e));
  }
  cudaEventElapsedTime(&millisecond, start, end);
  millisecond /= 5;
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

#define my_single_test(type, threadcount, isPow2, useSM, useWarpSerial,        \
                       useKernelLaunch, singleBlock)                           \
  do {                                                                         \
    double *lats = (double *)malloc(sizeof(double) * repeat);                  \
    for (int i = 0; i < repeat; i++) {                                         \
      single_test<type, threadcount, isPow2, useSM, useWarpSerial,             \
                  useKernelLaunch, singleBlock>(                               \
          millisecond, gpu_result, smx_count *block_per_sm, size, h_input);    \
      lats[i] = millisecond;                                                   \
    }                                                                          \
    millisecond = 0;                                                           \
    for (int i = skip; i < repeat; i++) {                                      \
      millisecond += lats[i];                                                  \
    }                                                                          \
    millisecond = millisecond / (repeat - skip);                               \
    free(lats);                                                                \
  } while (0)
#define switchBlock(type, threadcount, isPow2, useSM, useWarpSerial,           \
                    useKernelLaunch, singleBlock)                              \
  if (singleBlock == true) {                                                   \
    my_single_test(type, threadcount, isPow2, useSM, useWarpSerial,            \
                   useKernelLaunch, true);                                     \
  }                                                                            \
  if (singleBlock == false) {                                                  \
    my_single_test(type, threadcount, isPow2, useSM, useWarpSerial,            \
                   useKernelLaunch, false);                                    \
  }

#define switchuseSM(type, threadcount, isPow2, useSM, useWarpSerial,           \
                    useKernelLaunch, singleBlock)                              \
  if (useSM == true) {                                                         \
    switchBlock(type, threadcount, isPow2, true, useWarpSerial,                \
                useKernelLaunch, singleBlock);                                 \
  }                                                                            \
  if (useSM == false) {                                                        \
    switchBlock(type, threadcount, isPow2, false, useWarpSerial,               \
                useKernelLaunch, singleBlock);                                 \
  }

#define switchuseWarpSerial(type, threadcount, isPow2, useSM, useWarpSerial,   \
                            useKernelLaunch, singleBlock)                      \
  if (useWarpSerial == true) {                                                 \
    switchuseSM(type, threadcount, isPow2, useSM, true, useKernelLaunch,       \
                singleBlock);                                                  \
  }                                                                            \
  if (useWarpSerial == false) {                                                \
    switchuseSM(type, threadcount, isPow2, useSM, false, useKernelLaunch,      \
                singleBlock);                                                  \
  }

#define switchuseKernelLaunch(type, threadcount, isPow2, useSM, useWarpSerial, \
                              useKernelLaunch, singleBlock)                    \
  if (useKernelLaunch == true) {                                               \
    switchuseWarpSerial(type, threadcount, isPow2, useSM, useWarpSerial, true, \
                        singleBlock);                                          \
  }                                                                            \
  if (useKernelLaunch == false) {                                              \
    switchuseWarpSerial(type, threadcount, isPow2, useSM, useWarpSerial,       \
                        false, singleBlock);                                   \
  }

#define switchisPow2(type, threadcount, isPow2, useSM, useWarpSerial,          \
                     useKernelLaunch, singleBlock)                             \
  if (isPow2 == true) {                                                        \
    switchuseKernelLaunch(type, threadcount, true, useSM, useWarpSerial,       \
                          useKernelLaunch, singleBlock);                       \
  }                                                                            \
  if (isPow2 == false) {                                                       \
    switchuseKernelLaunch(type, threadcount, false, useSM, useWarpSerial,      \
                          useKernelLaunch, singleBlock);                       \
  }

#define switchall(type, threadcount, isPow2, useSM, useWarpSerial,             \
                  useKernelLaunch, singleBlock)                                \
  switch (threadcount) {                                                       \
  case 32:                                                                     \
    switchisPow2(type, 32, isPow2, useSM, useWarpSerial, useKernelLaunch,      \
                 singleBlock);                                                 \
    break;                                                                     \
  case 64:                                                                     \
    switchisPow2(type, 64, isPow2, useSM, useWarpSerial, useKernelLaunch,      \
                 singleBlock);                                                 \
    break;                                                                     \
  case 128:                                                                    \
    switchisPow2(type, 128, isPow2, useSM, useWarpSerial, useKernelLaunch,     \
                 singleBlock);                                                 \
    break;                                                                     \
  case 256:                                                                    \
    switchisPow2(type, 256, isPow2, useSM, useWarpSerial, useKernelLaunch,     \
                 singleBlock);                                                 \
    break;                                                                     \
  case 512:                                                                    \
    switchisPow2(type, 512, isPow2, useSM, useWarpSerial, useKernelLaunch,     \
                 singleBlock);                                                 \
    break;                                                                     \
  case 1024:                                                                   \
    switchisPow2(type, 1024, isPow2, useSM, useWarpSerial, useKernelLaunch,    \
                 singleBlock);                                                 \
    break;                                                                     \
  }

template <class T>
void runTest(unsigned int thread_per_block, unsigned int block_per_sm,
             unsigned int smx_count, unsigned int size, unsigned int repeat,
             unsigned int skip, bool useSM, bool useWarpSerial,
             bool useKernelLaunch, bool singleBlock) {
  float millisecond;
  bool isPow2 = false;
  T gpu_result;
  T *h_input = (T *)malloc(sizeof(T) * size);
  for (int i = 0; i < size; i++) {
    h_input[i] = 1;
  }
  double cpu_result = cpu_reduce<T>(h_input, size);
  if (size % (thread_per_block * smx_count * 2) == 0) {
    isPow2 = true;
  } else {
    isPow2 = false;
  }

  switchall(T, thread_per_block, isPow2, useSM, useWarpSerial, useKernelLaunch,
            singleBlock);

  fprintf(stderr, "%f-%f=%f\n", cpu_result, (double)gpu_result,
          cpu_result - gpu_result);
  printf("useSM: %d, use warp serial:%d, use kernel launch:%d, block/SM %d "
         "thread %d totalsize %d time: %f ms speed: %f GB/s\n",
         useSM, useWarpSerial, useKernelLaunch, block_per_sm, thread_per_block,
         size, (double)millisecond,
         (double)size * sizeof(T) / 1000 / 1000 / 1000 / (millisecond / 1000));

  free(h_input);
}

void PrintHelp() {
  printf("--thread <n>(t):           thread per block\n \
             --block <n>(b):            block per sm\n \
             --base_array <n>(a):       average array per thread\n \
             --array <n>(n):            total array size\n \
             --repeat <n>(r):           time of experiment (larger than 2)\n \
             --type <n>(v):             type of experiment (0:int 1:float 2:double)\n \
             --sharememory(s):          use shared memory at block level reduction (default false)\n \
             --warpserial(w):           use warpserial implementation (default false)\n \
             --singleblk(g):            use single block instead of a grid\n \
             --kernellaunch(k):         use kernel launch as an implicit barrier (default false)\n");
  exit(1);
}

#include <getopt.h>
#include <iostream>
int main(int argc, char **argv) {
  cudaDeviceProp deviceProp;
  cudaSetDevice(0);
  cudaGetDeviceProperties(&deviceProp, 0);
  unsigned int smx_count = deviceProp.multiProcessorCount;

  unsigned int thread_per_block = 1024;
  unsigned int block_per_sm = 2;
  unsigned int data_per_thread = 4;
  unsigned int type = 0;

  bool useSM = false;
  bool useWarpSerial = false;
  bool useKernelLaunch = false;
  bool singleBlock = false;
  unsigned int size = 0;

  unsigned int repeat = 12;
  unsigned int skip = 2;

  const char *const short_opts = "t:b:a:n:r:v:swkg";
  const option long_opts[] = {
    { "thread", required_argument, nullptr, 't' },
    { "block", required_argument, nullptr, 'b' },
    { "base_array", required_argument, nullptr, 'a' },
    { "array", required_argument, nullptr, 'n' },
    { "repeat", required_argument, nullptr, 'r' },
    { "type", required_argument, nullptr, 'v' },
    { "sharememory", no_argument, nullptr, 's' },
    { "warpserial", no_argument, nullptr, 'w' },
    { "kernellaunch", no_argument, nullptr, 'k' },
    { "singleblock", no_argument, nullptr, 'g' },
    { nullptr, no_argument, nullptr, 0 }
  };

  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

    if (-1 == opt)
      break;

    switch (opt) {
    case 't':
      thread_per_block = std::stoi(optarg);
      fprintf(stderr, "thread set to: %d\n", thread_per_block);
      break;

    case 'b':
      block_per_sm = std::stoi(optarg);
      fprintf(stderr, "block set to: %d\n", block_per_sm);
      break;

    case 'a':
      data_per_thread = std::stoi(optarg);
      fprintf(stderr, "data per thread set to: %d\n", data_per_thread);
      break;

    case 'n':
      size = std::stoi(optarg);
      fprintf(stderr, "array size set to: %d\n", size);
      break;

    case 'r':
      repeat = std::stoi(optarg);
      if (repeat <= 2) {
        repeat = 1;
        skip = 0;
        fprintf(stderr, "repeat set to: %d and skip 0 experiment\n", repeat);
      } else {
        fprintf(stderr, "repeat set to: %d\n", repeat);
      }
      break;
    case 'v':
      type = std::stoi(optarg);
      type = type >= 3 ? 0 : type;
      fprintf(stderr, "type set to (0:int 1:float 2:double): %d\n", type);
      break;
    case 's':
      useSM = true;
      fprintf(stderr, "useSM is set to true\n");
      break;

    case 'w':
      useWarpSerial = true;
      fprintf(stderr, "useWarpSerial is set to true\n");
      break;

    case 'k':
      useKernelLaunch = true;
      fprintf(stderr, "useKernelLaunch is set to true\n");
      break;
    case 'g':
      singleBlock = true;
      fprintf(stderr, "useSingleBlock is set to true\n");
      break;
    default:
      PrintHelp();
      break;
    }
  }

  size = size == 0
             ? block_per_sm * thread_per_block * smx_count * data_per_thread
             : size;
  switch (type) {
  case 0:
    runTest<int>(thread_per_block, block_per_sm, smx_count, size, repeat, skip,
                 useSM, useWarpSerial, useKernelLaunch, singleBlock);
    break;
  case 1:
    runTest<float>(thread_per_block, block_per_sm, smx_count, size, repeat,
                   skip, useSM, useWarpSerial, useKernelLaunch, singleBlock);
    break;
  case 2:
    runTest<double>(thread_per_block, block_per_sm, smx_count, size, repeat,
                    skip, useSM, useWarpSerial, useKernelLaunch, singleBlock);
    break;
  }
}
