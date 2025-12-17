// CUDA Graph benchmark kernels
// Uses sleep instruction to simulate workload (consistent with Implicit_Barrier)

#include "../../share/repeat.h"
#include "cuda_runtime.h"

// Sleep instruction: 1000 ns per iteration
#define SLP asm volatile("nanosleep.u32 1000;");

// Sleep kernel with configurable duration
#define SLEEP_KERNEL(DEP) \
__global__ void sleep_kernel_##DEP() \
{ \
    repeat##DEP(SLP;); \
}

#define DEC_SLEEP_KERNEL(DEP) __global__ void sleep_kernel_##DEP();

// Declare sleep kernels:
// Basic kernels: 5us and 10us workloads
DEC_SLEEP_KERNEL(5);   // 5000 ns (basic)
DEC_SLEEP_KERNEL(10);  // 10000 ns (basic)

// Extended kernels: for dual-workload method validation
DEC_SLEEP_KERNEL(20);  // 20000 ns
DEC_SLEEP_KERNEL(40);  // 40000 ns

// Fused kernels: 16× the basic workload (for eliminating workload uncertainty)
DEC_SLEEP_KERNEL(80);  // 80000 ns = 16 × 5000 ns
DEC_SLEEP_KERNEL(160); // 160000 ns = 16 × 10000 ns

typedef void (*KernelFunc)();

// Sleep kernel with counter increment (for verification)
// No atomic needed since kernels execute serially
#define SLEEP_KERNEL_COUNT(DEP) \
__global__ void sleep_kernel_count_##DEP(int* counter) \
{ \
    repeat##DEP(SLP;); \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        (*counter)++; \
    } \
}

#define DEC_SLEEP_KERNEL_COUNT(DEP) __global__ void sleep_kernel_count_##DEP(int* counter);

DEC_SLEEP_KERNEL_COUNT(5);
DEC_SLEEP_KERNEL_COUNT(10);

typedef void (*KernelFuncCount)(int*);
