// CUDA Graph benchmark kernel definitions
// All kernels defined here to avoid multiple definition errors

#include "CudaGraph_Kernel.cuh"

// Define sleep kernels
SLEEP_KERNEL(5);
SLEEP_KERNEL(10);
SLEEP_KERNEL(20);
SLEEP_KERNEL(40);
SLEEP_KERNEL(80);
SLEEP_KERNEL(160);

// Define sleep kernels with counter
SLEEP_KERNEL_COUNT(5);
SLEEP_KERNEL_COUNT(10);
