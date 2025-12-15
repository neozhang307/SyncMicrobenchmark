// CUDA Graph Overhead Benchmark
// Measures graph construction and launch overhead with varying kernel counts
// Uses fused kernel methodology to eliminate workload uncertainty (same as Implicit_Barrier)

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>

// External test functions
void Test_StreamCapture_All(unsigned int blocks, unsigned int threads);
void Test_WhileConditional(unsigned int blocks, unsigned int threads);
void Test_ChildGraph(unsigned int blocks, unsigned int threads);

int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    unsigned int smx_count = deviceProp.multiProcessorCount;

    printf("CUDA Graph Overhead Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("=======================================================================\n\n");

    Test_StreamCapture_All(smx_count, 1024);

    printf("\n\n");

    Test_WhileConditional(smx_count, 1024);

    printf("\n\n");

    Test_ChildGraph(smx_count, 1024);

    return 0;
}
