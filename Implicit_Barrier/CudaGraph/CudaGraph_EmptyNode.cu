// Empty Node benchmark: test if adding empty nodes increases overhead
// Compare flat graph with kernels only vs kernels + empty nodes

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

//=============================================================================
// Flat graph: N kernels in sequence
//=============================================================================
double build_flat_graph(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                        int iterations, int sleep_count,
                        unsigned int blocks, unsigned int threads)
{
    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    cudaGraphCreate(graph, 0);

    cudaGraphNode_t prevNode = NULL;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;

    for (int i = 0; i < iterations; i++) {
        cudaGraphNode_t kernelNode;
        if (prevNode) {
            cudaGraphAddKernelNode(&kernelNode, *graph, &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&kernelNode, *graph, NULL, 0, &kernelParams);
        }
        prevNode = kernelNode;
    }

    cudaGraphInstantiate(graphExec, *graph, 0);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

//=============================================================================
// Flat graph with empty nodes: kernel -> empty -> kernel -> empty -> ...
//=============================================================================
double build_flat_graph_with_empty(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                                    int iterations, int sleep_count,
                                    unsigned int blocks, unsigned int threads)
{
    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    cudaGraphCreate(graph, 0);

    cudaGraphNode_t prevNode = NULL;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;

    for (int i = 0; i < iterations; i++) {
        cudaGraphNode_t kernelNode, emptyNode;

        // Add kernel node
        if (prevNode) {
            cudaGraphAddKernelNode(&kernelNode, *graph, &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&kernelNode, *graph, NULL, 0, &kernelParams);
        }

        // Add empty node after kernel
        cudaGraphAddEmptyNode(&emptyNode, *graph, &kernelNode, 1);
        prevNode = emptyNode;
    }

    cudaGraphInstantiate(graphExec, *graph, 0);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

//=============================================================================
// Measure functions
//=============================================================================
double measure_flat_graph(int iterations, int sleep_count,
                          unsigned int blocks, unsigned int threads,
                          double* out_construct_time)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        total_construct += build_flat_graph(&g, &ge, iterations, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    build_flat_graph(&graph, &graphExec, iterations, sleep_count, blocks, threads);

    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    return total_time / MEASURE_RUNS;
}

double measure_flat_graph_with_empty(int iterations, int sleep_count,
                                      unsigned int blocks, unsigned int threads,
                                      double* out_construct_time)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        total_construct += build_flat_graph_with_empty(&g, &ge, iterations, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    build_flat_graph_with_empty(&graph, &graphExec, iterations, sleep_count, blocks, threads);

    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    return total_time / MEASURE_RUNS;
}

//=============================================================================
// Test Function
//=============================================================================
void Test_EmptyNode(unsigned int blocks, unsigned int threads)
{
    printf("_______________________________________________________________________\n");
    printf("Empty Node Overhead Test\n");
    printf("Compare: flat graph (kernels only) vs flat graph (kernel + empty node)\n\n");

    int iteration_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    printf("=== Graph Construction Overhead ===\n");
    printf("method\t\t\titerations\tnodes\t\tconstruct(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        double construct_flat, construct_empty;

        measure_flat_graph(iters, 5, blocks, threads, &construct_flat);
        measure_flat_graph_with_empty(iters, 5, blocks, threads, &construct_empty);

        printf("flat_kernel_only\t%d\t\t%d\t\t%.2f\n", iters, iters, construct_flat);
        printf("flat_kernel+empty\t%d\t\t%d\t\t%.2f\n", iters, iters * 2, construct_empty);
        printf("\n");
    }

    printf("=== Runtime Overhead ===\n");
    printf("method\t\t\titer\tblk\tthrd\tworkload\ttime(ns)\tper_iter(ns)\n");

    for (int sleep_count : {5, 10}) {
        int workload_ns = sleep_count * 1000;
        printf("\n--- Workload: %d ns per iteration ---\n", workload_ns);

        for (int t = 0; t < num_tests; t++) {
            int iters = iteration_counts[t];
            double construct_tmp;

            double time_flat = measure_flat_graph(iters, sleep_count, blocks, threads, &construct_tmp);
            double time_empty = measure_flat_graph_with_empty(iters, sleep_count, blocks, threads, &construct_tmp);

            printf("flat_kernel_only\t%d\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_flat, time_flat / iters);
            printf("flat_kernel+empty\t%d\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_empty, time_empty / iters);
            printf("\n");
        }
    }

    // Real overhead comparison
    printf("=== Real Per-Iteration Overhead (workload error eliminated) ===\n");
    printf("Formula: O = 2 * per_iter_5us - per_iter_10us\n\n");
    printf("method\t\t\titerations\treal_overhead(ns)\tempty_overhead(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        double construct_tmp;

        double time_flat_5 = measure_flat_graph(iters, 5, blocks, threads, &construct_tmp);
        double time_flat_10 = measure_flat_graph(iters, 10, blocks, threads, &construct_tmp);
        double real_flat = 2 * (time_flat_5 / iters) - (time_flat_10 / iters);

        double time_empty_5 = measure_flat_graph_with_empty(iters, 5, blocks, threads, &construct_tmp);
        double time_empty_10 = measure_flat_graph_with_empty(iters, 10, blocks, threads, &construct_tmp);
        double real_empty = 2 * (time_empty_5 / iters) - (time_empty_10 / iters);

        double empty_overhead = real_empty - real_flat;

        printf("flat_kernel_only\t%d\t\t%.2f\t\t-\n", iters, real_flat);
        printf("flat_kernel+empty\t%d\t\t%.2f\t\t%.2f\n", iters, real_empty, empty_overhead);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    unsigned int smx_count = deviceProp.multiProcessorCount;

    printf("CUDA Graph Empty Node Overhead Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("=======================================================================\n\n");

    Test_EmptyNode(smx_count, 1024);

    return 0;
}
