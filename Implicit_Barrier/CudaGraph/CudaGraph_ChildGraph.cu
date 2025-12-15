// Child Graph benchmark: measure overhead of CUDA Graph child graph iteration
// Builds iteration count using binary composition: g1, g2=g1+g1, g4=g2+g2, etc.
// Iteration count must be power of 2

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>
#include <cassert>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

// Check if n is power of 2
static inline bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Get log2 of power of 2
static inline int log2_int(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

//=============================================================================
// Build child graph hierarchy: g1 -> g2 -> g4 -> ... -> gN
// Returns array of graphs where graphs[i] has 2^i iterations
//=============================================================================
double build_child_graph_hierarchy(cudaGraph_t* graphs, cudaGraphExec_t* graphExec,
                                    int max_iterations, int sleep_count,
                                    unsigned int blocks, unsigned int threads)
{
    assert(is_power_of_2(max_iterations));
    int levels = log2_int(max_iterations) + 1;  // levels needed: 0,1,2,...,log2(N)

    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    // Level 0: g1 - single kernel
    cudaGraphCreate(&graphs[0], 0);
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;
    cudaGraphAddKernelNode(&kernelNode, graphs[0], NULL, 0, &kernelParams);

    // Build hierarchy: g[i] = g[i-1] + g[i-1]
    for (int level = 1; level < levels; level++) {
        cudaGraphCreate(&graphs[level], 0);

        // Add first child graph node
        cudaGraphNode_t childNode1;
        cudaGraphAddChildGraphNode(&childNode1, graphs[level], NULL, 0, graphs[level - 1]);

        // Add second child graph node (depends on first to ensure sequential execution)
        cudaGraphNode_t childNode2;
        cudaGraphAddChildGraphNode(&childNode2, graphs[level], &childNode1, 1, graphs[level - 1]);
    }

    // Instantiate the final graph (graphs[levels-1] has max_iterations kernels)
    cudaGraphInstantiate(graphExec, graphs[levels - 1], 0);

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count();
}

//=============================================================================
// Measure child graph method
//=============================================================================
double measure_child_graph(int iterations, int sleep_count,
                           unsigned int blocks, unsigned int threads,
                           double* out_construct_time)
{
    assert(is_power_of_2(iterations));
    int levels = log2_int(iterations) + 1;

    // Allocate array for graph hierarchy
    cudaGraph_t* graphs = new cudaGraph_t[levels];
    cudaGraphExec_t graphExec;

    // Measure construction time (average over multiple builds)
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t* g = new cudaGraph_t[levels];
        cudaGraphExec_t ge;

        total_construct += build_child_graph_hierarchy(g, &ge, iterations, sleep_count, blocks, threads);

        cudaGraphExecDestroy(ge);
        for (int i = 0; i < levels; i++) {
            cudaGraphDestroy(g[i]);
        }
        delete[] g;
    }
    *out_construct_time = total_construct / 10.0;

    // Build final graph for launch measurement
    build_child_graph_hierarchy(graphs, &graphExec, iterations, sleep_count, blocks, threads);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure launch time
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    for (int i = 0; i < levels; i++) {
        cudaGraphDestroy(graphs[i]);
    }
    delete[] graphs;

    return total_time / MEASURE_RUNS;
}

//=============================================================================
// For comparison: flat graph with N kernels (no child graphs)
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

double measure_flat_graph(int iterations, int sleep_count,
                          unsigned int blocks, unsigned int threads,
                          double* out_construct_time)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Measure construction time
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        total_construct += build_flat_graph(&g, &ge, iterations, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    // Build final graph
    build_flat_graph(&graph, &graphExec, iterations, sleep_count, blocks, threads);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure
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
// Merged flat graph: build flat graph with N/2 kernels, then merge two copies
//=============================================================================
double build_merged_flat_graph(cudaGraph_t* baseGraph, cudaGraph_t* mergedGraph,
                                cudaGraphExec_t* graphExec,
                                int total_iterations, int sleep_count,
                                unsigned int blocks, unsigned int threads)
{
    assert(total_iterations >= 2 && total_iterations % 2 == 0);
    int half_iters = total_iterations / 2;

    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    // Build flat base graph with half the kernels
    cudaGraphCreate(baseGraph, 0);

    cudaGraphNode_t prevNode = NULL;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;

    for (int i = 0; i < half_iters; i++) {
        cudaGraphNode_t kernelNode;
        if (prevNode) {
            cudaGraphAddKernelNode(&kernelNode, *baseGraph, &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&kernelNode, *baseGraph, NULL, 0, &kernelParams);
        }
        prevNode = kernelNode;
    }

    // Create merged graph with two child graph nodes
    cudaGraphCreate(mergedGraph, 0);

    cudaGraphNode_t childNode1, childNode2;
    cudaGraphAddChildGraphNode(&childNode1, *mergedGraph, NULL, 0, *baseGraph);
    cudaGraphAddChildGraphNode(&childNode2, *mergedGraph, &childNode1, 1, *baseGraph);

    cudaGraphInstantiate(graphExec, *mergedGraph, 0);

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count();
}

double measure_merged_flat_graph(int iterations, int sleep_count,
                                  unsigned int blocks, unsigned int threads,
                                  double* out_construct_time)
{
    cudaGraph_t baseGraph, mergedGraph;
    cudaGraphExec_t graphExec;

    // Measure construction time
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t bg, mg;
        cudaGraphExec_t ge;
        total_construct += build_merged_flat_graph(&bg, &mg, &ge, iterations, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(mg);
        cudaGraphDestroy(bg);
    }
    *out_construct_time = total_construct / 10.0;

    // Build final graph
    build_merged_flat_graph(&baseGraph, &mergedGraph, &graphExec, iterations, sleep_count, blocks, threads);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(mergedGraph);
    cudaGraphDestroy(baseGraph);

    return total_time / MEASURE_RUNS;
}

//=============================================================================
// Test Function
//=============================================================================
void Test_ChildGraph(unsigned int blocks, unsigned int threads)
{
    printf("_______________________________________________________________________\n");
    printf("Child Graph Iteration Test\n");
    printf("Comparing: child_graph (g2=g1+g1, g4=g2+g2, ...) vs flat_graph\n");
    printf("Iteration count must be power of 2\n\n");

    // Test iteration counts (powers of 2)
    int iteration_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    printf("=== Graph Construction Overhead ===\n");
    printf("method\t\titerations\tconstruct(ns)\tlevels\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        double construct_child, construct_flat;

        measure_child_graph(iters, 5, blocks, threads, &construct_child);
        measure_flat_graph(iters, 5, blocks, threads, &construct_flat);

        int levels = log2_int(iters) + 1;
        printf("child_graph\t%d\t\t%.2f\t\t%d\n", iters, construct_child, levels);
        printf("flat_graph\t%d\t\t%.2f\t\t1\n", iters, construct_flat);
        printf("\n");
    }

    printf("=== Runtime Overhead ===\n");
    printf("method\t\titerations\tblk\tthrd\tworkload(ns)\ttime(ns)\tper_iter(ns)\n");

    for (int sleep_count : {5, 10}) {
        int workload_ns = sleep_count * 1000;
        printf("\n--- Workload: %d ns per iteration ---\n", workload_ns);

        for (int t = 0; t < num_tests; t++) {
            int iters = iteration_counts[t];
            double construct_tmp;

            // Child graph
            double time_child = measure_child_graph(iters, sleep_count, blocks, threads, &construct_tmp);
            double per_iter_child = time_child / iters;
            printf("child_graph\t%d\t\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_child, per_iter_child);

            // Flat graph
            double time_flat = measure_flat_graph(iters, sleep_count, blocks, threads, &construct_tmp);
            double per_iter_flat = time_flat / iters;
            printf("flat_graph\t%d\t\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_flat, per_iter_flat);

            printf("\n");
        }
    }

    // Calculate real overhead using dual-workload method
    printf("=== Real Per-Iteration Overhead (workload error eliminated) ===\n");
    printf("Formula: O = 2 * per_iter_5us - per_iter_10us\n\n");
    printf("method\t\titerations\treal_overhead(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        double construct_tmp;

        // Child graph
        double time_child_5 = measure_child_graph(iters, 5, blocks, threads, &construct_tmp);
        double time_child_10 = measure_child_graph(iters, 10, blocks, threads, &construct_tmp);
        double per_iter_child_5 = time_child_5 / iters;
        double per_iter_child_10 = time_child_10 / iters;
        double real_child = 2 * per_iter_child_5 - per_iter_child_10;
        printf("child_graph\t%d\t\t%.2f\n", iters, real_child);

        // Flat graph
        double time_flat_5 = measure_flat_graph(iters, 5, blocks, threads, &construct_tmp);
        double time_flat_10 = measure_flat_graph(iters, 10, blocks, threads, &construct_tmp);
        double per_iter_flat_5 = time_flat_5 / iters;
        double per_iter_flat_10 = time_flat_10 / iters;
        double real_flat = 2 * per_iter_flat_5 - per_iter_flat_10;
        printf("flat_graph\t%d\t\t%.2f\n", iters, real_flat);

        printf("\n");
    }

    // Test merged flat graphs: merge two flat_N/2 graphs to get N iterations
    printf("=== Merged Flat Graph Test ===\n");
    printf("Compare: flat_128 vs merged_flat_64x2 (two flat_64 merged with child nodes)\n\n");

    printf("method\t\t\titerations\tconstruct(ns)\n");

    // Test different merge sizes
    int merge_tests[] = {8, 16, 32, 64, 128};
    for (int total : merge_tests) {
        double construct_flat, construct_merged;

        measure_flat_graph(total, 5, blocks, threads, &construct_flat);
        measure_merged_flat_graph(total, 5, blocks, threads, &construct_merged);

        printf("flat_%d\t\t\t%d\t\t%.2f\n", total, total, construct_flat);
        printf("merged_flat_%dx2\t\t%d\t\t%.2f\n", total/2, total, construct_merged);
        printf("\n");
    }

    // Runtime comparison for 128 iterations
    printf("Runtime comparison (128 iterations):\n");
    printf("method\t\t\tworkload(ns)\ttime(ns)\tper_iter(ns)\n");

    for (int sleep_count : {5, 10}) {
        int workload_ns = sleep_count * 1000;
        double construct_tmp;

        double time_flat = measure_flat_graph(128, sleep_count, blocks, threads, &construct_tmp);
        double time_merged = measure_merged_flat_graph(128, sleep_count, blocks, threads, &construct_tmp);

        printf("flat_128\t\t%d\t\t%.2f\t\t%.2f\n", workload_ns, time_flat, time_flat / 128);
        printf("merged_flat_64x2\t%d\t\t%.2f\t\t%.2f\n", workload_ns, time_merged, time_merged / 128);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    unsigned int smx_count = deviceProp.multiProcessorCount;

    printf("CUDA Graph Child Graph Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("=======================================================================\n\n");

    Test_ChildGraph(smx_count, 1024);

    return 0;
}
