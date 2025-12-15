// While Conditional Node benchmark: measure overhead of CUDA Graph while loop
// Compares: while_conditional vs host_loop_graph vs device_loop
// Requires CUDA 12.4+

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include "../../share/repeat.h"
#include <stdio.h>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Sleep kernels defined in CudaGraph_Kernels.cu

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

// Kernel that decrements counter and sets condition to 0 when done
__global__ void decrement_and_check(cudaGraphConditionalHandle handle, int* counter)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int val = atomicSub(counter, 1);
        if (val <= 1) {
            cudaGraphSetConditional(handle, 0);
        }
    }
}

// Device-side loop kernel with grid sync
__global__ void device_loop_kernel(int iterations, int sleep_count)
{
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < iterations; i++) {
        // Sleep workload
        for (int s = 0; s < sleep_count; s++) {
            asm volatile("nanosleep.u32 1000;");
        }
        // Grid sync between iterations
        grid.sync();
    }
}

//=============================================================================
// Method 1: While Conditional Node
//=============================================================================

// Helper to build while conditional graph (returns construction time in ns)
double build_while_conditional_graph(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                                      int sleep_count, unsigned int blocks, unsigned int threads,
                                      int* d_counter, cudaGraphConditionalHandle* out_handle)
{
    cudaGraphNode_t condNode;
    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    // Create the main graph
    cudaGraphCreate(graph, 0);

    // Create conditional handle with default value = 1 (loop starts)
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, *graph, 1, cudaGraphCondAssignDefault);
    *out_handle = handle;

    // Create while conditional node
    cudaGraphNodeParams cParams = {};
    cParams.type = cudaGraphNodeTypeConditional;
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeWhile;
    cParams.conditional.size = 1;
    cudaGraphAddNode(&condNode, *graph, NULL, NULL, 0, &cParams);

    // Get the body graph
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Add sleep kernel to body graph
    cudaGraphNode_t sleepNode;
    cudaKernelNodeParams sleepParams = {};
    sleepParams.func = (void*)sleep_kernel;
    sleepParams.gridDim = dim3(blocks);
    sleepParams.blockDim = dim3(threads);
    sleepParams.sharedMemBytes = 0;
    sleepParams.kernelParams = NULL;
    sleepParams.extra = NULL;
    cudaGraphAddKernelNode(&sleepNode, bodyGraph, NULL, 0, &sleepParams);

    // Add decrement kernel to body graph (depends on sleep kernel)
    cudaGraphNode_t decNode;
    cudaKernelNodeParams decParams = {};
    void* decArgs[2] = {&handle, &d_counter};
    decParams.func = (void*)decrement_and_check;
    decParams.gridDim = dim3(1);
    decParams.blockDim = dim3(1);
    decParams.sharedMemBytes = 0;
    decParams.kernelParams = decArgs;
    decParams.extra = NULL;
    cudaGraphAddKernelNode(&decNode, bodyGraph, &sleepNode, 1, &decParams);

    // Instantiate the graph
    cudaGraphInstantiate(graphExec, *graph, 0);

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count();
}

double measure_while_conditional(int iterations, int sleep_count,
                                  unsigned int blocks, unsigned int threads,
                                  int* d_counter, double* out_construct_time)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphConditionalHandle handle;

    // Measure construction time (average over multiple builds)
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        cudaGraphConditionalHandle h;
        total_construct += build_while_conditional_graph(&g, &ge, sleep_count, blocks, threads, d_counter, &h);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    // Build final graph for launch measurement
    build_while_conditional_graph(&graph, &graphExec, sleep_count, blocks, threads, d_counter, &handle);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaMemcpy(d_counter, &iterations, sizeof(int), cudaMemcpyHostToDevice);
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure launch time
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaMemcpy(d_counter, &iterations, sizeof(int), cudaMemcpyHostToDevice);

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
// Method 2: Host Loop with Graph Launch
//=============================================================================
double measure_host_loop_graph(int iterations, int sleep_count,
                                unsigned int blocks, unsigned int threads,
                                double* out_construct_time)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    // Measure construction time
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;

        auto start = std::chrono::high_resolution_clock::now();
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        sleep_kernel<<<blocks, threads, 0, stream>>>();
        cudaStreamEndCapture(stream, &g);
        cudaGraphInstantiate(&ge, g, 0);
        auto end = std::chrono::high_resolution_clock::now();

        total_construct += std::chrono::duration<double, std::nano>(end - start).count();
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    // Build final graph for launch measurement
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    sleep_kernel<<<blocks, threads, 0, stream>>>();
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, 0);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (int i = 0; i < iterations; i++) {
            cudaGraphLaunch(graphExec, stream);
        }
        cudaStreamSynchronize(stream);
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            cudaGraphLaunch(graphExec, stream);
        }
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return total_time / MEASURE_RUNS;
}

//=============================================================================
// Method 3: Device Loop with Grid Sync
//=============================================================================
double measure_device_loop(int iterations, int sleep_count,
                           unsigned int blocks, unsigned int threads)
{
    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        void* args[] = {&iterations, &sleep_count};
        cudaLaunchCooperativeKernel((void*)device_loop_kernel,
                                     dim3(blocks), dim3(threads), args, 0, 0);
        cudaDeviceSynchronize();
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        void* args[] = {&iterations, &sleep_count};

        auto start = std::chrono::high_resolution_clock::now();
        cudaLaunchCooperativeKernel((void*)device_loop_kernel,
                                     dim3(blocks), dim3(threads), args, 0, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    return total_time / MEASURE_RUNS;
}

//=============================================================================
// Test Function
//=============================================================================
void Test_WhileConditional(unsigned int blocks, unsigned int threads)
{
    printf("_______________________________________________________________________\n");
    printf("While Conditional Node Test\n");
    printf("Comparing: while_conditional vs host_loop_graph vs device_loop\n\n");

    // Allocate counter for while conditional
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));

    // First, report construction overhead
    printf("=== Graph Construction Overhead ===\n");
    double construct_while, construct_host;

    // Measure construction for while conditional (1 iteration just for construction measurement)
    measure_while_conditional(1, 5, blocks, threads, d_counter, &construct_while);
    measure_host_loop_graph(1, 5, blocks, threads, &construct_host);

    printf("method\t\t\tconstruct(ns)\n");
    printf("while_conditional\t%.2f\n", construct_while);
    printf("host_loop_graph\t\t%.2f\n", construct_host);
    printf("device_loop\t\tN/A (no graph)\n");
    printf("\n");

    // Now report runtime overhead
    printf("=== Runtime Overhead ===\n");
    int iteration_counts[] = {1, 4, 16, 64, 128};
    int num_tests = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    printf("method\titerations\tblk\tthrd\tworkload(ns)\ttime(ns)\tper_iter(ns)\n");

    for (int sleep_count : {5, 10}) {
        int workload_ns = sleep_count * 1000;
        printf("\n--- Workload: %d ns per iteration ---\n", workload_ns);

        for (int t = 0; t < num_tests; t++) {
            int iters = iteration_counts[t];
            double construct_tmp;

            // While conditional
            double time_while = measure_while_conditional(iters, sleep_count, blocks, threads, d_counter, &construct_tmp);
            double per_iter_while = time_while / iters;
            printf("while_conditional\t%d\t%u\t%u\t%d\t%.2f\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_while, per_iter_while);

            // Host loop graph
            double time_host = measure_host_loop_graph(iters, sleep_count, blocks, threads, &construct_tmp);
            double per_iter_host = time_host / iters;
            printf("host_loop_graph\t%d\t%u\t%u\t%d\t%.2f\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_host, per_iter_host);

            // Device loop
            double time_device = measure_device_loop(iters, sleep_count, blocks, threads);
            double per_iter_device = time_device / iters;
            printf("device_loop\t%d\t%u\t%u\t%d\t%.2f\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_device, per_iter_device);

            printf("\n");
        }
    }

    cudaFree(d_counter);
}
