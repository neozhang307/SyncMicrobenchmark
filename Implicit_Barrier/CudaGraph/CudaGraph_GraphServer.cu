// Graph Server Pattern Test
// Compare tail-launch chain vs while conditional loop
//
// Both patterns execute the same workload N times:
// 1. Tail-launch chain: Single graph tail-launches itself N times
// 2. While conditional: Single graph with while conditional loops N times
//
// Requires CUDA 12.0+

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>

#define WARMUP_RUNS 3
#define MEASURE_RUNS 10

// Sleep instruction: 1000 ns per iteration (same as CudaGraph_Kernel.cuh)
#define SLP asm volatile("nanosleep.u32 1000;");

// 20us workload using inline assembly (more accurate than __nanosleep)
#define WORKLOAD_20US \
    SLP SLP SLP SLP SLP SLP SLP SLP SLP SLP \
    SLP SLP SLP SLP SLP SLP SLP SLP SLP SLP

// Expected workload duration in microseconds
#define EXPECTED_WORKLOAD_US 20.0

//=============================================================================
// Workload kernel with counter
//=============================================================================
__global__ void workload_with_counter(int* counter, int* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = atomicAdd(counter, 1);
        WORKLOAD_20US;
        output[idx % 1024] = idx;
    }
}

//=============================================================================
// Tail-launch kernel: runs workload then tail-launches self if iterations remain
// All threads do the sleep workload, only thread 0 handles counter and relaunch
//=============================================================================
__global__ void tail_launch_kernel(int* iteration_counter, int max_iterations, int* output) {
    // All threads do workload (20us sleep)
    WORKLOAD_20US;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current = atomicAdd(iteration_counter, 1);
        output[current % 1024] = current;

        // Tail-launch self if more iterations needed
        if (current < max_iterations - 1) {
            cudaGraphExec_t self = cudaGetCurrentGraphExec();
            cudaGraphLaunch(self, cudaStreamGraphTailLaunch);
        }
    }
}

//=============================================================================
// Test 1: Tail-launch chain
//=============================================================================
void Test_TailLaunchChain(int num_iterations)
{
    printf("=== Test 1: Tail-Launch Chain (%d iterations) ===\n", num_iterations);

    // Allocate counters and output
    int* d_counter;
    int* d_output;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_output, 1024 * sizeof(int));

    // Build graph with tail-launch kernel
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphCreate(&graph, 0);

    cudaKernelNodeParams kernelParams = {};
    void* kernelArgs[] = { &d_counter, &num_iterations, &d_output };
    kernelParams.func = (void*)tail_launch_kernel;
    kernelParams.gridDim = dim3(48);      // Match existing benchmark
    kernelParams.blockDim = dim3(1024);   // Match existing benchmark
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams);

    // Instantiate for device launch (required for tail-launch)
    cudaGraphInstantiate(&graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(graphExec, 0);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaMemset(d_counter, 0, sizeof(int));
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaMemset(d_counter, 0, sizeof(int));
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::micro>(end - start).count();
    }

    double avg_time = total_time / MEASURE_RUNS;
    double per_iter = avg_time / num_iterations;

    // Verify
    int final_count;
    cudaMemcpy(&final_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("  Total time: %.2f us\n", avg_time);
    printf("  Per-iteration: %.2f us\n", per_iter);
    printf("  Iterations executed: %d (expected: %d)\n", final_count, num_iterations);
    printf("  Overhead (subtract %.0fus workload): %.2f us\n\n", EXPECTED_WORKLOAD_US, per_iter - EXPECTED_WORKLOAD_US);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_counter);
    cudaFree(d_output);
}

//=============================================================================
// Combined workload + check kernel for while conditional (1 kernel per iter)
// All threads do the sleep workload, only thread 0 handles counter and condition
//=============================================================================
__global__ void workload_and_check(cudaGraphConditionalHandle handle, int* counter,
                                    int max_iterations, int* output) {
    // All threads do workload (20us sleep)
    WORKLOAD_20US;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = atomicAdd(counter, 1);
        output[idx % 1024] = idx;

        // Check and set condition for next iteration
        cudaGraphSetConditional(handle, (idx + 1) < max_iterations ? 1 : 0);
    }
}

//=============================================================================
// Test 2: While Conditional Loop
//=============================================================================
void Test_WhileConditional(int num_iterations)
{
    printf("=== Test 2: While Conditional Loop (%d iterations) ===\n", num_iterations);

    // Allocate counters and output
    int* d_counter;
    int* d_output;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_output, 1024 * sizeof(int));

    // Build graph with while conditional
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphCreate(&graph, 0);

    // Create conditional handle (with default value = 1 to start loop)
    cudaGraphConditionalHandle condHandle;
    cudaGraphConditionalHandleCreate(&condHandle, graph, 1, cudaGraphCondAssignDefault);

    // Conditional node params
    cudaGraphNodeParams condParams = {};
    condParams.type = cudaGraphNodeTypeConditional;
    condParams.conditional.handle = condHandle;
    condParams.conditional.type = cudaGraphCondTypeWhile;
    condParams.conditional.size = 1;

    cudaGraphNode_t condNode;
    cudaGraphAddNode(&condNode, graph, NULL, NULL, 0, &condParams);

    // Get body graph from params after node is added
    cudaGraph_t condBody = condParams.conditional.phGraph_out[0];

    // Add single combined workload+check kernel to conditional body graph
    cudaKernelNodeParams bodyParams = {};
    void* bodyArgs[] = { &condHandle, &d_counter, &num_iterations, &d_output };
    bodyParams.func = (void*)workload_and_check;
    bodyParams.gridDim = dim3(48);      // Match existing benchmark
    bodyParams.blockDim = dim3(1024);   // Match existing benchmark
    bodyParams.sharedMemBytes = 0;
    bodyParams.kernelParams = bodyArgs;
    bodyParams.extra = NULL;

    cudaGraphNode_t bodyNode;
    cudaGraphAddKernelNode(&bodyNode, condBody, NULL, 0, &bodyParams);

    // Instantiate
    cudaGraphInstantiate(&graphExec, graph, 0);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaMemset(d_counter, 0, sizeof(int));
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaMemset(d_counter, 0, sizeof(int));
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::micro>(end - start).count();
    }

    double avg_time = total_time / MEASURE_RUNS;
    double per_iter = avg_time / num_iterations;

    // Verify
    int final_count;
    cudaMemcpy(&final_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("  Total time: %.2f us\n", avg_time);
    printf("  Per-iteration: %.2f us\n", per_iter);
    printf("  Iterations executed: %d (expected: %d)\n", final_count, num_iterations);
    printf("  Overhead (subtract %.0fus workload): %.2f us\n\n", EXPECTED_WORKLOAD_US, per_iter - EXPECTED_WORKLOAD_US);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_counter);
    cudaFree(d_output);
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    printf("CUDA Graph: Tail-Launch vs While Conditional Comparison\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, deviceProp.multiProcessorCount);
    printf("CUDA Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("=======================================================================\n\n");

    if (deviceProp.major < 9) {
        printf("WARNING: While conditional requires CUDA 12.4+ and SM 9.0+\n");
        printf("         Only running tail-launch test\n\n");
    }

    int iterations[] = {10, 50, 100, 500};

    for (int num_iter : iterations) {
        printf("-----------------------------------------------------------------------\n");
        Test_TailLaunchChain(num_iter);

        if (deviceProp.major >= 9) {
            Test_WhileConditional(num_iter);
        }
    }

    printf("=======================================================================\n");
    printf("Summary:\n");
    printf("- Tail-launch: Self-relaunches via cudaStreamGraphTailLaunch\n");
    printf("- While conditional: Loops via cudaGraphCondTypeWhile\n");
    printf("- Both execute same %.0fus workload per iteration\n", EXPECTED_WORKLOAD_US);

    return 0;
}
