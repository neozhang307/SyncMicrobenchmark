// Event Node benchmark: measure overhead of inter-graph event synchronization
// Ping-pong pattern: two graphs alternating execution via event record/wait
// Execution order: A.sleep0 -> B.sleep0 -> A.sleep1 -> B.sleep1 -> ...

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

//=============================================================================
// Build ping-pong graphs with event synchronization
// Graph A: sleep[0] -> record[A0] -> wait[B0] -> sleep[1] -> ...
// Graph B: wait[A0] -> sleep[0] -> record[B0] -> wait[A1] -> ...
//=============================================================================
double build_pingpong_graphs(cudaGraph_t* graphA, cudaGraph_t* graphB,
                              cudaGraphExec_t* graphExecA, cudaGraphExec_t* graphExecB,
                              cudaEvent_t* eventsAtoB, cudaEvent_t* eventsBtoA,
                              int iterations, int sleep_count,
                              unsigned int blocks, unsigned int threads)
{
    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    // Create events
    // N events for A->B (A records after each sleep, B waits before each sleep)
    // N-1 events for B->A (B records after sleep[0..N-2], A waits before sleep[1..N-1])
    for (int i = 0; i < iterations; i++) {
        cudaEventCreate(&eventsAtoB[i]);
    }
    for (int i = 0; i < iterations - 1; i++) {
        cudaEventCreate(&eventsBtoA[i]);
    }

    // Build Graph A
    cudaGraphCreate(graphA, 0);

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;

    cudaGraphNode_t prevNodeA = NULL;

    for (int i = 0; i < iterations; i++) {
        // Wait for B (if not first iteration)
        if (i > 0) {
            cudaGraphNode_t waitNode;
            cudaGraphAddEventWaitNode(&waitNode, *graphA, &prevNodeA, 1, eventsBtoA[i - 1]);
            prevNodeA = waitNode;
        }

        // Sleep kernel
        cudaGraphNode_t sleepNode;
        if (prevNodeA) {
            cudaGraphAddKernelNode(&sleepNode, *graphA, &prevNodeA, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&sleepNode, *graphA, NULL, 0, &kernelParams);
        }
        prevNodeA = sleepNode;

        // Record event for B
        cudaGraphNode_t recordNode;
        cudaGraphAddEventRecordNode(&recordNode, *graphA, &prevNodeA, 1, eventsAtoB[i]);
        prevNodeA = recordNode;
    }

    // Build Graph B
    cudaGraphCreate(graphB, 0);

    cudaGraphNode_t prevNodeB = NULL;

    for (int i = 0; i < iterations; i++) {
        // Wait for A
        cudaGraphNode_t waitNode;
        if (prevNodeB) {
            cudaGraphAddEventWaitNode(&waitNode, *graphB, &prevNodeB, 1, eventsAtoB[i]);
        } else {
            cudaGraphAddEventWaitNode(&waitNode, *graphB, NULL, 0, eventsAtoB[i]);
        }
        prevNodeB = waitNode;

        // Sleep kernel
        cudaGraphNode_t sleepNode;
        cudaGraphAddKernelNode(&sleepNode, *graphB, &prevNodeB, 1, &kernelParams);
        prevNodeB = sleepNode;

        // Record event for A (if not last iteration)
        if (i < iterations - 1) {
            cudaGraphNode_t recordNode;
            cudaGraphAddEventRecordNode(&recordNode, *graphB, &prevNodeB, 1, eventsBtoA[i]);
            prevNodeB = recordNode;
        }
    }

    // Instantiate both graphs
    cudaGraphInstantiate(graphExecA, *graphA, 0);
    cudaGraphInstantiate(graphExecB, *graphB, 0);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

void cleanup_pingpong(cudaGraph_t graphA, cudaGraph_t graphB,
                       cudaGraphExec_t graphExecA, cudaGraphExec_t graphExecB,
                       cudaEvent_t* eventsAtoB, cudaEvent_t* eventsBtoA,
                       int iterations)
{
    cudaGraphExecDestroy(graphExecA);
    cudaGraphExecDestroy(graphExecB);
    cudaGraphDestroy(graphA);
    cudaGraphDestroy(graphB);

    for (int i = 0; i < iterations; i++) {
        cudaEventDestroy(eventsAtoB[i]);
    }
    for (int i = 0; i < iterations - 1; i++) {
        cudaEventDestroy(eventsBtoA[i]);
    }
}

//=============================================================================
// Measure ping-pong execution
//=============================================================================
double measure_pingpong(int iterations, int sleep_count,
                        unsigned int blocks, unsigned int threads,
                        double* out_construct_time)
{
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Measure construction time
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t gA, gB;
        cudaGraphExec_t geA, geB;
        cudaEvent_t* eAtoB = new cudaEvent_t[iterations];
        cudaEvent_t* eBtoA = new cudaEvent_t[iterations - 1];

        total_construct += build_pingpong_graphs(&gA, &gB, &geA, &geB,
                                                  eAtoB, eBtoA, iterations, sleep_count,
                                                  blocks, threads);

        cleanup_pingpong(gA, gB, geA, geB, eAtoB, eBtoA, iterations);
        delete[] eAtoB;
        delete[] eBtoA;
    }
    *out_construct_time = total_construct / 10.0;

    // Build for measurement
    cudaGraph_t graphA, graphB;
    cudaGraphExec_t graphExecA, graphExecB;
    cudaEvent_t* eventsAtoB = new cudaEvent_t[iterations];
    cudaEvent_t* eventsBtoA = new cudaEvent_t[iterations - 1];

    build_pingpong_graphs(&graphA, &graphB, &graphExecA, &graphExecB,
                          eventsAtoB, eventsBtoA, iterations, sleep_count,
                          blocks, threads);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExecA, stream1);
        cudaGraphLaunch(graphExecB, stream2);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    }

    // Measure
    double total_time = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExecA, stream1);
        cudaGraphLaunch(graphExecB, stream2);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    // Cleanup
    cleanup_pingpong(graphA, graphB, graphExecA, graphExecB,
                     eventsAtoB, eventsBtoA, iterations);
    delete[] eventsAtoB;
    delete[] eventsBtoA;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return total_time / MEASURE_RUNS;
}

//=============================================================================
// Single graph with ping-pong deps: each node depends on previous TWO nodes
// Mimics the cross-graph dependency pattern within one graph
// k0 -> k1 -> k2 (deps: k0,k1) -> k3 (deps: k1,k2) -> ...
//=============================================================================
double build_single_graph_pingpong_deps(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                                         int total_kernels, int sleep_count,
                                         unsigned int blocks, unsigned int threads)
{
    KernelFunc sleep_kernel = (sleep_count == 5) ? sleep_kernel_5 : sleep_kernel_10;

    auto start = std::chrono::high_resolution_clock::now();

    cudaGraphCreate(graph, 0);

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)sleep_kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = NULL;
    kernelParams.extra = NULL;

    // Store all nodes for dependency tracking
    cudaGraphNode_t* nodes = new cudaGraphNode_t[total_kernels];

    for (int i = 0; i < total_kernels; i++) {
        if (i == 0) {
            // First node: no dependencies
            cudaGraphAddKernelNode(&nodes[i], *graph, NULL, 0, &kernelParams);
        } else if (i == 1) {
            // Second node: depends on first
            cudaGraphAddKernelNode(&nodes[i], *graph, &nodes[0], 1, &kernelParams);
        } else {
            // All other nodes: depend on previous TWO nodes
            cudaGraphNode_t deps[2] = {nodes[i-2], nodes[i-1]};
            cudaGraphAddKernelNode(&nodes[i], *graph, deps, 2, &kernelParams);
        }
    }

    cudaGraphInstantiate(graphExec, *graph, 0);

    delete[] nodes;

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

double measure_single_graph_pingpong_deps(int total_kernels, int sleep_count,
                                           unsigned int blocks, unsigned int threads,
                                           double* out_construct_time)
{
    // Measure construction
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        total_construct += build_single_graph_pingpong_deps(&g, &ge, total_kernels, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    // Build for measurement
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    build_single_graph_pingpong_deps(&graph, &graphExec, total_kernels, sleep_count, blocks, threads);

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
// Single graph baseline: 2N kernels in sequence (no event sync)
//=============================================================================
double build_single_graph(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                          int total_kernels, int sleep_count,
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

    for (int i = 0; i < total_kernels; i++) {
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

double measure_single_graph(int total_kernels, int sleep_count,
                            unsigned int blocks, unsigned int threads,
                            double* out_construct_time)
{
    // Measure construction
    double total_construct = 0;
    for (int c = 0; c < 10; c++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        total_construct += build_single_graph(&g, &ge, total_kernels, sleep_count, blocks, threads);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    *out_construct_time = total_construct / 10.0;

    // Build for measurement
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    build_single_graph(&graph, &graphExec, total_kernels, sleep_count, blocks, threads);

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
// Test Function
//=============================================================================
void Test_EventNode(unsigned int blocks, unsigned int threads)
{
    printf("_______________________________________________________________________\n");
    printf("Event Node Ping-Pong Test\n");
    printf("Two graphs alternating via event record/wait nodes\n");
    printf("Execution: A.sleep0 -> B.sleep0 -> A.sleep1 -> B.sleep1 -> ...\n\n");

    int iteration_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_tests = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    printf("=== Graph Construction Overhead ===\n");
    printf("method\t\t\t\titers\tkernels\tevents\tconstruct(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        int total_events = (iters > 1) ? (2 * iters - 1) : 1;
        double construct_pingpong, construct_single, construct_ppdeps;

        measure_pingpong(iters, 5, blocks, threads, &construct_pingpong);
        measure_single_graph(2 * iters, 5, blocks, threads, &construct_single);
        measure_single_graph_pingpong_deps(2 * iters, 5, blocks, threads, &construct_ppdeps);

        printf("pingpong_2graph\t\t\t%d\t%d\t%d\t%.2f\n",
               iters, 2 * iters, total_events, construct_pingpong);
        printf("single_graph_linear\t\t%d\t%d\t0\t%.2f\n",
               iters, 2 * iters, construct_single);
        printf("single_graph_ppdeps\t\t%d\t%d\t0\t%.2f\n",
               iters, 2 * iters, construct_ppdeps);
        printf("\n");
    }

    printf("=== Runtime Overhead ===\n");
    printf("method\t\t\titers\tblk\tthrd\tworkload\ttime(ns)\tper_kernel(ns)\n");

    for (int sleep_count : {5, 10}) {
        int workload_ns = sleep_count * 1000;
        printf("\n--- Workload: %d ns per kernel ---\n", workload_ns);

        for (int t = 0; t < num_tests; t++) {
            int iters = iteration_counts[t];
            double construct_tmp;

            double time_pingpong = measure_pingpong(iters, sleep_count, blocks, threads, &construct_tmp);
            double time_single = measure_single_graph(2 * iters, sleep_count, blocks, threads, &construct_tmp);
            double time_ppdeps = measure_single_graph_pingpong_deps(2 * iters, sleep_count, blocks, threads, &construct_tmp);

            printf("pingpong_2graph\t\t%d\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_pingpong, time_pingpong / (2 * iters));
            printf("single_graph_linear\t%d\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_single, time_single / (2 * iters));
            printf("single_graph_ppdeps\t%d\t%u\t%u\t%d\t\t%.2f\t\t%.2f\n",
                   iters, blocks, threads, workload_ns, time_ppdeps, time_ppdeps / (2 * iters));
            printf("\n");
        }
    }

    // Per-sync overhead using dual-workload method
    printf("=== Per-Sync Event Overhead (workload error eliminated) ===\n");
    printf("Formula: total_overhead = 2 * time_5us - time_10us\n");
    printf("         per_sync_overhead = (pp_overhead - baseline_overhead) / num_sync_events\n\n");
    printf("iters\tnum_syncs\tpp_overhead\tlinear_overhead\tppdeps_overhead\tper_sync_vs_linear\tper_sync_vs_ppdeps\n");

    for (int t = 0; t < num_tests; t++) {
        int iters = iteration_counts[t];
        int num_syncs = (iters > 1) ? (2 * iters - 1) : 1;
        double construct_tmp;

        // Measure total times
        double time_pp_5 = measure_pingpong(iters, 5, blocks, threads, &construct_tmp);
        double time_pp_10 = measure_pingpong(iters, 10, blocks, threads, &construct_tmp);

        double time_sg_5 = measure_single_graph(2 * iters, 5, blocks, threads, &construct_tmp);
        double time_sg_10 = measure_single_graph(2 * iters, 10, blocks, threads, &construct_tmp);

        double time_ppdeps_5 = measure_single_graph_pingpong_deps(2 * iters, 5, blocks, threads, &construct_tmp);
        double time_ppdeps_10 = measure_single_graph_pingpong_deps(2 * iters, 10, blocks, threads, &construct_tmp);

        // Real total overhead (workload eliminated)
        double real_pp_total = 2 * time_pp_5 - time_pp_10;
        double real_sg_total = 2 * time_sg_5 - time_sg_10;
        double real_ppdeps_total = 2 * time_ppdeps_5 - time_ppdeps_10;

        // Per-sync overhead vs each baseline
        double per_sync_vs_linear = (real_pp_total - real_sg_total) / num_syncs;
        double per_sync_vs_ppdeps = (real_pp_total - real_ppdeps_total) / num_syncs;

        printf("%d\t%d\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\t\t\t%.0f\n",
               iters, num_syncs, real_pp_total, real_sg_total, real_ppdeps_total,
               per_sync_vs_linear, per_sync_vs_ppdeps);
    }
}

int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    unsigned int smx_count = deviceProp.multiProcessorCount;

    printf("CUDA Graph Event Node Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("=======================================================================\n\n");

    Test_EventNode(smx_count, 1024);

    return 0;
}
