// Stream Capture benchmark: measure graph construction and launch overhead
// Tests how overhead scales with number of captured kernels

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include "../../share/repeat.h"
#include <stdio.h>
#include <chrono>

// Sleep kernels defined in CudaGraph_Kernels.cu

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

// Global to store incremental overhead for cross-test calculation
double g_last_incr_overhead = 0;

// Measure graph with N kernels: construction time and launch+sync time
void measureGraph(KernelFunc kernel, int numKernels,
                  unsigned int blocks, unsigned int threads,
                  cudaStream_t stream,
                  double* mean_construct, double* std_construct,
                  double* mean_launch, double* std_launch)
{
    double construct_times[MEASURE_RUNS];
    double launch_times[MEASURE_RUNS];

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < numKernels; i++) {
            kernel<<<blocks, threads, 0, stream>>>();
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, 0);
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }

    // Measure
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        // Measure construction
        auto start_c = std::chrono::high_resolution_clock::now();
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < numKernels; i++) {
            kernel<<<blocks, threads, 0, stream>>>();
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, 0);
        auto end_c = std::chrono::high_resolution_clock::now();
        construct_times[r] = std::chrono::duration<double, std::nano>(end_c - start_c).count();

        // Measure launch + sync (total latency)
        auto start_l = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        auto end_l = std::chrono::high_resolution_clock::now();
        launch_times[r] = std::chrono::duration<double, std::nano>(end_l - start_l).count();

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }

    getStatistics(*mean_construct, *std_construct, construct_times, MEASURE_RUNS);
    getStatistics(*mean_launch, *std_launch, launch_times, MEASURE_RUNS);
}

void Test_StreamCapture_Scaling(unsigned int blocks, unsigned int threads, int workload_ns)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    KernelFunc kernel = (workload_ns == 5000) ? sleep_kernel_5 : sleep_kernel_10;

    printf("_______________________________________________________________________\n");
    printf("Stream Capture Scaling Test (workload=%d ns per kernel)\n", workload_ns);
    printf("method\tnKernels\tblk\tthrd\tm(construct)\ts(construct)\tm(launch)\ts(launch)\tm(overhead)\tideal_work\n");

    int kernel_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(kernel_counts) / sizeof(kernel_counts[0]);

    double results_construct[8], results_launch[8], results_overhead[8];

    for (int t = 0; t < num_tests; t++) {
        int n = kernel_counts[t];
        double mean_c, std_c, mean_l, std_l;

        measureGraph(kernel, n, blocks, threads, stream,
                     &mean_c, &std_c, &mean_l, &std_l);

        double ideal_work = (double)n * workload_ns;
        double overhead = mean_l - ideal_work;

        results_construct[t] = mean_c;
        results_launch[t] = mean_l;
        results_overhead[t] = overhead;

        printf("stream_capture\t%d\t%u\t%u\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.0f\n",
               n, blocks, threads, mean_c, std_c, mean_l, std_l, overhead, ideal_work);
    }

    // Compute and print incremental overhead
    printf("\n");
    printf("Overhead Analysis:\n");
    printf("  Basic overhead (1 kernel):        %.2f ns\n", results_overhead[0]);

    // Incremental overhead: (overhead_128 - overhead_1) / 127
    double incr_overhead = (results_overhead[7] - results_overhead[0]) / 127.0;
    printf("  Incremental overhead (per kernel): %.2f ns\n", incr_overhead);
    g_last_incr_overhead = incr_overhead;

    printf("\nConstruction Time Analysis:\n");
    printf("  Basic construction (1 kernel):     %.2f ns\n", results_construct[0]);
    double incr_construct = (results_construct[7] - results_construct[0]) / 127.0;
    printf("  Incremental construction (per kernel): %.2f ns\n", incr_construct);

    cudaStreamDestroy(stream);
}

// Run both 5us and 10us tests, then compute real overhead by eliminating workload error
// incr_5 = O + E  (O = real overhead, E = workload error per 5us)
// incr_10 = O + 2*E
// Therefore: O = 2*incr_5 - incr_10
double g_incr_overhead_5us = 0;
double g_incr_overhead_10us = 0;

void Test_StreamCapture_All(unsigned int blocks, unsigned int threads)
{
    // Test with 5us workload per kernel
    Test_StreamCapture_Scaling(blocks, threads, 5000);
    g_incr_overhead_5us = g_last_incr_overhead;

    printf("\n");

    // Test with 10us workload per kernel
    Test_StreamCapture_Scaling(blocks, threads, 10000);
    g_incr_overhead_10us = g_last_incr_overhead;

    // Compute real per-kernel overhead by eliminating workload error
    // incr_5 = O + E, incr_10 = O + 2*E => O = 2*incr_5 - incr_10
    double real_overhead = 2 * g_incr_overhead_5us - g_incr_overhead_10us;

    printf("\n");
    printf("=======================================================================\n");
    printf("Real Per-Kernel Overhead (workload error eliminated):\n");
    printf("  Formula: O = 2 * incr_5us - incr_10us\n");
    printf("  O = 2 * %.2f - %.2f = %.2f ns\n",
           g_incr_overhead_5us, g_incr_overhead_10us, real_overhead);
    printf("=======================================================================\n");
}
