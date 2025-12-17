// Device Graph Launch benchmark: measure overhead of device-side vs host-side graph launch
// Tests fire-and-forget launch, tail launch, and compares with host launch
// Uses dual-workload method: overhead = 2 * time_short - time_long
// Requires CUDA 12.0+

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>
#include <cassert>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

//=============================================================================
// Device-side launcher kernel for fire-and-forget
//=============================================================================
__global__ void launcher_kernel_fire_and_forget(cudaGraphExec_t graphExec, int num_launches) {
    // Only thread 0 launches the graph
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_launches; i++) {
            cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForget);
        }
    }
}

//=============================================================================
// Device-side launcher kernel with timing (measures API call overhead)
//=============================================================================
__global__ void launcher_kernel_timed(cudaGraphExec_t graphExec, int num_launches,
                                       long long* launch_times) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_launches; i++) {
            long long start = clock64();
            cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForget);
            long long end = clock64();
            launch_times[i] = end - start;
        }
    }
}

//=============================================================================
// Device-side tail launch kernel with timing
//=============================================================================
__global__ void launcher_kernel_tail_timed(int* counter, int max_iterations,
                                            long long* launch_times) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current = atomicAdd(counter, 1);
        if (current < max_iterations - 1) {
            long long start = clock64();
            cudaGraphExec_t currentGraph = cudaGetCurrentGraphExec();
            cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
            long long end = clock64();
            launch_times[current] = end - start;
        }
    }
}

//=============================================================================
// Device-side launcher kernel for tail launch (self-relaunch pattern)
//=============================================================================
__global__ void launcher_kernel_tail(int* counter, int max_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current = atomicAdd(counter, 1);
        if (current < max_iterations - 1) {
            // Relaunch self
            cudaGraphExec_t currentGraph = cudaGetCurrentGraphExec();
            cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
        }
    }
}

//=============================================================================
// Build a simple workload graph with N kernels
//=============================================================================
void build_workload_graph(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                          int num_kernels, KernelFunc kernel, void** kernel_args,
                          unsigned int blocks, unsigned int threads,
                          bool device_launch)
{
    cudaGraphCreate(graph, 0);

    cudaGraphNode_t prevNode = NULL;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;

    for (int i = 0; i < num_kernels; i++) {
        cudaGraphNode_t kernelNode;
        if (prevNode) {
            cudaGraphAddKernelNode(&kernelNode, *graph, &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&kernelNode, *graph, NULL, 0, &kernelParams);
        }
        prevNode = kernelNode;
    }

    // Instantiate with device launch flag if requested
    if (device_launch) {
        cudaGraphInstantiate(graphExec, *graph, cudaGraphInstantiateFlagDeviceLaunch);
        // Upload to device
        cudaGraphUpload(*graphExec, 0);
    } else {
        cudaGraphInstantiate(graphExec, *graph, 0);
    }
}

//=============================================================================
// Build launcher graph that launches workload graph via fire-and-forget
//=============================================================================
void build_fire_and_forget_launcher(cudaGraph_t* launcherGraph, cudaGraphExec_t* launcherExec,
                                    cudaGraphExec_t workloadExec, int num_launches)
{
    cudaGraphCreate(launcherGraph, 0);

    // Add launcher kernel node
    cudaGraphNode_t launcherNode;
    cudaKernelNodeParams launcherParams = {};
    launcherParams.func = (void*)launcher_kernel_fire_and_forget;
    launcherParams.gridDim = dim3(1);
    launcherParams.blockDim = dim3(1);
    launcherParams.sharedMemBytes = 0;
    void* args[] = { &workloadExec, &num_launches };
    launcherParams.kernelParams = args;
    launcherParams.extra = NULL;

    cudaGraphAddKernelNode(&launcherNode, *launcherGraph, NULL, 0, &launcherParams);

    cudaGraphInstantiate(launcherExec, *launcherGraph, 0);
}

//=============================================================================
// Build self-relaunching graph using tail launch
//=============================================================================
void build_tail_launch_graph(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                             int* d_counter, int max_iterations,
                             KernelFunc workload_kernel, void** workload_args,
                             unsigned int blocks, unsigned int threads)
{
    cudaGraphCreate(graph, 0);

    // First node: workload kernel
    cudaGraphNode_t workloadNode;
    cudaKernelNodeParams workloadParams = {};
    workloadParams.func = (void*)workload_kernel;
    workloadParams.gridDim = dim3(blocks);
    workloadParams.blockDim = dim3(threads);
    workloadParams.sharedMemBytes = 0;
    workloadParams.kernelParams = workload_args;
    workloadParams.extra = NULL;
    cudaGraphAddKernelNode(&workloadNode, *graph, NULL, 0, &workloadParams);

    // Second node: tail launcher kernel (depends on workload)
    cudaGraphNode_t tailNode;
    cudaKernelNodeParams tailParams = {};
    tailParams.func = (void*)launcher_kernel_tail;
    tailParams.gridDim = dim3(1);
    tailParams.blockDim = dim3(1);
    tailParams.sharedMemBytes = 0;
    void* tailArgs[] = { &d_counter, &max_iterations };
    tailParams.kernelParams = tailArgs;
    tailParams.extra = NULL;
    cudaGraphAddKernelNode(&tailNode, *graph, &workloadNode, 1, &tailParams);

    // Instantiate with device launch flag for tail launch
    cudaGraphInstantiate(graphExec, *graph, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(*graphExec, 0);
}

//=============================================================================
// Measure time for a given configuration
//=============================================================================
double measure_time(cudaGraphExec_t graphExec) {
    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    }

    // Measure
    double total = 0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::nano>(end - start).count();
    }
    return total / MEASURE_RUNS;
}

//=============================================================================
// Test 1: Host Launch Overhead with Dual-Workload Method
//=============================================================================
void Test_HostLaunch(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 1: Host Launch Overhead (Dual-Workload Method) ===\n");
    printf("overhead = 2 * time_short - time_long\n\n");

    int num_kernels_list[] = {1, 4, 16, 64};
    int num_launches_list[] = {1, 10, 50, 100};

    // Test with different workload pairs to find stable measurement
    struct WorkloadPair {
        KernelFunc short_kernel;
        KernelFunc long_kernel;
        const char* name;
    } workloads[] = {
        { sleep_kernel_5, sleep_kernel_10, "5us/10us" },
        { sleep_kernel_10, sleep_kernel_20, "10us/20us" },
        { sleep_kernel_20, sleep_kernel_40, "20us/40us" },
    };

    printf("kernels\tlaunches\tworkload\ttime_short(ns)\ttime_long(ns)\toverhead(ns)\tper_launch(ns)\n");

    for (int k = 0; k < 4; k++) {
        int num_kernels = num_kernels_list[k];

        for (int l = 0; l < 4; l++) {
            int num_launches = num_launches_list[l];

            for (int w = 0; w < 3; w++) {
                // Build short workload graph
                cudaGraph_t shortGraph;
                cudaGraphExec_t shortExec;
                build_workload_graph(&shortGraph, &shortExec, num_kernels,
                                     workloads[w].short_kernel, NULL, blocks, threads, false);

                // Build long workload graph
                cudaGraph_t longGraph;
                cudaGraphExec_t longExec;
                build_workload_graph(&longGraph, &longExec, num_kernels,
                                     workloads[w].long_kernel, NULL, blocks, threads, false);

                // Measure with host launch loop
                double short_time = 0, long_time = 0;

                // Warmup
                for (int r = 0; r < WARMUP_RUNS; r++) {
                    for (int i = 0; i < num_launches; i++) cudaGraphLaunch(shortExec, 0);
                    cudaDeviceSynchronize();
                }

                // Measure short
                for (int r = 0; r < MEASURE_RUNS; r++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < num_launches; i++) cudaGraphLaunch(shortExec, 0);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    short_time += std::chrono::duration<double, std::nano>(end - start).count();
                }
                short_time /= MEASURE_RUNS;

                // Warmup
                for (int r = 0; r < WARMUP_RUNS; r++) {
                    for (int i = 0; i < num_launches; i++) cudaGraphLaunch(longExec, 0);
                    cudaDeviceSynchronize();
                }

                // Measure long
                for (int r = 0; r < MEASURE_RUNS; r++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < num_launches; i++) cudaGraphLaunch(longExec, 0);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    long_time += std::chrono::duration<double, std::nano>(end - start).count();
                }
                long_time /= MEASURE_RUNS;

                double overhead = 2 * short_time - long_time;
                double per_launch = overhead / num_launches;

                printf("%d\t%d\t\t%s\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\n",
                       num_kernels, num_launches, workloads[w].name,
                       short_time, long_time, overhead, per_launch);

                cudaGraphExecDestroy(shortExec);
                cudaGraphDestroy(shortGraph);
                cudaGraphExecDestroy(longExec);
                cudaGraphDestroy(longGraph);
            }
            printf("\n");
        }
    }
}

//=============================================================================
// Test 2: Device Fire-and-Forget Launch Overhead with Dual-Workload Method
//=============================================================================
void Test_DeviceFireAndForget(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 2: Device Fire-and-Forget Launch Overhead ===\n");
    printf("Launcher graph launches workload graph N times via cudaStreamGraphFireAndForget\n");
    printf("overhead = 2 * time_short - time_long\n\n");

    int num_kernels_list[] = {1, 4, 16, 64};
    int num_launches_list[] = {1, 10, 50, 100};

    struct WorkloadPair {
        KernelFunc short_kernel;
        KernelFunc long_kernel;
        const char* name;
    } workloads[] = {
        { sleep_kernel_5, sleep_kernel_10, "5us/10us" },
        { sleep_kernel_10, sleep_kernel_20, "10us/20us" },
        { sleep_kernel_20, sleep_kernel_40, "20us/40us" },
    };

    printf("kernels\tlaunches\tworkload\ttime_short(ns)\ttime_long(ns)\toverhead(ns)\tper_launch(ns)\n");

    for (int k = 0; k < 4; k++) {
        int num_kernels = num_kernels_list[k];

        for (int l = 0; l < 4; l++) {
            int num_launches = num_launches_list[l];

            for (int w = 0; w < 3; w++) {
                // Build short workload graph (device-launchable)
                cudaGraph_t shortGraph;
                cudaGraphExec_t shortExec;
                build_workload_graph(&shortGraph, &shortExec, num_kernels,
                                     workloads[w].short_kernel, NULL, blocks, threads, true);

                // Build long workload graph (device-launchable)
                cudaGraph_t longGraph;
                cudaGraphExec_t longExec;
                build_workload_graph(&longGraph, &longExec, num_kernels,
                                     workloads[w].long_kernel, NULL, blocks, threads, true);

                // Build launcher graphs
                cudaGraph_t shortLauncherGraph, longLauncherGraph;
                cudaGraphExec_t shortLauncherExec, longLauncherExec;
                build_fire_and_forget_launcher(&shortLauncherGraph, &shortLauncherExec,
                                               shortExec, num_launches);
                build_fire_and_forget_launcher(&longLauncherGraph, &longLauncherExec,
                                               longExec, num_launches);

                double short_time = measure_time(shortLauncherExec);
                double long_time = measure_time(longLauncherExec);

                double overhead = 2 * short_time - long_time;
                double per_launch = overhead / num_launches;

                printf("%d\t%d\t\t%s\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\n",
                       num_kernels, num_launches, workloads[w].name,
                       short_time, long_time, overhead, per_launch);

                cudaGraphExecDestroy(shortLauncherExec);
                cudaGraphDestroy(shortLauncherGraph);
                cudaGraphExecDestroy(longLauncherExec);
                cudaGraphDestroy(longLauncherGraph);
                cudaGraphExecDestroy(shortExec);
                cudaGraphDestroy(shortGraph);
                cudaGraphExecDestroy(longExec);
                cudaGraphDestroy(longGraph);
            }
            printf("\n");
        }
    }
}

//=============================================================================
// Test 3: Tail Launch Overhead with Dual-Workload Method
//=============================================================================
void Test_TailLaunch(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 3: Tail Launch (Self-Relaunch) Overhead ===\n");
    printf("Graph relaunches itself N times via cudaStreamGraphTailLaunch\n");
    printf("overhead = 2 * time_short - time_long\n\n");

    int iterations_list[] = {1, 10, 50, 100};

    struct WorkloadPair {
        KernelFunc short_kernel;
        KernelFunc long_kernel;
        const char* name;
    } workloads[] = {
        { sleep_kernel_5, sleep_kernel_10, "5us/10us" },
        { sleep_kernel_10, sleep_kernel_20, "10us/20us" },
        { sleep_kernel_20, sleep_kernel_40, "20us/40us" },
    };

    printf("iterations\tworkload\ttime_short(ns)\ttime_long(ns)\toverhead(ns)\tper_iter(ns)\n");

    for (int i = 0; i < 4; i++) {
        int num_iter = iterations_list[i];

        for (int w = 0; w < 3; w++) {
            // Allocate counter
            int* d_counter;
            cudaMalloc(&d_counter, sizeof(int));

            // Build short tail launch graph
            cudaGraph_t shortGraph;
            cudaGraphExec_t shortExec;
            build_tail_launch_graph(&shortGraph, &shortExec, d_counter, num_iter,
                                    workloads[w].short_kernel, NULL, blocks, threads);

            // Build long tail launch graph
            cudaGraph_t longGraph;
            cudaGraphExec_t longExec;
            build_tail_launch_graph(&longGraph, &longExec, d_counter, num_iter,
                                    workloads[w].long_kernel, NULL, blocks, threads);

            // Measure short
            double short_time = 0;
            for (int r = 0; r < WARMUP_RUNS; r++) {
                cudaMemset(d_counter, 0, sizeof(int));
                cudaGraphLaunch(shortExec, 0);
                cudaDeviceSynchronize();
            }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                cudaMemset(d_counter, 0, sizeof(int));
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(shortExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                short_time += std::chrono::duration<double, std::nano>(end - start).count();
            }
            short_time /= MEASURE_RUNS;

            // Measure long
            double long_time = 0;
            for (int r = 0; r < WARMUP_RUNS; r++) {
                cudaMemset(d_counter, 0, sizeof(int));
                cudaGraphLaunch(longExec, 0);
                cudaDeviceSynchronize();
            }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                cudaMemset(d_counter, 0, sizeof(int));
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(longExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                long_time += std::chrono::duration<double, std::nano>(end - start).count();
            }
            long_time /= MEASURE_RUNS;

            double overhead = 2 * short_time - long_time;
            double per_iter = overhead / num_iter;

            printf("%d\t\t%s\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\n",
                   num_iter, workloads[w].name, short_time, long_time, overhead, per_iter);

            cudaGraphExecDestroy(shortExec);
            cudaGraphDestroy(shortGraph);
            cudaGraphExecDestroy(longExec);
            cudaGraphDestroy(longGraph);
            cudaFree(d_counter);
        }
        printf("\n");
    }
}

//=============================================================================
// Test 4: Host vs Device Launch Comparison (Best Workload)
//=============================================================================
void Test_HostVsDevice(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 4: Host vs Device Launch Comparison ===\n");
    printf("Using 20us/40us workload for stable measurement\n\n");

    int num_kernels = 16;
    int num_launches_list[] = {10, 50, 100};

    printf("launches\thost_overhead(ns)\tdevice_overhead(ns)\tspeedup\n");

    for (int l = 0; l < 3; l++) {
        int num_launches = num_launches_list[l];

        // --- Host Launch ---
        cudaGraph_t hostShortGraph, hostLongGraph;
        cudaGraphExec_t hostShortExec, hostLongExec;
        build_workload_graph(&hostShortGraph, &hostShortExec, num_kernels,
                             sleep_kernel_20, NULL, blocks, threads, false);
        build_workload_graph(&hostLongGraph, &hostLongExec, num_kernels,
                             sleep_kernel_40, NULL, blocks, threads, false);

        double host_short = 0, host_long = 0;

        // Warmup & measure short
        for (int r = 0; r < WARMUP_RUNS; r++) {
            for (int i = 0; i < num_launches; i++) cudaGraphLaunch(hostShortExec, 0);
            cudaDeviceSynchronize();
        }
        for (int r = 0; r < MEASURE_RUNS; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_launches; i++) cudaGraphLaunch(hostShortExec, 0);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            host_short += std::chrono::duration<double, std::nano>(end - start).count();
        }
        host_short /= MEASURE_RUNS;

        // Warmup & measure long
        for (int r = 0; r < WARMUP_RUNS; r++) {
            for (int i = 0; i < num_launches; i++) cudaGraphLaunch(hostLongExec, 0);
            cudaDeviceSynchronize();
        }
        for (int r = 0; r < MEASURE_RUNS; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_launches; i++) cudaGraphLaunch(hostLongExec, 0);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            host_long += std::chrono::duration<double, std::nano>(end - start).count();
        }
        host_long /= MEASURE_RUNS;

        double host_overhead = 2 * host_short - host_long;

        cudaGraphExecDestroy(hostShortExec);
        cudaGraphDestroy(hostShortGraph);
        cudaGraphExecDestroy(hostLongExec);
        cudaGraphDestroy(hostLongGraph);

        // --- Device Launch ---
        cudaGraph_t devShortGraph, devLongGraph;
        cudaGraphExec_t devShortExec, devLongExec;
        build_workload_graph(&devShortGraph, &devShortExec, num_kernels,
                             sleep_kernel_20, NULL, blocks, threads, true);
        build_workload_graph(&devLongGraph, &devLongExec, num_kernels,
                             sleep_kernel_40, NULL, blocks, threads, true);

        cudaGraph_t shortLauncherGraph, longLauncherGraph;
        cudaGraphExec_t shortLauncherExec, longLauncherExec;
        build_fire_and_forget_launcher(&shortLauncherGraph, &shortLauncherExec,
                                       devShortExec, num_launches);
        build_fire_and_forget_launcher(&longLauncherGraph, &longLauncherExec,
                                       devLongExec, num_launches);

        double dev_short = measure_time(shortLauncherExec);
        double dev_long = measure_time(longLauncherExec);
        double dev_overhead = 2 * dev_short - dev_long;

        printf("%d\t\t%.0f\t\t\t%.0f\t\t\t%.2fx\n",
               num_launches, host_overhead, dev_overhead,
               host_overhead / dev_overhead);

        cudaGraphExecDestroy(shortLauncherExec);
        cudaGraphDestroy(shortLauncherGraph);
        cudaGraphExecDestroy(longLauncherExec);
        cudaGraphDestroy(longLauncherGraph);
        cudaGraphExecDestroy(devShortExec);
        cudaGraphDestroy(devShortGraph);
        cudaGraphExecDestroy(devLongExec);
        cudaGraphDestroy(devLongGraph);
    }
    printf("\n");
}

//=============================================================================
// Test 5: API Call Overhead (Device-side timing with clock64)
//=============================================================================
void Test_APICallOverhead(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 5: API Call Overhead (Device-side clock64 timing) ===\n");
    printf("Measures actual cudaGraphLaunch API call time inside kernel\n\n");

    // Get GPU clock rate for conversion
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    printf("GPU clock rate: %.0f MHz\n\n", clock_rate_khz / 1000.0);

    int num_kernels_list[] = {1, 4, 16, 64};
    int num_launches = 100;

    printf("kernels\tnum_launches\tavg_cycles\tmin_cycles\tmax_cycles\tavg_ns\n");

    for (int k = 0; k < 4; k++) {
        int num_kernels = num_kernels_list[k];

        // Build workload graph (device-launchable)
        cudaGraph_t workloadGraph;
        cudaGraphExec_t workloadExec;
        build_workload_graph(&workloadGraph, &workloadExec, num_kernels,
                             sleep_kernel_5, NULL, blocks, threads, true);

        // Allocate timing array on device
        long long* d_launch_times;
        cudaMalloc(&d_launch_times, num_launches * sizeof(long long));

        // Build timed launcher graph
        cudaGraph_t launcherGraph;
        cudaGraphExec_t launcherExec;
        cudaGraphCreate(&launcherGraph, 0);

        cudaGraphNode_t launcherNode;
        cudaKernelNodeParams launcherParams = {};
        launcherParams.func = (void*)launcher_kernel_timed;
        launcherParams.gridDim = dim3(1);
        launcherParams.blockDim = dim3(1);
        launcherParams.sharedMemBytes = 0;
        void* args[] = { &workloadExec, &num_launches, &d_launch_times };
        launcherParams.kernelParams = args;
        launcherParams.extra = NULL;
        cudaGraphAddKernelNode(&launcherNode, launcherGraph, NULL, 0, &launcherParams);
        cudaGraphInstantiate(&launcherExec, launcherGraph, 0);

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; w++) {
            cudaGraphLaunch(launcherExec, 0);
            cudaDeviceSynchronize();
        }

        // Run and collect timing
        cudaGraphLaunch(launcherExec, 0);
        cudaDeviceSynchronize();

        // Copy timing data back
        long long* h_launch_times = new long long[num_launches];
        cudaMemcpy(h_launch_times, d_launch_times, num_launches * sizeof(long long), cudaMemcpyDeviceToHost);

        // Calculate statistics
        long long total = 0, min_val = h_launch_times[0], max_val = h_launch_times[0];
        for (int i = 0; i < num_launches; i++) {
            total += h_launch_times[i];
            if (h_launch_times[i] < min_val) min_val = h_launch_times[i];
            if (h_launch_times[i] > max_val) max_val = h_launch_times[i];
        }
        double avg_cycles = (double)total / num_launches;
        double avg_ns = avg_cycles / (double)clock_rate_khz * 1e6;  // cycles / (cycles/us) * 1000 = ns

        printf("%d\t%d\t\t%.0f\t\t%lld\t\t%lld\t\t%.0f\n",
               num_kernels, num_launches, avg_cycles, min_val, max_val, avg_ns);

        delete[] h_launch_times;
        cudaFree(d_launch_times);
        cudaGraphExecDestroy(launcherExec);
        cudaGraphDestroy(launcherGraph);
        cudaGraphExecDestroy(workloadExec);
        cudaGraphDestroy(workloadGraph);
    }

    printf("\n");

    // Also test host-side API call overhead (time before sync)
    printf("Host-side API call overhead (time for cudaGraphLaunch to return, before sync):\n");
    printf("kernels\tnum_launches\tavg_api_call(ns)\n");

    for (int k = 0; k < 4; k++) {
        int num_kernels = num_kernels_list[k];

        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        build_workload_graph(&graph, &graphExec, num_kernels,
                             sleep_kernel_5, NULL, blocks, threads, false);

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; w++) {
            for (int i = 0; i < num_launches; i++) {
                cudaGraphLaunch(graphExec, 0);
            }
            cudaDeviceSynchronize();
        }

        // Measure API call time (not including sync)
        double total_api_time = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_launches; i++) {
                cudaGraphLaunch(graphExec, 0);
            }
            auto end = std::chrono::high_resolution_clock::now();
            // Don't sync yet - just measure API call time
            total_api_time += std::chrono::duration<double, std::nano>(end - start).count();
            cudaDeviceSynchronize();  // sync after timing
        }
        total_api_time /= MEASURE_RUNS;

        printf("%d\t%d\t\t%.0f\n", num_kernels, num_launches, total_api_time / num_launches);

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }
    printf("\n");
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();

    unsigned int smx_count = deviceProp.multiProcessorCount;

    printf("CUDA Graph Device Launch Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("CUDA Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Using dual-workload method: overhead = 2 * time_short - time_long\n");
    printf("=======================================================================\n\n");

    // Check for device graph launch support (requires sm_70+)
    if (deviceProp.major < 7) {
        printf("ERROR: Device graph launch requires compute capability 7.0+\n");
        return 1;
    }

    Test_HostLaunch(smx_count, 1024);
    Test_DeviceFireAndForget(smx_count, 1024);
    Test_TailLaunch(smx_count, 1024);
    Test_HostVsDevice(smx_count, 1024);
    Test_APICallOverhead(smx_count, 1024);

    return 0;
}
