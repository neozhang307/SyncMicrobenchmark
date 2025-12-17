// Graph Upload Overhead Analysis
// Tests CPU-side API overhead, GPU-side overhead, and contention effects
// Requires CUDA 12.0+

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>
#include <vector>
#include <cublas_v2.h>
#include <curand.h>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100

//=============================================================================
// Build a workload graph with N kernels (for upload testing)
//=============================================================================
void build_graph_for_upload(cudaGraph_t* graph, int num_kernels,
                            KernelFunc kernel, void** kernel_args,
                            unsigned int blocks, unsigned int threads)
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
}

//=============================================================================
// Test 1: CPU-side API overhead of cudaGraphInstantiate and cudaGraphUpload
//=============================================================================
void Test_InstantiateUploadAPIOverhead(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 1: CPU-side API Overhead ===\n");
    printf("Measures time for cudaGraphInstantiate and cudaGraphUpload to return\n\n");

    int num_kernels_list[] = {1, 4, 16, 64, 256};

    printf("kernels\tinstantiate_regular(us)\tinstantiate_device(us)\tupload(us)\ttotal_device(us)\n");

    for (int k = 0; k < 5; k++) {
        int num_kernels = num_kernels_list[k];

        double instantiate_regular_time = 0;
        double instantiate_device_time = 0;
        double upload_time = 0;

        for (int r = 0; r < MEASURE_RUNS; r++) {
            // Build fresh graph each iteration
            cudaGraph_t graph;
            build_graph_for_upload(&graph, num_kernels, sleep_kernel_5, NULL, blocks, threads);

            // Measure regular instantiate
            cudaGraphExec_t regularExec;
            auto start = std::chrono::high_resolution_clock::now();
            cudaGraphInstantiate(&regularExec, graph, 0);
            auto end = std::chrono::high_resolution_clock::now();
            instantiate_regular_time += std::chrono::duration<double, std::micro>(end - start).count();
            cudaGraphExecDestroy(regularExec);

            // Measure device-launchable instantiate
            cudaGraphExec_t deviceExec;
            start = std::chrono::high_resolution_clock::now();
            cudaGraphInstantiate(&deviceExec, graph, cudaGraphInstantiateFlagDeviceLaunch);
            end = std::chrono::high_resolution_clock::now();
            instantiate_device_time += std::chrono::duration<double, std::micro>(end - start).count();

            // Measure upload
            start = std::chrono::high_resolution_clock::now();
            cudaGraphUpload(deviceExec, 0);
            end = std::chrono::high_resolution_clock::now();
            upload_time += std::chrono::duration<double, std::micro>(end - start).count();

            cudaGraphExecDestroy(deviceExec);
            cudaGraphDestroy(graph);
        }

        instantiate_regular_time /= MEASURE_RUNS;
        instantiate_device_time /= MEASURE_RUNS;
        upload_time /= MEASURE_RUNS;

        printf("%d\t%.2f\t\t\t%.2f\t\t\t%.2f\t\t%.2f\n",
               num_kernels, instantiate_regular_time, instantiate_device_time,
               upload_time, instantiate_device_time + upload_time);
    }
    printf("\n");
}

//=============================================================================
// Test 2: Upload multiple graphs sequentially - measure scaling
//=============================================================================
void Test_MultipleGraphUpload(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 2: Multiple Graph Upload Scaling ===\n");
    printf("Upload N different graphs sequentially, measure total time\n\n");

    int num_graphs_list[] = {1, 10, 50, 100};
    int num_kernels = 16;  // Fixed graph size

    printf("num_graphs\ttotal_upload(us)\tper_graph_upload(us)\n");

    for (int g = 0; g < 4; g++) {
        int num_graphs = num_graphs_list[g];

        // Pre-build all graphs and instantiate
        std::vector<cudaGraph_t> graphs(num_graphs);
        std::vector<cudaGraphExec_t> execs(num_graphs);

        for (int i = 0; i < num_graphs; i++) {
            build_graph_for_upload(&graphs[i], num_kernels, sleep_kernel_5, NULL, blocks, threads);
            cudaGraphInstantiate(&execs[i], graphs[i], cudaGraphInstantiateFlagDeviceLaunch);
        }

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; w++) {
            for (int i = 0; i < num_graphs; i++) {
                cudaGraphUpload(execs[i], 0);
            }
        }

        // Measure upload time
        double total_time = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_graphs; i++) {
                cudaGraphUpload(execs[i], 0);
            }
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::micro>(end - start).count();
        }
        total_time /= MEASURE_RUNS;

        printf("%d\t\t%.2f\t\t\t%.2f\n",
               num_graphs, total_time, total_time / num_graphs);

        // Cleanup
        for (int i = 0; i < num_graphs; i++) {
            cudaGraphExecDestroy(execs[i]);
            cudaGraphDestroy(graphs[i]);
        }
    }
    printf("\n");
}

//=============================================================================
// Memory-bound streaming kernel (simple copy)
//=============================================================================
__global__ void stream_kernel(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

//=============================================================================
// Test 3: Contention test - does upload affect kernel performance?
// Use CUDA events to measure kernel execution time precisely
//=============================================================================
void Test_UploadContention_cuBLAS(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 3a: Upload Contention - cuBLAS SGEMM (Compute-bound) ===\n");
    printf("Measure cuBLAS kernel time and upload time with separate CUDA events\n\n");

    // Get theoretical peak FLOPS
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // FP32 peak = SM count * cores per SM * 2 (FMA) * clock rate
    // For modern GPUs, use cudaDevAttrMaxClockRate
    int clock_khz;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    // Estimate FP32 cores per SM (varies by arch, use 128 as typical for Ampere+)
    int cores_per_sm = 128;
    double peak_tflops = (double)prop.multiProcessorCount * cores_per_sm * 2.0 * clock_khz / 1e9;
    printf("GPU: %s, SMs: %d, Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, clock_khz/1000.0);
    printf("Estimated FP32 peak: %.2f TFLOPS\n\n", peak_tflops);

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t compute_stream, upload_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&upload_stream);
    cublasSetStream(handle, compute_stream);

    // Allocate matrices for SGEMM (make it long enough ~1-2ms)
    int M = 4096, N = 4096, K = 4096;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_A, M * K);
    curandGenerateUniform(gen, d_B, K * N);
    curandDestroyGenerator(gen);

    float alpha = 1.0f, beta = 0.0f;

    // FLOPS for SGEMM: 2*M*N*K (multiply-add)
    double flops = 2.0 * M * N * K;

    // CUDA events for timing - separate events for each stream
    cudaEvent_t compute_start, compute_stop;
    cudaEvent_t upload_start, upload_stop;
    cudaEventCreate(&compute_start);
    cudaEventCreate(&compute_stop);
    cudaEventCreate(&upload_start);
    cudaEventCreate(&upload_stop);

    // Measure baseline (no upload)
    float baseline_ms = 0;
    for (int w = 0; w < WARMUP_RUNS; w++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A, M, d_B, K, &beta, d_C, M);
        cudaStreamSynchronize(compute_stream);
    }
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaEventRecord(compute_start, compute_stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A, M, d_B, K, &beta, d_C, M);
        cudaEventRecord(compute_stop, compute_stream);
        cudaEventSynchronize(compute_stop);
        float ms;
        cudaEventElapsedTime(&ms, compute_start, compute_stop);
        baseline_ms += ms;
    }
    baseline_ms /= MEASURE_RUNS;
    double baseline_tflops = flops / baseline_ms / 1e9;
    printf("Baseline cuBLAS SGEMM: %.3f ms, %.2f TFLOPS (%.1f%% of peak)\n\n",
           baseline_ms, baseline_tflops, baseline_tflops / peak_tflops * 100);

    // Test with concurrent uploads (10x more to see overlap clearly)
    int num_uploads_list[] = {100, 500, 1000, 2000};
    int num_kernels = 16;

    printf("num_uploads\tkernel(ms)\tupload(ms)\tkernel_TFLOPS\tslowdown(%%)\n");

    for (int u = 0; u < 4; u++) {
        int num_uploads = num_uploads_list[u];

        // Pre-build graphs
        std::vector<cudaGraph_t> graphs(num_uploads);
        std::vector<cudaGraphExec_t> execs(num_uploads);
        for (int i = 0; i < num_uploads; i++) {
            build_graph_for_upload(&graphs[i], num_kernels, sleep_kernel_5, NULL, blocks, threads);
            cudaGraphInstantiate(&execs[i], graphs[i], cudaGraphInstantiateFlagDeviceLaunch);
        }

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; w++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                        &alpha, d_A, M, d_B, K, &beta, d_C, M);
            for (int i = 0; i < num_uploads; i++) {
                cudaGraphUpload(execs[i], upload_stream);
            }
            cudaDeviceSynchronize();
        }

        // Measure with concurrent uploads using separate events
        float kernel_ms = 0;
        float upload_ms = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            // Time kernel on compute_stream
            cudaEventRecord(compute_start, compute_stream);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                        &alpha, d_A, M, d_B, K, &beta, d_C, M);
            cudaEventRecord(compute_stop, compute_stream);

            // Time uploads on upload_stream
            cudaEventRecord(upload_start, upload_stream);
            for (int i = 0; i < num_uploads; i++) {
                cudaGraphUpload(execs[i], upload_stream);
            }
            cudaEventRecord(upload_stop, upload_stream);

            // Wait for both to complete
            cudaEventSynchronize(compute_stop);
            cudaEventSynchronize(upload_stop);

            float k_ms, u_ms;
            cudaEventElapsedTime(&k_ms, compute_start, compute_stop);
            cudaEventElapsedTime(&u_ms, upload_start, upload_stop);
            kernel_ms += k_ms;
            upload_ms += u_ms;
        }
        kernel_ms /= MEASURE_RUNS;
        upload_ms /= MEASURE_RUNS;

        double tflops = flops / kernel_ms / 1e9;
        double slowdown = (kernel_ms - baseline_ms) / baseline_ms * 100;
        printf("%d\t\t%.3f\t\t%.3f\t\t%.2f\t\t%.2f\n",
               num_uploads, kernel_ms, upload_ms, tflops, slowdown);

        // Cleanup
        for (int i = 0; i < num_uploads; i++) {
            cudaGraphExecDestroy(execs[i]);
            cudaGraphDestroy(graphs[i]);
        }
    }

    cudaEventDestroy(compute_start);
    cudaEventDestroy(compute_stop);
    cudaEventDestroy(upload_start);
    cudaEventDestroy(upload_stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(upload_stream);
    cublasDestroy(handle);
    printf("\n");
}

void Test_UploadContention_MemoryBound(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 3b: Upload Contention - Stream Copy (Memory-bound) ===\n");
    printf("Measure memory copy kernel time and upload time with separate CUDA events\n\n");

    // Get theoretical peak memory bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mem_clock_khz, mem_bus_width;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_bw_gbs = (double)mem_clock_khz * 2 * (mem_bus_width / 8) / 1e6;
    printf("GPU: %s, Memory Clock: %.0f MHz, Bus Width: %d-bit\n",
           prop.name, mem_clock_khz/1000.0, mem_bus_width);
    printf("Theoretical peak bandwidth: %.2f GB/s\n\n", peak_bw_gbs);

    cudaStream_t compute_stream, upload_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&upload_stream);

    // Allocate large arrays for memory-bound kernel (~1-2ms)
    size_t n = 256 * 1024 * 1024;  // 256M floats = 1GB
    float *d_src, *d_dst;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, n * sizeof(float));

    // Initialize
    cudaMemset(d_src, 1, n * sizeof(float));

    int grid_size = (n + 255) / 256;

    // Bytes transferred: read src + write dst
    double bytes_transferred = 2.0 * n * sizeof(float);

    // CUDA events for timing - separate events for each stream
    cudaEvent_t compute_start, compute_stop;
    cudaEvent_t upload_start, upload_stop;
    cudaEventCreate(&compute_start);
    cudaEventCreate(&compute_stop);
    cudaEventCreate(&upload_start);
    cudaEventCreate(&upload_stop);

    // Measure baseline (no upload)
    float baseline_ms = 0;
    for (int w = 0; w < WARMUP_RUNS; w++) {
        stream_kernel<<<grid_size, 256, 0, compute_stream>>>(d_dst, d_src, n);
        cudaStreamSynchronize(compute_stream);
    }
    for (int r = 0; r < MEASURE_RUNS; r++) {
        cudaEventRecord(compute_start, compute_stream);
        stream_kernel<<<grid_size, 256, 0, compute_stream>>>(d_dst, d_src, n);
        cudaEventRecord(compute_stop, compute_stream);
        cudaEventSynchronize(compute_stop);
        float ms;
        cudaEventElapsedTime(&ms, compute_start, compute_stop);
        baseline_ms += ms;
    }
    baseline_ms /= MEASURE_RUNS;
    double baseline_bw = bytes_transferred / baseline_ms / 1e6;
    printf("Baseline stream copy: %.3f ms, %.2f GB/s (%.1f%% of peak)\n\n",
           baseline_ms, baseline_bw, baseline_bw / peak_bw_gbs * 100);

    // Test with concurrent uploads (10x more to see overlap clearly)
    int num_uploads_list[] = {100, 500, 1000, 2000};
    int num_kernels = 16;

    printf("num_uploads\tkernel(ms)\tupload(ms)\tbandwidth(GB/s)\tslowdown(%%)\n");

    for (int u = 0; u < 4; u++) {
        int num_uploads = num_uploads_list[u];

        // Pre-build graphs
        std::vector<cudaGraph_t> graphs(num_uploads);
        std::vector<cudaGraphExec_t> execs(num_uploads);
        for (int i = 0; i < num_uploads; i++) {
            build_graph_for_upload(&graphs[i], num_kernels, sleep_kernel_5, NULL, blocks, threads);
            cudaGraphInstantiate(&execs[i], graphs[i], cudaGraphInstantiateFlagDeviceLaunch);
        }

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; w++) {
            stream_kernel<<<grid_size, 256, 0, compute_stream>>>(d_dst, d_src, n);
            for (int i = 0; i < num_uploads; i++) {
                cudaGraphUpload(execs[i], upload_stream);
            }
            cudaDeviceSynchronize();
        }

        // Measure with concurrent uploads using separate events
        float kernel_ms = 0;
        float upload_ms = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            // Time kernel on compute_stream
            cudaEventRecord(compute_start, compute_stream);
            stream_kernel<<<grid_size, 256, 0, compute_stream>>>(d_dst, d_src, n);
            cudaEventRecord(compute_stop, compute_stream);

            // Time uploads on upload_stream
            cudaEventRecord(upload_start, upload_stream);
            for (int i = 0; i < num_uploads; i++) {
                cudaGraphUpload(execs[i], upload_stream);
            }
            cudaEventRecord(upload_stop, upload_stream);

            // Wait for both to complete
            cudaEventSynchronize(compute_stop);
            cudaEventSynchronize(upload_stop);

            float k_ms, u_ms;
            cudaEventElapsedTime(&k_ms, compute_start, compute_stop);
            cudaEventElapsedTime(&u_ms, upload_start, upload_stop);
            kernel_ms += k_ms;
            upload_ms += u_ms;
        }
        kernel_ms /= MEASURE_RUNS;
        upload_ms /= MEASURE_RUNS;

        double bandwidth = bytes_transferred / kernel_ms / 1e6;
        double slowdown = (kernel_ms - baseline_ms) / baseline_ms * 100;
        printf("%d\t\t%.3f\t\t%.3f\t\t%.2f\t\t%.2f\n",
               num_uploads, kernel_ms, upload_ms, bandwidth, slowdown);

        // Cleanup
        for (int i = 0; i < num_uploads; i++) {
            cudaGraphExecDestroy(execs[i]);
            cudaGraphDestroy(graphs[i]);
        }
    }

    cudaEventDestroy(compute_start);
    cudaEventDestroy(compute_stop);
    cudaEventDestroy(upload_start);
    cudaEventDestroy(upload_stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(upload_stream);
    printf("\n");
}

//=============================================================================
// Test 4: Re-upload overhead - is re-uploading same graph faster?
//=============================================================================
void Test_ReuploadOverhead(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 4: Re-upload Overhead ===\n");
    printf("Compare first upload vs subsequent re-uploads of same graph\n\n");

    int num_kernels_list[] = {1, 16, 64, 256};

    printf("kernels\tfirst_upload(us)\treupload_avg(us)\tratio\n");

    for (int k = 0; k < 4; k++) {
        int num_kernels = num_kernels_list[k];

        cudaGraph_t graph;
        cudaGraphExec_t exec;
        build_graph_for_upload(&graph, num_kernels, sleep_kernel_5, NULL, blocks, threads);
        cudaGraphInstantiate(&exec, graph, cudaGraphInstantiateFlagDeviceLaunch);

        // Measure first upload (cold)
        double first_upload = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            // Destroy and recreate to get cold upload
            cudaGraphExecDestroy(exec);
            cudaGraphInstantiate(&exec, graph, cudaGraphInstantiateFlagDeviceLaunch);

            auto start = std::chrono::high_resolution_clock::now();
            cudaGraphUpload(exec, 0);
            auto end = std::chrono::high_resolution_clock::now();
            first_upload += std::chrono::duration<double, std::micro>(end - start).count();
        }
        first_upload /= MEASURE_RUNS;

        // Measure re-upload (warm)
        cudaGraphUpload(exec, 0);  // Initial upload
        double reupload = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            cudaGraphUpload(exec, 0);
            auto end = std::chrono::high_resolution_clock::now();
            reupload += std::chrono::duration<double, std::micro>(end - start).count();
        }
        reupload /= MEASURE_RUNS;

        printf("%d\t%.2f\t\t\t%.2f\t\t\t%.2fx\n",
               num_kernels, first_upload, reupload, first_upload / reupload);

        cudaGraphExecDestroy(exec);
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

    printf("CUDA Graph Upload Overhead Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("CUDA Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("=======================================================================\n\n");

    if (deviceProp.major < 7) {
        printf("ERROR: Device graph launch requires compute capability 7.0+\n");
        return 1;
    }

    Test_InstantiateUploadAPIOverhead(smx_count, 1024);
    Test_MultipleGraphUpload(smx_count, 1024);
    Test_UploadContention_cuBLAS(smx_count, 1024);
    Test_UploadContention_MemoryBound(smx_count, 1024);
    Test_ReuploadOverhead(smx_count, 1024);

    return 0;
}
