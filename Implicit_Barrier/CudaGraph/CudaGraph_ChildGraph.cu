// Child Graph benchmark: measure overhead of CUDA Graph child graph methods
// Three methods: flat (N kernels), hierarchy (g2=g1+g1), iterative (clone+insert)
// Tests: correctness, construction overhead, runtime overhead

#include "CudaGraph_Kernel.cuh"
#include "../../share/util.h"
#include <stdio.h>
#include <chrono>
#include <cassert>

#define WARMUP_RUNS 5
#define MEASURE_RUNS 100
#define CONSTRUCT_RUNS 10

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
// Construction overhead breakdown structure
//=============================================================================
struct ConstructionTimes {
    double total;
    double build;       // Time to add kernel nodes
    double clone;       // Time for cudaGraphClone (iterative only)
    double insert;      // Time for cudaGraphAddChildGraphNode
    double instantiate; // Time for cudaGraphInstantiate
};

//=============================================================================
// Method 1: FLAT - N kernels in sequence
//=============================================================================
void build_flat(cudaGraph_t* graph, cudaGraphExec_t* graphExec,
                int num_kernels, KernelFunc kernel, void** kernel_args,
                unsigned int blocks, unsigned int threads,
                ConstructionTimes* times)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    // Build phase: create graph and add kernel nodes
    auto build_start = std::chrono::high_resolution_clock::now();
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
    auto build_end = std::chrono::high_resolution_clock::now();

    // Instantiate phase
    auto inst_start = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(graphExec, *graph, 0);
    auto inst_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    if (times) {
        times->build = std::chrono::duration<double, std::nano>(build_end - build_start).count();
        times->clone = 0;
        times->insert = 0;
        times->instantiate = std::chrono::duration<double, std::nano>(inst_end - inst_start).count();
        times->total = std::chrono::duration<double, std::nano>(total_end - total_start).count();
    }
}

//=============================================================================
// Method 2: HIERARCHY - g1=1 kernel, g2=g1+g1, g4=g2+g2, ...
// Keeps all intermediate graphs
//=============================================================================
void build_hierarchy(cudaGraph_t* graphs, int num_levels, cudaGraphExec_t* graphExec,
                     KernelFunc kernel, void** kernel_args,
                     unsigned int blocks, unsigned int threads,
                     ConstructionTimes* times)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    double total_build = 0;
    double total_insert = 0;

    // Level 0: single kernel graph
    auto build_start = std::chrono::high_resolution_clock::now();
    cudaGraphCreate(&graphs[0], 0);
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;
    cudaGraphAddKernelNode(&kernelNode, graphs[0], NULL, 0, &kernelParams);
    auto build_end = std::chrono::high_resolution_clock::now();
    total_build = std::chrono::duration<double, std::nano>(build_end - build_start).count();

    // Build hierarchy: g[i] = g[i-1] + g[i-1]
    for (int level = 1; level < num_levels; level++) {
        cudaGraphCreate(&graphs[level], 0);

        auto insert_start = std::chrono::high_resolution_clock::now();
        cudaGraphNode_t childNode1, childNode2;
        cudaGraphAddChildGraphNode(&childNode1, graphs[level], NULL, 0, graphs[level - 1]);
        cudaGraphAddChildGraphNode(&childNode2, graphs[level], &childNode1, 1, graphs[level - 1]);
        auto insert_end = std::chrono::high_resolution_clock::now();
        total_insert += std::chrono::duration<double, std::nano>(insert_end - insert_start).count();
    }

    // Instantiate final graph
    auto inst_start = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(graphExec, graphs[num_levels - 1], 0);
    auto inst_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    if (times) {
        times->build = total_build;
        times->clone = 0;
        times->insert = total_insert;
        times->instantiate = std::chrono::duration<double, std::nano>(inst_end - inst_start).count();
        times->total = std::chrono::duration<double, std::nano>(total_end - total_start).count();
    }
}

//=============================================================================
// Method 3: ITERATIVE - clone current graph, insert as child, destroy old
// Only keeps 2 graphs at a time
//=============================================================================
void build_iterative(cudaGraph_t* outGraph, cudaGraphExec_t* graphExec,
                     int num_kernels, KernelFunc kernel, void** kernel_args,
                     unsigned int blocks, unsigned int threads,
                     ConstructionTimes* times)
{
    assert(is_power_of_2(num_kernels));
    int num_levels = log2_int(num_kernels);

    auto total_start = std::chrono::high_resolution_clock::now();

    double total_build = 0;
    double total_clone = 0;
    double total_insert = 0;

    // Start with single kernel graph
    auto build_start = std::chrono::high_resolution_clock::now();
    cudaGraph_t current;
    cudaGraphCreate(&current, 0);
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;
    cudaGraphAddKernelNode(&kernelNode, current, NULL, 0, &kernelParams);
    auto build_end = std::chrono::high_resolution_clock::now();
    total_build = std::chrono::duration<double, std::nano>(build_end - build_start).count();

    // Double iteratively: clone current, create new graph with 2 child nodes
    for (int i = 0; i < num_levels; i++) {
        // Clone
        auto clone_start = std::chrono::high_resolution_clock::now();
        cudaGraph_t clone;
        cudaGraphClone(&clone, current);
        auto clone_end = std::chrono::high_resolution_clock::now();
        total_clone += std::chrono::duration<double, std::nano>(clone_end - clone_start).count();

        // Create new graph and insert
        cudaGraph_t next;
        cudaGraphCreate(&next, 0);

        auto insert_start = std::chrono::high_resolution_clock::now();
        cudaGraphNode_t childNode1, childNode2;
        cudaGraphAddChildGraphNode(&childNode1, next, NULL, 0, clone);
        cudaGraphAddChildGraphNode(&childNode2, next, &childNode1, 1, clone);
        auto insert_end = std::chrono::high_resolution_clock::now();
        total_insert += std::chrono::duration<double, std::nano>(insert_end - insert_start).count();

        // Destroy old, keep next
        cudaGraphDestroy(current);
        cudaGraphDestroy(clone);
        current = next;
    }

    *outGraph = current;

    // Instantiate
    auto inst_start = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(graphExec, current, 0);
    auto inst_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    if (times) {
        times->build = total_build;
        times->clone = total_clone;
        times->insert = total_insert;
        times->instantiate = std::chrono::duration<double, std::nano>(inst_end - inst_start).count();
        times->total = std::chrono::duration<double, std::nano>(total_end - total_start).count();
    }
}

//=============================================================================
// Method 4: ITERATIVE_CHAIN - flat_64 + child(flat_32 + child(flat_16 + ...))
// Chain of decreasing flat portions, no cloning needed
//=============================================================================
void build_iterative_chain(cudaGraph_t* outGraph, cudaGraphExec_t* graphExec,
                           int num_kernels, KernelFunc kernel, void** kernel_args,
                           unsigned int blocks, unsigned int threads,
                           ConstructionTimes* times)
{
    assert(is_power_of_2(num_kernels));
    int num_levels = log2_int(num_kernels) + 1;  // e.g., 128 -> 8 levels (1,2,4,8,16,32,64,128)

    auto total_start = std::chrono::high_resolution_clock::now();

    double total_build = 0;
    double total_insert = 0;

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;

    // Level 0: single kernel (flat_1)
    cudaGraph_t current;
    auto build_start = std::chrono::high_resolution_clock::now();
    cudaGraphCreate(&current, 0);
    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, current, NULL, 0, &kernelParams);
    auto build_end = std::chrono::high_resolution_clock::now();
    total_build += std::chrono::duration<double, std::nano>(build_end - build_start).count();

    // Build chain: each level has flat_N + child(previous level)
    // Level 1: flat_1 + child(level0) = 2 kernels
    // Level 2: flat_2 + child(level1) = 4 kernels
    // Level k: flat_(2^(k-1)) + child(level(k-1)) = 2^k kernels
    int flat_size = 1;  // starts at 1 for level 1
    for (int level = 1; level < num_levels; level++) {
        cudaGraph_t next;
        cudaGraphCreate(&next, 0);

        // Build flat portion: flat_size kernel nodes
        build_start = std::chrono::high_resolution_clock::now();
        cudaGraphNode_t prevNode = NULL;
        for (int i = 0; i < flat_size; i++) {
            cudaGraphNode_t node;
            if (prevNode) {
                cudaGraphAddKernelNode(&node, next, &prevNode, 1, &kernelParams);
            } else {
                cudaGraphAddKernelNode(&node, next, NULL, 0, &kernelParams);
            }
            prevNode = node;
        }
        build_end = std::chrono::high_resolution_clock::now();
        total_build += std::chrono::duration<double, std::nano>(build_end - build_start).count();

        // Insert previous level as child (depends on last kernel in flat portion)
        auto insert_start = std::chrono::high_resolution_clock::now();
        cudaGraphNode_t childNode;
        cudaGraphAddChildGraphNode(&childNode, next, &prevNode, 1, current);
        auto insert_end = std::chrono::high_resolution_clock::now();
        total_insert += std::chrono::duration<double, std::nano>(insert_end - insert_start).count();

        // Move to next level
        cudaGraphDestroy(current);
        current = next;
        flat_size *= 2;  // double for next level
    }

    *outGraph = current;

    // Instantiate
    auto inst_start = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(graphExec, current, 0);
    auto inst_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    if (times) {
        times->build = total_build;
        times->clone = 0;
        times->insert = total_insert;
        times->instantiate = std::chrono::duration<double, std::nano>(inst_end - inst_start).count();
        times->total = std::chrono::duration<double, std::nano>(total_end - total_start).count();
    }
}

//=============================================================================
// Method 5: MERGED_FLAT - flat_N/2 + child(flat_N/2)
// Two equal flat graphs combined with one child node
//=============================================================================
void build_merged_flat(cudaGraph_t* graphs, cudaGraphExec_t* graphExec,
                       int num_kernels, KernelFunc kernel, void** kernel_args,
                       unsigned int blocks, unsigned int threads,
                       ConstructionTimes* times)
{
    assert(num_kernels >= 2 && num_kernels % 2 == 0);
    int half = num_kernels / 2;

    auto total_start = std::chrono::high_resolution_clock::now();

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)kernel;
    kernelParams.gridDim = dim3(blocks);
    kernelParams.blockDim = dim3(threads);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernel_args;
    kernelParams.extra = NULL;

    // Build first flat graph (graphs[0]): half kernels
    auto build_start = std::chrono::high_resolution_clock::now();
    cudaGraphCreate(&graphs[0], 0);
    cudaGraphNode_t prevNode = NULL;
    for (int i = 0; i < half; i++) {
        cudaGraphNode_t node;
        if (prevNode) {
            cudaGraphAddKernelNode(&node, graphs[0], &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&node, graphs[0], NULL, 0, &kernelParams);
        }
        prevNode = node;
    }

    // Build second flat graph (graphs[1]): half kernels
    cudaGraphCreate(&graphs[1], 0);
    prevNode = NULL;
    for (int i = 0; i < half; i++) {
        cudaGraphNode_t node;
        if (prevNode) {
            cudaGraphAddKernelNode(&node, graphs[1], &prevNode, 1, &kernelParams);
        } else {
            cudaGraphAddKernelNode(&node, graphs[1], NULL, 0, &kernelParams);
        }
        prevNode = node;
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double, std::nano>(build_end - build_start).count();

    // Insert graphs[0] as child of graphs[1] (after the last kernel)
    auto insert_start = std::chrono::high_resolution_clock::now();
    cudaGraphNode_t childNode;
    cudaGraphAddChildGraphNode(&childNode, graphs[1], &prevNode, 1, graphs[0]);
    auto insert_end = std::chrono::high_resolution_clock::now();
    double insert_time = std::chrono::duration<double, std::nano>(insert_end - insert_start).count();

    // Instantiate
    auto inst_start = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(graphExec, graphs[1], 0);
    auto inst_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    if (times) {
        times->build = build_time;
        times->clone = 0;
        times->insert = insert_time;
        times->instantiate = std::chrono::duration<double, std::nano>(inst_end - inst_start).count();
        times->total = std::chrono::duration<double, std::nano>(total_end - total_start).count();
    }
}

//=============================================================================
// Cleanup helpers
//=============================================================================
void cleanup_flat(cudaGraph_t graph, cudaGraphExec_t graphExec) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

void cleanup_hierarchy(cudaGraph_t* graphs, int num_levels, cudaGraphExec_t graphExec) {
    cudaGraphExecDestroy(graphExec);
    for (int i = 0; i < num_levels; i++) {
        cudaGraphDestroy(graphs[i]);
    }
}

void cleanup_iterative(cudaGraph_t graph, cudaGraphExec_t graphExec) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

void cleanup_iterative_chain(cudaGraph_t graph, cudaGraphExec_t graphExec) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

void cleanup_merged_flat(cudaGraph_t* graphs, cudaGraphExec_t graphExec) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graphs[0]);
    cudaGraphDestroy(graphs[1]);
}

//=============================================================================
// Test 1: CORRECTNESS - verify kernel execution count
//=============================================================================
void Test_Correctness(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 1: Correctness Verification ===\n");
    printf("Verify that all methods execute the correct number of kernels\n\n");

    int test_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("method\t\texpected\tactual\t\tstatus\n");

    bool all_passed = true;

    for (int t = 0; t < num_tests; t++) {
        int num_kernels = test_sizes[t];
        int num_levels = log2_int(num_kernels) + 1;

        // Allocate counter
        int* d_counter;
        cudaMalloc(&d_counter, sizeof(int));
        void* kernel_args[] = { &d_counter };

        // Test FLAT
        {
            cudaMemset(d_counter, 0, sizeof(int));
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_flat(&graph, &graphExec, num_kernels, (KernelFunc)sleep_kernel_count_5, kernel_args, blocks, threads, NULL);
            cudaGraphLaunch(graphExec, 0);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            bool passed = (count == num_kernels);
            printf("flat\t\t%d\t\t%d\t\t%s\n", num_kernels, count, passed ? "PASS" : "FAIL");
            if (!passed) all_passed = false;
            cleanup_flat(graph, graphExec);
        }

        // Test HIERARCHY
        {
            cudaMemset(d_counter, 0, sizeof(int));
            cudaGraph_t* graphs = new cudaGraph_t[num_levels];
            cudaGraphExec_t graphExec;
            build_hierarchy(graphs, num_levels, &graphExec, (KernelFunc)sleep_kernel_count_5, kernel_args, blocks, threads, NULL);
            cudaGraphLaunch(graphExec, 0);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            bool passed = (count == num_kernels);
            printf("hierarchy\t%d\t\t%d\t\t%s\n", num_kernels, count, passed ? "PASS" : "FAIL");
            if (!passed) all_passed = false;
            cleanup_hierarchy(graphs, num_levels, graphExec);
            delete[] graphs;
        }

        // Test ITERATIVE (clone+insert)
        {
            cudaMemset(d_counter, 0, sizeof(int));
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_iterative(&graph, &graphExec, num_kernels, (KernelFunc)sleep_kernel_count_5, kernel_args, blocks, threads, NULL);
            cudaGraphLaunch(graphExec, 0);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            bool passed = (count == num_kernels);
            printf("iter_clone\t%d\t\t%d\t\t%s\n", num_kernels, count, passed ? "PASS" : "FAIL");
            if (!passed) all_passed = false;
            cleanup_iterative(graph, graphExec);
        }

        // Test ITERATIVE_CHAIN (flat+insert)
        {
            cudaMemset(d_counter, 0, sizeof(int));
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_iterative_chain(&graph, &graphExec, num_kernels, (KernelFunc)sleep_kernel_count_5, kernel_args, blocks, threads, NULL);
            cudaGraphLaunch(graphExec, 0);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            bool passed = (count == num_kernels);
            printf("iter_chain\t%d\t\t%d\t\t%s\n", num_kernels, count, passed ? "PASS" : "FAIL");
            if (!passed) all_passed = false;
            cleanup_iterative_chain(graph, graphExec);
        }

        // Test MERGED_FLAT (only for num_kernels >= 2)
        if (num_kernels >= 2) {
            cudaMemset(d_counter, 0, sizeof(int));
            cudaGraph_t graphs[2];
            cudaGraphExec_t graphExec;
            build_merged_flat(graphs, &graphExec, num_kernels, (KernelFunc)sleep_kernel_count_5, kernel_args, blocks, threads, NULL);
            cudaGraphLaunch(graphExec, 0);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            bool passed = (count == num_kernels);
            printf("merged_flat\t%d\t\t%d\t\t%s\n", num_kernels, count, passed ? "PASS" : "FAIL");
            if (!passed) all_passed = false;
            cleanup_merged_flat(graphs, graphExec);
        }

        printf("\n");
        cudaFree(d_counter);
    }

    printf("Overall: %s\n\n", all_passed ? "ALL PASSED" : "SOME FAILED");
}

//=============================================================================
// Test 2: CONSTRUCTION OVERHEAD - measure build, clone, insert, instantiate
//=============================================================================
void Test_Construction(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 2: Construction Overhead ===\n");
    printf("Breakdown: build (add kernel nodes), clone, insert (add child nodes), instantiate\n\n");

    int test_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("method\t\tkernels\ttotal(ns)\tbuild(ns)\tclone(ns)\tinsert(ns)\tinstantiate(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int num_kernels = test_sizes[t];
        int num_levels = log2_int(num_kernels) + 1;

        ConstructionTimes flat_times = {0}, hier_times = {0}, iter_times = {0};
        ConstructionTimes chain_times = {0}, merged_times = {0};

        // Measure FLAT (average over multiple runs)
        for (int r = 0; r < CONSTRUCT_RUNS; r++) {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            ConstructionTimes t;
            build_flat(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, &t);
            flat_times.total += t.total;
            flat_times.build += t.build;
            flat_times.instantiate += t.instantiate;
            cleanup_flat(graph, graphExec);
        }
        flat_times.total /= CONSTRUCT_RUNS;
        flat_times.build /= CONSTRUCT_RUNS;
        flat_times.instantiate /= CONSTRUCT_RUNS;

        // Measure HIERARCHY
        for (int r = 0; r < CONSTRUCT_RUNS; r++) {
            cudaGraph_t* graphs = new cudaGraph_t[num_levels];
            cudaGraphExec_t graphExec;
            ConstructionTimes t;
            build_hierarchy(graphs, num_levels, &graphExec, sleep_kernel_5, NULL, blocks, threads, &t);
            hier_times.total += t.total;
            hier_times.build += t.build;
            hier_times.insert += t.insert;
            hier_times.instantiate += t.instantiate;
            cleanup_hierarchy(graphs, num_levels, graphExec);
            delete[] graphs;
        }
        hier_times.total /= CONSTRUCT_RUNS;
        hier_times.build /= CONSTRUCT_RUNS;
        hier_times.insert /= CONSTRUCT_RUNS;
        hier_times.instantiate /= CONSTRUCT_RUNS;

        // Measure ITERATIVE (clone+insert)
        for (int r = 0; r < CONSTRUCT_RUNS; r++) {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            ConstructionTimes t;
            build_iterative(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, &t);
            iter_times.total += t.total;
            iter_times.build += t.build;
            iter_times.clone += t.clone;
            iter_times.insert += t.insert;
            iter_times.instantiate += t.instantiate;
            cleanup_iterative(graph, graphExec);
        }
        iter_times.total /= CONSTRUCT_RUNS;
        iter_times.build /= CONSTRUCT_RUNS;
        iter_times.clone /= CONSTRUCT_RUNS;
        iter_times.insert /= CONSTRUCT_RUNS;
        iter_times.instantiate /= CONSTRUCT_RUNS;

        // Measure ITERATIVE_CHAIN (flat+insert)
        for (int r = 0; r < CONSTRUCT_RUNS; r++) {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            ConstructionTimes t;
            build_iterative_chain(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, &t);
            chain_times.total += t.total;
            chain_times.build += t.build;
            chain_times.insert += t.insert;
            chain_times.instantiate += t.instantiate;
            cleanup_iterative_chain(graph, graphExec);
        }
        chain_times.total /= CONSTRUCT_RUNS;
        chain_times.build /= CONSTRUCT_RUNS;
        chain_times.insert /= CONSTRUCT_RUNS;
        chain_times.instantiate /= CONSTRUCT_RUNS;

        // Measure MERGED_FLAT (only for num_kernels >= 2)
        if (num_kernels >= 2) {
            for (int r = 0; r < CONSTRUCT_RUNS; r++) {
                cudaGraph_t graphs[2];
                cudaGraphExec_t graphExec;
                ConstructionTimes t;
                build_merged_flat(graphs, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, &t);
                merged_times.total += t.total;
                merged_times.build += t.build;
                merged_times.insert += t.insert;
                merged_times.instantiate += t.instantiate;
                cleanup_merged_flat(graphs, graphExec);
            }
            merged_times.total /= CONSTRUCT_RUNS;
            merged_times.build /= CONSTRUCT_RUNS;
            merged_times.insert /= CONSTRUCT_RUNS;
            merged_times.instantiate /= CONSTRUCT_RUNS;
        }

        // Print results
        printf("flat\t\t%d\t%.0f\t\t%.0f\t\t-\t\t-\t\t%.0f\n",
               num_kernels, flat_times.total, flat_times.build, flat_times.instantiate);
        printf("hierarchy\t%d\t%.0f\t\t%.0f\t\t-\t\t%.0f\t\t%.0f\n",
               num_kernels, hier_times.total, hier_times.build, hier_times.insert, hier_times.instantiate);
        printf("iter_clone\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.0f\n",
               num_kernels, iter_times.total, iter_times.build, iter_times.clone, iter_times.insert, iter_times.instantiate);
        printf("iter_chain\t%d\t%.0f\t\t%.0f\t\t-\t\t%.0f\t\t%.0f\n",
               num_kernels, chain_times.total, chain_times.build, chain_times.insert, chain_times.instantiate);
        if (num_kernels >= 2) {
            printf("merged_flat\t%d\t%.0f\t\t%.0f\t\t-\t\t%.0f\t\t%.0f\n",
                   num_kernels, merged_times.total, merged_times.build, merged_times.insert, merged_times.instantiate);
        }
        printf("\n");
    }
}

//=============================================================================
// Test 3: RUNTIME OVERHEAD - measure execution time and per-kernel overhead
//=============================================================================
void Test_Runtime(unsigned int blocks, unsigned int threads)
{
    printf("=== Test 3: Runtime Overhead ===\n");
    printf("Using dual-workload method: overhead = 2 * time_5us - time_10us\n\n");

    int test_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("method\t\tkernels\ttime_5us(ns)\ttime_10us(ns)\toverhead(ns)\tper_kernel(ns)\n");

    for (int t = 0; t < num_tests; t++) {
        int num_kernels = test_sizes[t];
        int num_levels = log2_int(num_kernels) + 1;

        // Measure FLAT
        double flat_5 = 0, flat_10 = 0;
        {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_flat(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                flat_5 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            flat_5 /= MEASURE_RUNS;
            cleanup_flat(graph, graphExec);

            build_flat(&graph, &graphExec, num_kernels, sleep_kernel_10, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                flat_10 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            flat_10 /= MEASURE_RUNS;
            cleanup_flat(graph, graphExec);
        }
        double flat_overhead = 2 * flat_5 - flat_10;
        printf("flat\t\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\n",
               num_kernels, flat_5, flat_10, flat_overhead, flat_overhead / num_kernels);

        // Measure HIERARCHY
        double hier_5 = 0, hier_10 = 0;
        {
            cudaGraph_t* graphs = new cudaGraph_t[num_levels];
            cudaGraphExec_t graphExec;
            build_hierarchy(graphs, num_levels, &graphExec, sleep_kernel_5, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                hier_5 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            hier_5 /= MEASURE_RUNS;
            cleanup_hierarchy(graphs, num_levels, graphExec);
            delete[] graphs;

            graphs = new cudaGraph_t[num_levels];
            build_hierarchy(graphs, num_levels, &graphExec, sleep_kernel_10, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                hier_10 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            hier_10 /= MEASURE_RUNS;
            cleanup_hierarchy(graphs, num_levels, graphExec);
            delete[] graphs;
        }
        double hier_overhead = 2 * hier_5 - hier_10;
        printf("hierarchy\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\n",
               num_kernels, hier_5, hier_10, hier_overhead, hier_overhead / num_kernels);

        // Measure ITERATIVE (clone+insert)
        double iter_5 = 0, iter_10 = 0;
        {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_iterative(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                iter_5 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            iter_5 /= MEASURE_RUNS;
            cleanup_iterative(graph, graphExec);

            build_iterative(&graph, &graphExec, num_kernels, sleep_kernel_10, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                iter_10 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            iter_10 /= MEASURE_RUNS;
            cleanup_iterative(graph, graphExec);
        }
        double iter_overhead = 2 * iter_5 - iter_10;
        printf("iter_clone\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\n",
               num_kernels, iter_5, iter_10, iter_overhead, iter_overhead / num_kernels);

        // Measure ITERATIVE_CHAIN (flat+insert)
        double chain_5 = 0, chain_10 = 0;
        {
            cudaGraph_t graph;
            cudaGraphExec_t graphExec;
            build_iterative_chain(&graph, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                chain_5 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            chain_5 /= MEASURE_RUNS;
            cleanup_iterative_chain(graph, graphExec);

            build_iterative_chain(&graph, &graphExec, num_kernels, sleep_kernel_10, NULL, blocks, threads, NULL);
            for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
            for (int r = 0; r < MEASURE_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(graphExec, 0);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                chain_10 += std::chrono::duration<double, std::nano>(end - start).count();
            }
            chain_10 /= MEASURE_RUNS;
            cleanup_iterative_chain(graph, graphExec);
        }
        double chain_overhead = 2 * chain_5 - chain_10;
        printf("iter_chain\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\n",
               num_kernels, chain_5, chain_10, chain_overhead, chain_overhead / num_kernels);

        // Measure MERGED_FLAT (only for num_kernels >= 2)
        if (num_kernels >= 2) {
            double merged_5 = 0, merged_10 = 0;
            {
                cudaGraph_t graphs[2];
                cudaGraphExec_t graphExec;
                build_merged_flat(graphs, &graphExec, num_kernels, sleep_kernel_5, NULL, blocks, threads, NULL);
                for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
                for (int r = 0; r < MEASURE_RUNS; r++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    cudaGraphLaunch(graphExec, 0);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    merged_5 += std::chrono::duration<double, std::nano>(end - start).count();
                }
                merged_5 /= MEASURE_RUNS;
                cleanup_merged_flat(graphs, graphExec);

                build_merged_flat(graphs, &graphExec, num_kernels, sleep_kernel_10, NULL, blocks, threads, NULL);
                for (int w = 0; w < WARMUP_RUNS; w++) { cudaGraphLaunch(graphExec, 0); cudaDeviceSynchronize(); }
                for (int r = 0; r < MEASURE_RUNS; r++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    cudaGraphLaunch(graphExec, 0);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    merged_10 += std::chrono::duration<double, std::nano>(end - start).count();
                }
                merged_10 /= MEASURE_RUNS;
                cleanup_merged_flat(graphs, graphExec);
            }
            double merged_overhead = 2 * merged_5 - merged_10;
            printf("merged_flat\t%d\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\n",
                   num_kernels, merged_5, merged_10, merged_overhead, merged_overhead / num_kernels);
        }

        printf("\n");
    }
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

    printf("CUDA Graph Child Graph Benchmark\n");
    printf("GPU: %s (SM count: %u)\n", deviceProp.name, smx_count);
    printf("Methods:\n");
    printf("  flat        - N kernels in sequence\n");
    printf("  hierarchy   - g2=g1+g1 (balanced binary tree)\n");
    printf("  iter_clone  - clone+insert (balanced, expensive clone)\n");
    printf("  iter_chain  - flat_N/2 + child (chain of decreasing flats)\n");
    printf("  merged_flat - flat_N/2 + child(flat_N/2) (two equal flats)\n");
    printf("=======================================================================\n\n");

    Test_Correctness(smx_count, 1024);
    Test_Construction(smx_count, 1024);
    Test_Runtime(smx_count, 1024);

    return 0;
}
