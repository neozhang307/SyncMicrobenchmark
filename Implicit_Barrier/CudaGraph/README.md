# CUDA Graph Overhead Analysis

This module provides comprehensive benchmarks for understanding CUDA Graph overheads.

## Potential Overheads to Test

### 1. Graph Creation Overhead

| Test | Description | Method |
|------|-------------|--------|
| Stream Capture | Time to capture kernels into a graph | `cudaStreamBeginCapture` / `cudaStreamEndCapture` |
| Graph Instantiation | Time to create executable graph | `cudaGraphInstantiate` |
| Explicit Graph API | Time to build graph using explicit API | `cudaGraphCreate` / `cudaGraphAddKernelNode` |

**Scaling dimensions:**
- Number of kernel nodes (1, 2, 4, 8, 16, 32, 64, 128)
- Graph topology (linear chain vs parallel vs complex DAG)

### 2. Graph Launch Overhead

| Test | Description | Status |
|------|-------------|--------|
| Per-kernel overhead | Overhead per kernel within a graph | Done (`cuda_graph_launch`) |
| Per-launch overhead | Overhead of `cudaGraphLaunch` API call | Done (`graph_replay`) |
| Cold vs warm launch | First launch vs subsequent launches | Planned |

### 3. Graph Update Overhead

| Test | Description | Method |
|------|-------------|--------|
| Full graph update | Replace entire graph executable | `cudaGraphExecUpdate` |
| Kernel parameter update | Update kernel arguments only | `cudaGraphExecKernelNodeSetParams` |
| Node enable/disable | Conditionally skip nodes | `cudaGraphNodeSetEnabled` |

### 4. Graph Destruction Overhead

| Test | Description | Method |
|------|-------------|--------|
| Graph destruction | Time to destroy graph object | `cudaGraphDestroy` |
| Executable destruction | Time to destroy executable | `cudaGraphExecDestroy` |

### 5. Scaling Analysis

| Dimension | Range | Purpose |
|-----------|-------|---------|
| Node count | 1 - 1024 | How overhead scales with graph size |
| Launch count | 1 - 10000 | Amortization of creation overhead |
| Kernel complexity | empty - compute-bound | Effect of kernel duration on overhead |

## Current Results (from parent benchmarks)

From `../run_benchmark.sh`:

| Metric | Traditional | Cooperative | CUDA Graph | Graph Replay |
|--------|-------------|-------------|------------|--------------|
| Empty kernel overhead | ~2000 ns | ~2000 ns | ~470 ns | ~2600 ns |
| With workload overhead | ~2500 ns | ~2000 ns | ~10 ns | ~2900 ns |

**Key insights:**
- CUDA Graph reduces per-kernel overhead by ~77% (empty kernel)
- CUDA Graph reduces per-kernel overhead by ~99% when kernels have work
- `cudaGraphLaunch` API overhead (~2.6 us) is similar to traditional launch
- Benefit comes from batching multiple kernels into a single graph

## CUDA Graph Available Features

| Feature | Node Type | API | Min CUDA | Tested | Description |
|---------|-----------|-----|----------|--------|-------------|
| Kernel Node | Basic | `cudaGraphAddKernelNode` | 10.0 | | Standard kernel execution |
| Memcpy Node | Basic | `cudaGraphAddMemcpyNode` | 10.0 | | Memory copy operations |
| Memset Node | Basic | `cudaGraphAddMemsetNode` | 10.0 | | Memory set operations |
| Host Node | Basic | `cudaGraphAddHostNode` | 10.0 | | Host callback function |
| **Child Graph** | Basic | `cudaGraphAddChildGraphNode` | 10.0 | ✓ | Embed subgraph as node |
| **Empty Node** | Basic | `cudaGraphAddEmptyNode` | 10.0 | ✓ | Synchronization/dependency point |
| **Event Wait** | Event | `cudaGraphAddEventWaitNode` | 11.1 | ✓ | Wait for CUDA event |
| **Event Record** | Event | `cudaGraphAddEventRecordNode` | 11.1 | ✓ | Record CUDA event |
| External Semaphore Wait | Sync | `cudaGraphAddExternalSemaphoresWaitNode` | 11.2 | | Wait external semaphore |
| External Semaphore Signal | Sync | `cudaGraphAddExternalSemaphoresSignalNode` | 11.2 | | Signal external semaphore |
| Memory Alloc | Memory | `cudaGraphAddMemAllocNode` | 11.4 | | Allocate memory in graph |
| Memory Free | Memory | `cudaGraphAddMemFreeNode` | 11.4 | | Free memory in graph |
| If Conditional | Conditional | `cudaGraphConditionalHandleCreate` | 12.4 | | Conditional branch execution |
| **While Conditional** | Conditional | `cudaGraphConditionalHandleCreate` | 12.4 | ✓ | Loop execution in graph |

### While Conditional Overhead Test

Compare while conditional loop against alternatives using sleep instruction to simulate workload (consistent with other Implicit_Barrier benchmarks).

| Method | Description |
|--------|-------------|
| `while_conditional` | CUDA Graph with while conditional node, N iterations |
| `graph_replay` | Host loop calling `cudaGraphLaunch` N times (1-kernel graph) |
| `host_loop` | Host loop launching graph N times with `cudaGraphLaunch` |
| `device_loop` | Single kernel with device-side loop + grid sync per iteration |

Each iteration executes a sleep kernel (~5000 ns workload) to measure per-iteration overhead under realistic conditions.

### Child Graph Iteration Test

Compare iterative execution using child graph composition vs flat graph.

| Method | Description |
|--------|-------------|
| `child_graph` | Hierarchical: g1=1 kernel, g2=g1+g1, g4=g2+g2, ..., gN has N iterations |
| `flat_graph` | Single graph with N kernel nodes in sequence |
| `merged_flat_NxM` | Merge M copies of flat_N graph using child graph nodes |

**Key findings:**
- **Construction overhead**: Child graph hierarchy is expensive (411 us for 128 iter) vs flat graph (81 us)
- **Merged flat overhead**: Merging two flat_64 adds ~18% overhead vs flat_128 directly
- **Runtime**: All methods have identical per-iteration runtime once instantiated (~39 ns real overhead at 128 iter)
- CUDA optimizes/flattens the graph structure during instantiation

### Empty Node Overhead Test

Compare flat graphs with and without empty nodes to measure empty node overhead.

| Method | Description |
|--------|-------------|
| `flat_kernel_only` | N kernels in sequence |
| `flat_kernel+empty` | N kernels with empty node after each (2N nodes total) |

**Key findings:**
- **Construction overhead**: Empty nodes add ~60% construction overhead (87 us → 138 us for 128 iter)
- **Runtime overhead**: Empty nodes add **zero runtime overhead** (~0 ns difference)
- CUDA optimizes away empty nodes at runtime - they are purely dependency markers

### Event Node Ping-Pong Test

Test inter-graph synchronization using event record/wait nodes. Two graphs on separate streams alternate execution via events.

```
Execution: A.sleep0 -> B.sleep0 -> A.sleep1 -> B.sleep1 -> ...

Graph A (stream1): sleep[0] -> record[A0] -> wait[B0] -> sleep[1] -> ...
Graph B (stream2): wait[A0] -> sleep[0] -> record[B0] -> wait[A1] -> ...
```

| Method | Description |
|--------|-------------|
| `pingpong_2graph` | Two graphs with N kernels each, alternating via 2N-1 events |
| `single_graph` | One graph with 2N kernels using dependency edges |

**Key findings:**
- **Construction overhead**: Ping-pong is ~2.7x more expensive (208 us vs 76 us for 128 kernels)
- **Per-sync overhead**: ~**1500 ns per event record/wait** pair at scale (64 iterations, 127 syncs)
- **Comparison**: In-graph dependency edges have ~40 ns per-kernel overhead
- Use single graph with dependency edges when possible; event nodes only for true multi-graph scenarios

## Planned Implementation

### Phase 1: Graph Creation Analysis
- Measure stream capture overhead vs number of kernels
- Measure instantiation overhead vs number of kernels
- Compare stream capture vs explicit graph API

### Phase 2: Graph Update Analysis
- Measure `cudaGraphExecUpdate` overhead
- Measure kernel parameter update overhead
- Determine when update is faster than rebuild

### Phase 3: Comprehensive Scaling
- Create scaling curves for all overheads
- Identify crossover points (when CUDA Graph becomes beneficial)
- Analyze memory overhead
