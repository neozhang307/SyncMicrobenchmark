# Sync_Microbenchmark
## Abstract
This work aims at characterizing the synchronization methods in CUDA. It mainly includes two parts:
1. Non-primitive Synchronization:
  * Implicit Barrier, i.e. overhead of launching a kernel, including the kernel launch function for Cooperative Groups.
2. Primitive Synchronization Methods in Nvidia GPUs:
  * Warp level, Thread block level, and grid-level synchronization.

Note: Multi-grid synchronization (cudaLaunchCooperativeKernelMultiDevice) was removed in CUDA 13.0. This codebase has been updated to remove all multi-grid related code.

For stable results, please avoid using dynamic frequency.

## Requirements
* CUDA 13.1+
* Supported architectures: sm_80, sm_90, sm_100, sm_120

## Quick Start

### Implicit Barrier
```bash
cd Implicit_Barrier
make
./run_benchmark.sh
```

### Explicit Barrier
```bash
cd Explicit_Barrier
make
./run_benchmark.sh
```

## Implicit Barrier
### Compile
Directly compile with the Makefile in Implicit_Barrier folder.

The sleep function is only available after sm_70.

### Output explanation

#### Null Kernel
* method: the method to do kernel launch \[traditional_launch|cooperative_launch\]
* GPUCount: how many gpu involved (always 1)
* rep: repeat calling launch function times
* blk: griddim
* thrd: blockdim
* m(clk) s(clk): mean and standard variation of CPU clock() function
* m(sync) s(sync): mean and standard variation of DeviceSynchronization();
* m(laun) s(laun): latency of launch functions (test before DeviceSynchronization());
* m(ttl) s(ttl): latency of kernels total execution (test after Device Synchronization());
* m(avelaun) s(avelaun): mean and standard variation of average latency of each launch function, compute with m(laun)/rep
* m(addl) s(addl): mean and standard variation of average additional latency for each additional launch function. By default this code use repeat 1 times and 128 times and compute by m(ttl_128)-m(ttl_1)/(128-1)

By using "additional latency", it will be possible to eliminate the overhead of synchronization (which is not negligible when considering kernel overhead) and other unrelated parts. Details are explained in the Use_Microbenchmark_To_Better_Understand_The_Overhead_Of_CUDA_Kernels__Poster_.pdf in the same folder.

#### Sleep Kernel (Fused Sleep Kernels to test the kernel overhead when kernel execution latency is long enough)
* method: the method to do kernel launch \[traditional_launch|cooperative_launch\]
* GPUCount: how many GPU involved (always 1)
* rep: repeat calling launch function times for both the basic kernel and the fused kernel.
* blk: griddim
* thrd: blockdim
* idea(wkld): the work unit. When fuse two kernel means the kernel execution latency of fused kernel should be twice the idea(wkld)
* m(wkld) s(wkld): the basic workload deduce from the measurements.

#### Workload Test
The same as NULL KERNEL. Using "Sleep" to act as "workload"

Just to show how additional latency tested is related to the real kernel execution latency. Default sequencing is "(0 1 2 4 8 16 32 64 128)X2000ns". Before a certain point, increasing kernel execution latency would not affect the additional latency caused by the additional kernel (imagine a pipeline when pipeline is full, the most timeconsuming step mainly influence the performance of the whole system).

Details are explained in the Use_Microbenchmark_To_Better_Understand_The_Overhead_Of_CUDA_Kernels__Poster_.pdf in the same folder.

## Explicit Barrier
### Compile
Directly compile with the Makefile in Explicit_Barrier folder. Three executable files will be created:
* TestRepeat
* BenchmarkIntraSM
* BenchmarkInterSM

#### TestRepeat
To show if repeating a synchronization instruction will influence the performance itself

The result shows that the result of shuffle, block sync and grid-level syncs become more accurate as the repeat times increase. But for warp level syncs, this would happen, probably because the current implementation is based on software codes, repeat too many times will cause instruction overflow, harming the performance.

##### Execution
./TestRepeat

##### Output
* method: kernel function name
* rep: repeat instruction times
* blk: griddim
* thrd: blockdim
* tile: control the group size of coalesced group and tile group
* m(cycle) s(cycle): mean and standard variation of total instruction execution
* m(ns) s(ns): mean and standard variation of total kernel execution (meaningless here just for comparison)
* m(ave_cycle) s(ave_cycle): mean and standard variation of average instruction(cycle)

#### BenchmarkIntraSM
Benchmark measurements that only need clock inside an SM. Includes:

Throughput of Warp level syncs and block sync
Latency of block sync for each possible group

##### Execution
./BenchmarkIntraSM

##### Output
###### Latency
* method: kernel function name
* GPUcount: 1
* rep: repeat instruction times
* blk: griddim
* thrd: blockdim
* m(ave_cycle) s(ave_cycle): mean and standard variation of average instruction(cycle)
###### Throughput
* method: kernel function name
* GPUcount: 1
* rep: repeat instruction times
* blk: griddim
* thrd: blockdim
* tile: used to control tile group size
* m(ttl_latency): total latency for synchronization
* m(thrput): throughput (warp/cycle) computed base on m(ttl_latency)

#### BenchmarkInterSM
Benchmark measurements involve several SMs

Latency of grid-level syncs (using cooperative launch with grid_group.sync())

##### Execution
./BenchmarkInterSM

##### Output
* method: kernel function name
* GPUcount: GPU involved (always 1)
* basicrep: repeat instruction times in basic kernel
* morerep: repeat instruction times in more kernel
* blk: griddim
* thrd: blockdim
* m(basic_ttl) s(basic_ttl): mean and standard variation of total kernel latency (ns) for executing basic kernel
* m(more_ttl) s(more_ttl): mean and standard variation of total kernel latency (ns) for executing more kernel
* m(avginstr) s(avginstr): mean and standard variation of average instruction (ns) deduced

## Reduction
A simple grid-level reduction benchmark using cooperative groups.

### Compile
```bash
cd Reduction
make
./greduce
```

## Version History
* **v2.0** (2024): Updated for CUDA 13.1, removed deprecated multi-grid synchronization APIs
* **v1.0** (2020): Original release with multi-grid support (CUDA 9.0+)

## Citation

This research was published in IPDPS 2020. Please cite:

Lingqi Zhang, Mohamed Wahib, Haoyu Zhang, Satoshi Matsuoka. A Study of Single and Multi-device Synchronization Methods in Nvidia GPUs. In Proceedings of the IPDPS 2020.
