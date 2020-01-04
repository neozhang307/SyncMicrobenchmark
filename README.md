# Sync_Microbenchmark
## Abstract
This work aims at characterizing the synchronization methods in CUDA. It mainly includes two-part:
1. Non-primitive Synchronization:
  * Implicit Barrier, i.e. overhead of launching a kernel, including the new kernel launch function introduced for Cooperative Groups.
  * Multi-GPU synchronization with OpenMP in a single node.
2. Primitive Synchronization Methods in Nvidia GPUs introduced in CUDA 9.0:
  * Warp level, Thread block level, and grid-level synchronization.
  * Multi-grid synchronization for Multi-GPU synchronization.
## Requirements
 CUDA 9.0
 sm_60 for most of the measurements
 sm_70 for sleep instruction involved measurements.
 
## Implicit Barrier
### Compile
Directly compile with the Makefile in Implicit_Barrier folder

The sleep function is only available after sm_70. We tried to use other instructions to control the kernel execute latency, but the result is not so stable as sleep instruction, and larger than sleep instruction.
### Input explanation
 * ImplicitBarrier \[gpu_count\]
 * gpu_count is 2 by default. When testing multi-GPU related latencies, the experiments would iterate from 1 to gpu_count.
### Output explanation

#### Null Kernel 
* method: the method to do kernel launch \[traditional_launch|cooperative_launch|multi_cooperative_launch|single_omp_traditional_launch\]
* GPUCount: how many gpu involved
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
* method: the method to do kernel launch \[traditional_launch|cooperative_launch|multi_cooperative_launch|single_omp_traditional_launch\]
* GPUCount: how many GPU involved
* rep: repeat calling launch function times for both the basic kernel and the fused kernel.
* blk: griddim
* thrd: blockdim
* idea(wkld): the work unit. When fuse two kernel means the kernel execution latency of fused kernel should be twice the idea(wkld)
* m(wkld) s(wkld): the basic workload deduce from the measurements. 

#### Workload Test
The same of NULL KERNEL

Just to show how additioanl latency tested is related to the real kernel execution latency. Before a certain point, increasing kernel execution latency would not affect the additional latency caused by the additional kernel. 

Details are explained in the Use_Microbenchmark_To_Better_Understand_The_Overhead_Of_CUDA_Kernels__Poster_.pdf in the same folder.

## Explicit Barrier
### Compile
Directly compile with the Makefile in Explicit_Barrier folder. Three executable files will be created:
* TestRepeat
* BenchmarkIntraSM
* BenchmarkInterSM

#### TestRepeat
Used to show if repeating a synchronization instruction will influence the performance itself

The result shows that the result of shufl, block sync and grid-level syncs become more accurate as the repeat times increase. But for warp level syncs, this would happen, probably because the current implementation is based on software codes, repeat too many times will cause instruction overflow, harming the performance.

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
Used to benchmark measurements that only need clock inside an SM

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
* m(thrput): throuput (warp/cycle) computed base on m(ttl_latency)

#### BenchmarkInterSM
Used to benchmark measurements involve several SMs

Latency of grid-level syncs

##### Execution
./BenchmarkInterSM \[gpu_count\]

##### Output
* method: kernel function name
* GPUcount: GPU involved
* basicrep: repeat instruction times in basic kernel
* moredrep: repeat instruction times in more kernel
* blk: griddim
* thrd: blockdim
* m(basic_ttl) s(basic_ttl): mean and standard variation of total kernel latency (ns) for executing basic kernel
* m(more_ttl) s(more_ttl): mean and standard variation of total kernel latency (ns) for executing more kernel
* m(avginstru) s(avginstr): mean and standard variation of average instruction (ns) deduced


## Citation
  For the study of implicit barrier synchronization, please cite:
  
  Lingqi Zhang, Mohamed Wahib, Satoshi Matsuoka. Understanding the Overheads of Launching CUDA Kernels. Poster presented at: The 48th International Conference on Parallel Processing (ICPP 2019); 2019 August 5-8; Kyoto, Japan. 
    
  This research will be published in IPDPS20. Please cite:
  
  Lingqi Zhang, Mohamed Wahib, Haoyu Zhang, Satoshi Matsuoka. A Study of Single and Multi-device Synchronization Methods in Nvidia GPUs. InProceedings of the IPDPS 2020.
