# Sync_Microbenchmark
## Abstract
This work aims at characterizing the synchronization methods in CUDA. It mainly includes two part:
1. Non-primizive Synchronization:
  * Implicit Barrier, i.e. overhead of launching a kernel, including the new kernel launch function introduced for Cooperative Groups.
  * Multi-GPU synchronization with OpenMP in single node.
2. Primitive Synchronization Methods in Nvidia GPUs introduced in CUDA 9.0:
  * Warp level, Thread block level and grid level synchronization.
  * Multi-grid synchronization for Multi-GPU synchronization.
  
## Implicit Barrier
### Compile
Directly compile with make file in Implicit_Barrier folder

The sleep function is only available after sm_70. We tried to use other instruction to control the kernel execute latency, but the result is not so stable as sleep instruction, and larger than sleep instruction.
### Output explaination

#### Null Kernel 
method: the method to do kernel launch \[traditional_launch|cooperative_launch|multi_cooperative_launch|single_omp_traditional_launch\]
GPUCount: how many gpu involved
rep: repeat calling launch function times
blk: griddim
thrd: blockdim
m(clk) s(clk): mean and standard variation of CPU clock() function
m(sync) s(sync): mean and standard variation of DeviceSynchronization(); 
m(laun) s(laun): latency of launch functions (test before DeviceSynchronization());
m(ttl) s(ttl): latency of kernels total execution (test after Device Synchronization());
P.S. time test pls refer to Implicit_Barrier/measurement.cpp measureLatencys function
m(avelaun) s(avelaun): mean and standard variation of average latency of each launch function, compute with m(laun)/rep
m(addl) s(addl): mean and standard variation of average additional latency for each additional launch function. By default this code use repeat 1 times and 128 times and compute by m(ttl_128)-m(ttl_1)/(128-1)

By using "additional latency", it will be possible to eliminate the overhead of synchronization (which is really not neglitable when considering kernel overhead) and other unrelated parts. Details is explained in the Use_Microbenchmark_To_Better_Understand_The_Overhead_Of_CUDA_Kernels__Poster_.pdf in the same folder.

#### Sleep Kernel (Fused Sleep Kernels to test the kernel overhead when kernel execution latency is long enough)
method: the method to do kernel launch \[traditional_launch|cooperative_launch|multi_cooperative_launch|single_omp_traditional_launch\]
GPUCount: how many gpu involved
rep: repeat calling launch function times for both the basic kernel and the fused kernel.
blk: griddim
thrd: blockdim
idea(wkld): the work unit. When fuse two kernel means the kernel execution latency of fused kernel should be twice the idea(wkld)
m(wkld) s(wkld): the basic workload deduce from the measurements. 

#### Workload Test
The same of NULL KERNEL
Just to show how additioanl latency tested is related to the real kernel execution latency. Details is explained in the Use_Microbenchmark_To_Better_Understand_The_Overhead_Of_CUDA_Kernels__Poster_.pdf in the same folder.

## Explicit Barrier
TBC

## Citation
  This research will be published in IPDPS20 
  
