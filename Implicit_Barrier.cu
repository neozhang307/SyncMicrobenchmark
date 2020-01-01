#include "repeat.h"
// #include "Implicit_Barrier_Kernel.cuh"
#include "Implicit_Barrier.h"
#include "util.h"

#include <stdio.h>
/*
Definition:
* {Kernel Execution Latency:} Total time spent in executing the kernel, excluding any overhead for launching the kernel.
* {Launch Overhead:} Latency that is not related to kernel execution. 
* {Kernel Total Latency:} Total latency to run kernels.
Depends on different situation, the launch overhead could be different:
Situation 1: Launch a single kernel
Situation 2: Launch additional "small kenel" (By "small" we mean the device is not saturate at all, in this experiment in single GPU if each kernel lasts less then 5us it is defined as "small")
Situation 3: Launch additional "big kernel" (By "big" we mean the device is saturate enough while the workload is not severe, in this experiment in single GPU if each kernel lasts longer than 5us, it is defined as "big")

When kernels are "small" or each kernel lasts fewer than 5us, it would not be practical to offload these workloads to GPU at all. So, we only include the launch overhead of "big kernel" in our IPDPS20 paper.

But in this microbenchmark, we include the measurements of launch overhead in all three situations. The detailed information about these measurements are explained in an ICPP19 Poster in this same folder. 

*/

void workload_influence()
{
	//increase the workload only test the additional latency.

}


int main(int argc, char **argv)
{
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();
   	unsigned int smx_count = deviceProp.multiProcessorCount;
//	double* result=(double*)malloc(sizeof(double)*6*8);
	//show how total latency is influenced by execution (traditional launch)


	//merge this two situation together
	//launch single null kernel and different features
	//launch additional null kernel and compute the kernel overhead here
	
	// Test_Null_Kernel(smx_count,1024);
	// Test_Null_Kernel_MGPU<2>(1,32);
	
	//launch big kernel and additional big kernel to compute the kernel overhead
	Test_Sleep_Kernel(smx_count,1024);
	Test_Sleep_Kernel_MGPU<8>(smx_count,1024);

	Test_Workload_Influence(smx_count,1024);
}



