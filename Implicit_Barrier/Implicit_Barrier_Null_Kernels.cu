#include "Implicit_Barrier_Kernel.cuh"
#include "Implicit_Barrier.h"

#include "../share/util.h"


#include <string.h>
#include <stdio.h>


__global__ void null_kernel(){}



void Test_Null_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock)
{
	latencys* result  = (latencys*)malloc(2*sizeof(latencys));

	printf("_______________________________________________________________________\n");
	printf("Empty Kernel\n");
	printf("When Calling count is one, the result of total latency (ns)\n");

	TEST_ADDITIONAL_LATENCY(traditional_launch, null_kernel,1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(cooperative_launch, null_kernel,1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(cuda_graph_launch, null_kernel,1,128,1, block_perGPU, thread_perBlock);

	free(result);
}
