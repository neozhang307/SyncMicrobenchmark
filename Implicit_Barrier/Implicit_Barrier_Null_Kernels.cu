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

	free(result);
}


#define NULL_KERNEL_TEST_8GPU(callfunc, basicDEP,moreDEP, block_perGPU, thread_perBlock) \
	if(gpu_count>=1)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 1, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=2)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 2, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=3)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 3, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=4)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 4, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=5)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 5, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=6)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 6, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=7)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 7, block_perGPU, thread_perBlock);\
	}\
	if(gpu_count>=8)\
	{\
		TEST_ADDITIONAL_LATENCY(callfunc, null_kernel, 1, 128, 8, block_perGPU, thread_perBlock);\
	}\

void Test_Null_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock, unsigned int gpu_count)
{

	printf("_______________________________________________________________________\n");
	printf("Empty Kernel for multi-GPU\n");
	printf("When Calling count is one, the result of total latency (ns)\n");

	latencys* result  = (latencys*)malloc(2*sizeof(latencys));
	
	NULL_KERNEL_TEST_8GPU(multi_cooperative_launch,1,128,block_perGPU,thread_perBlock);
	NULL_KERNEL_TEST_8GPU(omp_traditional_launch,1,128,block_perGPU,thread_perBlock);

	free(result);
}

// int main(int argc, char **argv)
// {
// 	cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, 0);
//     cudaCheckError();
//    	unsigned int smx_count = deviceProp.multiProcessorCount;

// 	Null_Kernel(smx_count,1024);
// 	Null_Kernel_MGPU<2>(1,32);

// }
