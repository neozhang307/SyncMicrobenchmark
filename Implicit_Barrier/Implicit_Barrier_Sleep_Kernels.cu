#include "Implicit_Barrier_Kernel.cuh"
#include "Implicit_Barrier.h"
#include "../share/wrap_launch_functions.cuh"

#include <stdio.h>
//The kernel should last long enough to study the launch overhead hidden in kernel launch. 
//These kernels are generated for DGX1. 
//1 node
N_KERNEL(5);
	//80
//2 node
N_KERNEL(15);
	N_KERNEL(240);
//3 node
N_KERNEL(30);
	N_KERNEL(480);
//4 node
N_KERNEL(50);
	N_KERNEL(800);
//5 node
N_KERNEL(80);
	N_KERNEL(1280);
//6 node
N_KERNEL(105);
	N_KERNEL(1680);
//7 node
N_KERNEL(160);
	N_KERNEL(2560);
//8 node
N_KERNEL(200);
	N_KERNEL(3200);


//to study the relationship between kernel total latency and execution latency


#define PACKED_SLEEP_TEST(callfunc,basickernel,fusedkernel, gpu_count, block_perGPU, thread_perBlock,idea_workload); \
	TEST_FUSED_KERNEL_1V16_DIFERENCE(callfunc, basickernel, fusedkernel,1,16,gpu_count, block_perGPU,thread_perBlock,idea_workload);\

void Test_Sleep_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock)
{
	latencys* result  = (latencys*)malloc(3*sizeof(latencys));
	printf("_______________________________________________________________________\n");

	printf("Fuse Sleep Kernels to test overhead when kernel execution latency is long enough\n");
	PACKED_SLEEP_TEST(traditional_launch, null_kernel_5, null_kernel_80, 1, block_perGPU, thread_perBlock,5000);
	PACKED_SLEEP_TEST(cooperative_launch, null_kernel_5, null_kernel_80, 1, block_perGPU, thread_perBlock,5000);

	free(result);
}


// template <int gpu_count>
void Test_Sleep_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock, unsigned int gpu_count)
{
	printf("_______________________________________________________________________\n");
	printf("Fuse Sleep Kernel for multi-GPU\n");
	latencys* result  = (latencys*)malloc(3*sizeof(latencys));

	if(gpu_count>=1)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_5, null_kernel_80, 1, block_perGPU, thread_perBlock,5000);
	}
	if(gpu_count>=2)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_15, null_kernel_240, 2, block_perGPU, thread_perBlock,15000);
	}
	if(gpu_count>=3)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_30, null_kernel_480, 3, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=4)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_50, null_kernel_800, 4, block_perGPU, thread_perBlock,50000);
	}
	if(gpu_count>=5)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_80, null_kernel_1280, 5, block_perGPU, thread_perBlock,80000);
	}
	if(gpu_count>=6)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_105, null_kernel_1680, 6, block_perGPU, thread_perBlock,105000);
	}
	if(gpu_count>=7)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_160, null_kernel_2560, 7, block_perGPU, thread_perBlock,160000);
	}
	if(gpu_count>=8)
	{
		PACKED_SLEEP_TEST(multi_cooperative_launch, null_kernel_200, null_kernel_3200, 8, block_perGPU, thread_perBlock,200000);
	}


	//in fact there is no need to use sleep to test the overhead here, because the main overhead actually comes from cleaning the pipeline when calling cudasyncdevice()
	if(gpu_count>=1)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=2)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=3)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=4)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=5)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=6)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=7)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	if(gpu_count>=8)
	{
		PACKED_SLEEP_TEST(omp_traditional_launch, null_kernel_30, null_kernel_480, 1, block_perGPU, thread_perBlock,30000);
	}
	free(result);
}



N2_KERNEL(0);
N2_KERNEL(1);
N2_KERNEL(2);
N2_KERNEL(4);
N2_KERNEL(8);
N2_KERNEL(16);
N2_KERNEL(32);
N2_KERNEL(64);
N2_KERNEL(128);


void Test_Workload_Influence(unsigned int block_perGPU, unsigned int thread_perBlock)
{
	printf("_______________________________________________________________________\n");
	printf("Test the influence of kernel execution latency with additional latency\n");	
	latencys* result  = (latencys*)malloc(3*sizeof(latencys));
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_0, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_1, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_2, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_4, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_8, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_16, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_32, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_64, 1,128,1, block_perGPU, thread_perBlock);
	TEST_ADDITIONAL_LATENCY(traditional_launch,null2_kernel_128, 1,128,1, block_perGPU, thread_perBlock);
	free(result);
}

