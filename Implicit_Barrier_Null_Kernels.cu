#include "Implicit_Barrier_Kernel.cuh"
#include "Implicit_Barrier.h"
#include "wrap_launch_functions.cuh"
#include "util.h"
#include "measurement.cuh"

#include <stdio.h>


__global__ void null_kernel(){}


//In order to reduce overhead, we use repeat MACRO instead of forloop here. 
//The inconvenience part is that when we need to test overhead, we would need to introduce additional MACRO
#define NULL_KERNEL_TEST(callfunc, basicDEP, moreDEP, gpu_count) \
	printf("method\tGPUcount\trep\tblk\tthrd\tm(clk)\ts(clk)\tm(sync)\ts(sync)\tm(laun)\ts(laun)\tm(ttl)\ts(ttl)\tm(avelaun)\ts(avelaun)\tm(addl)\ts(addl)\n");\
	measureLatencys<gpu_count>(result, callfunc##_##basicDEP, null_kernel,block_perGPU,thread_perBlock);\
	printf("%s\t%s\t%s\t%u\t%u\t",#callfunc,#gpu_count,#basicDEP,block_perGPU,thread_perBlock);\
	showlatency(result);printf("%f\t%f\t",result[0].mean_laun/basicDEP,result[0].s_laun/basicDEP);nxtline();\
	measureLatencys<gpu_count>(result+1, callfunc##_##moreDEP, null_kernel,block_perGPU,thread_perBlock);\
	printf("%s\t%s\t%s\t%u\t%u\t",#callfunc,#gpu_count,#moreDEP,block_perGPU,thread_perBlock);\
	showlatency(result+1);printf("%f\t%f\t",result[1].mean_laun/moreDEP,result[1].s_laun/moreDEP);printf("%f\t%f\t",computeAddLat(result,moreDEP-basicDEP),computeAddLats(result,moreDEP-basicDEP));nxtline();


void Null_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock)
{
	latencys* result  = (latencys*)malloc(2*sizeof(latencys));

	printf("Empty Kernel\n");
	printf("When Calling count is one, the result of total latency (ns)\n");

	NULL_KERNEL_TEST(traditional_launch,1,128,1);
	NULL_KERNEL_TEST(cooperative_launch,1,128,1);

	free(result);
}


#define NULL_KERNEL_TEST_8GPU(callfunc, basicDEP,moreDEP) \
	if(gpu_count>=1)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 1);\
	}\
	if(gpu_count>=2)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 2);\
	}\
	if(gpu_count>=3)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 3);\
	}\
	if(gpu_count>=4)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 4);\
	}\
	if(gpu_count>=5)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 5);\
	}\
	if(gpu_count>=6)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 6);\
	}\
	if(gpu_count>=7)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 7);\
	}\
	if(gpu_count>=8)\
	{\
		NULL_KERNEL_TEST(callfunc, 1, 128, 8);\
	}\

template <int gpu_count>
void Null_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock)
{

	printf("Empty Kernel for multi-GPU\n");
	printf("When Calling count is one, the result of total latency (ns)\n");

	latencys* result  = (latencys*)malloc(2*sizeof(latencys));
	
	NULL_KERNEL_TEST_8GPU(multi_cooperative_launch,1,128);
	NULL_KERNEL_TEST_8GPU(omp_traditional_launch,1,128);

	free(result);
}


template void Null_Kernel_MGPU<1>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<2>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<3>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<4>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<5>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<6>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<7>(unsigned int block_perGPU, unsigned int thread_perBlock);
template void Null_Kernel_MGPU<8>(unsigned int block_perGPU, unsigned int thread_perBlock);


// int main(int argc, char **argv)
// {
// 	cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, 0);
//     cudaCheckError();
//    	unsigned int smx_count = deviceProp.multiProcessorCount;

// 	Null_Kernel(smx_count,1024);
// 	Null_Kernel_MGPU<2>(1,32);

// }
