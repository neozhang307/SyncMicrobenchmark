#include "Implicit_Barrier_Kernel.cuh"
#include "repeat.h"


#ifndef DEF_WRAP_LAUNCH_FUNCTION
typedef void(*launchfunction)(nKernel, unsigned int, unsigned int, unsigned int, cudaLaunchParams*);

void __forceinline__ traditional_launch(nKernel func, unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	func<<<blockPerGPU,threadPerBlock>>>();
}

void __forceinline__ cooperative_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	void*KernelArgs[] = {};
	cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
}

void __forceinline__ multi_cooperative_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	cudaLaunchCooperativeKernelMultiDevice(launchParamsList, GPU_count);
}

#define repeatlaunch(fname, DEP) \
void __forceinline__ fname##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)\
{\
	repeat##DEP(fname(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList););\
}

#define gencallfun(callfunc) \
	repeatlaunch(callfunc,1); \
	repeatlaunch(callfunc,128); \

gencallfun(traditional_launch);
gencallfun(cooperative_launch);
gencallfun(multi_cooperative_launch);

//only for single GPU
#include<omp.h>

//The way to repeat omp function is a little different here. Because I found I can not include #pragma statement in MACRO
void __forceinline__ single_omp_traditional_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL, unsigned int repeat=2)
{
	#pragma omp parallel num_threads(GPU_count)
	{
		/* code */
		int tid = omp_get_thread_num();
		cudaSetDevice(tid);
		for(int j=0; j<repeat; j++)
		{
		func<<<blockPerGPU,threadPerBlock,0,launchParamsList[tid].stream>>>();
			cudaDeviceSynchronize();
			#pragma omp barrier
		}
	}
	
}

//A little different here
#define repeatomp(DEP) \
void __forceinline__ omp_traditional_launch##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL)\
{\
	single_omp_traditional_launch(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList,DEP);\
}

repeatomp(1);
repeatomp(128); 

#define DEF_WRAP_LAUNCH_FUNCTION
#endif