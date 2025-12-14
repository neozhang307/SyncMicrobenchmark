// Wrapper functions for kernel launch methods
// Updated for CUDA 13.1 - removed deprecated cudaLaunchParams and multi-device APIs
#include "repeat.h"

typedef void (*nKernel)();
typedef void (*fbaseKernel)(float,float,double*,unsigned int*,unsigned int*, unsigned int);

typedef void(*launchfunction_nkernel)(nKernel,
	unsigned int, unsigned int);

typedef void(*launchfunction_rkernel)(fbaseKernel,
	unsigned int, unsigned int,
	void**, unsigned int);

#ifndef DEF_WRAP_LAUNCH_FUNCTION

void __forceinline__ traditional_launch(fbaseKernel func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, void** KernelArgs,
	unsigned int GPU_count=1)
{
	func<<<blockPerGPU,threadPerBlock>>>(((float*)KernelArgs[0])[0],((float*)KernelArgs[1])[0],
		((double**)KernelArgs[2])[0],((unsigned int**)KernelArgs[3])[0],
		((unsigned int**)KernelArgs[4])[0],(( unsigned int*)KernelArgs[5])[0]);
}

void __forceinline__ cooperative_launch(fbaseKernel func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, void** KernelArgs,
	unsigned int GPU_count=1)
{
	cudaLaunchCooperativeKernel((void*)func, blockPerGPU, threadPerBlock, KernelArgs, 32, 0);
}

void __forceinline__ traditional_launch(nKernel func, unsigned int blockPerGPU, unsigned int threadPerBlock)
{
	func<<<blockPerGPU,threadPerBlock>>>();
}

void __forceinline__ cooperative_launch(nKernel func, unsigned int blockPerGPU, unsigned int threadPerBlock)
{
	void* KernelArgs[] = {};
	cudaLaunchCooperativeKernel((void*)func, blockPerGPU, threadPerBlock, KernelArgs, 32, 0);
}

#define repeatlaunch(fname, DEP) \
void __forceinline__ fname##_##DEP(nKernel func, unsigned int blockPerGPU, unsigned int threadPerBlock)\
{\
	repeat##DEP(fname(func,blockPerGPU,threadPerBlock););\
}

#define gencallfun(callfunc) \
	repeatlaunch(callfunc,1); \
	repeatlaunch(callfunc,16); \
	repeatlaunch(callfunc,128); \

gencallfun(traditional_launch);
gencallfun(cooperative_launch);

#define DEF_WRAP_LAUNCH_FUNCTION
#endif
