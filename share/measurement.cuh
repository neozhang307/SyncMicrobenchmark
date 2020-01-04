// #include "Explicit_Barrier_Kernel.cuh"

#include "../share/util.h"
#include "../share/wrap_launch_functions.cuh"

#ifndef SIZE
	#define SIZE 101
#endif

int measureInterSMLatency(latencys* result, launchfunction_rkernel run_func, 
	fbaseKernel kernel_func, unsigned int gpu_count,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
//suceess when return 1
int measureIntraSMLatency(latencys* result, 
	launchfunction_rkernel run_func, fbaseKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, 
	float a=2, float b=2, unsigned int tile=32);

template <int gpu_count>
void measureKernelLatency(latencys* result, 
	launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);