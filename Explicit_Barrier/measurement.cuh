#include "Explicit_Barrier_Kernel.cuh"

#include "../share/util.h"
#include "wrap_launch_functions.cuh"

#ifndef SIZE
	#define SIZE 101
#endif

int measureInterSMLatency(latencys* result, launchfunction run_func, 
	fbaseKernel kernel_func, unsigned int gpu_count,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
//suceess when return 1
int measureIntraSMLatency(latencys* result, 
	launchfunction run_func, fbaseKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, 
	float a=2, float b=2, unsigned int tile=32);