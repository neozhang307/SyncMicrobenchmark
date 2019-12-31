#include "wrap_launch_functions.cuh"
#include "Implicit_Barrier_Kernel.cuh"
#include "util.h"
#ifndef SIZE
	#define SIZE 101
#endif
template <int gpu_count>
void measureLatencys(latencys* result, launchfunction run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);

void getStatistics(double &mean, double &s,
	double* list, unsigned int size);

// void measureLatencys<1>(latencys* result, launchfunction run_func, nKernel kernel_func,
	// unsigned int blockPerGPU, unsigned int threadPerBlock);