#include "repeat.h"
#include "measurement.cuh"
#include "wrap_launch_functions.cuh"
#include "util.h"

//In order to reduce overhead, we use repeat MACRO instead of forloop here. 
//MACRO is easier here
#define TEST_ADDITIONAL_LATENCY(callfunc, kernel,basicDEP, moreDEP, gpu_count, block_perGPU,thread_perBlock); \
	measureLatencys<gpu_count>(result, callfunc##_##basicDEP, kernel,block_perGPU,thread_perBlock);\
	measureLatencys<gpu_count>(result+1, callfunc##_##moreDEP, kernel,block_perGPU,thread_perBlock);\
	prepare_showAdditionalLatency();\
	showAdditionalLatency(result[0],result[1],#callfunc,gpu_count,block_perGPU,thread_perBlock,basicDEP,moreDEP);\

//MACRO is easier to write here
#define TEST_FUSED_KERNEL_1V16_DIFERENCE(callfunc, baisckernel,fusedkernel, basicDEP, moreDEP, gpu_count,block_perGPU,thread_perBlock,idea_basic_workload) \
	measureLatencys<gpu_count>(result,   callfunc##_1,  fusedkernel, block_perGPU, thread_perBlock);\
	measureLatencys<gpu_count>(result+1, callfunc##_16, baisckernel, block_perGPU, thread_perBlock);\
	measureLatencys<gpu_count>(result+2, callfunc##_1,  baisckernel, block_perGPU, thread_perBlock);\
	prepare_showFusedResult();\
	showFusedResult(result[2],result[0],result[1],#callfunc,gpu_count,block_perGPU,thread_perBlock,basicDEP,moreDEP,idea_basic_workload);\



void Test_Null_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock);

template <int gpu_count>
void Test_Null_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock);

void Test_Sleep_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock);

template <int gpu_count>
void Test_Sleep_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock);
