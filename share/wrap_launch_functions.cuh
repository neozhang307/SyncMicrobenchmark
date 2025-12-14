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

// CUDA Graph launch - captures N kernel launches into a graph, then launches it
// Cache: first call builds graph, subsequent calls reuse it
// Cache is invalidated if kernel function changes
static cudaGraphExec_t g_graph_exec_1 = nullptr;
static cudaGraphExec_t g_graph_exec_16 = nullptr;
static cudaGraphExec_t g_graph_exec_128 = nullptr;
static cudaStream_t g_graph_stream = nullptr;
static nKernel g_cached_func_1 = nullptr;
static nKernel g_cached_func_16 = nullptr;
static nKernel g_cached_func_128 = nullptr;

#define CUDA_GRAPH_KERNEL_LAUNCH(f, b, t, s) f<<<b, t, 0, s>>>()

#define cuda_graph_launch_func(DEP) \
void __forceinline__ cuda_graph_launch_##DEP(nKernel func, unsigned int blockPerGPU, unsigned int threadPerBlock)\
{\
	if (g_graph_stream == nullptr) {\
		cudaStreamCreate(&g_graph_stream);\
	}\
	if (g_cached_func_##DEP != func) {\
		if (g_graph_exec_##DEP != nullptr) {\
			cudaGraphExecDestroy(g_graph_exec_##DEP);\
			g_graph_exec_##DEP = nullptr;\
		}\
		g_cached_func_##DEP = func;\
	}\
	if (g_graph_exec_##DEP == nullptr) {\
		cudaGraph_t graph;\
		cudaStreamBeginCapture(g_graph_stream, cudaStreamCaptureModeGlobal);\
		repeat##DEP(CUDA_GRAPH_KERNEL_LAUNCH(func, blockPerGPU, threadPerBlock, g_graph_stream);)\
		cudaStreamEndCapture(g_graph_stream, &graph);\
		cudaGraphInstantiate(&g_graph_exec_##DEP, graph, 0);\
		cudaGraphDestroy(graph);\
	}\
	cudaGraphLaunch(g_graph_exec_##DEP, 0);\
}

cuda_graph_launch_func(1);
cuda_graph_launch_func(16);
cuda_graph_launch_func(128);

// Graph replay - builds graph once with 1 kernel, then launches it N times
// Measures cudaGraphLaunch API overhead (not per-kernel overhead)
// Cache is shared across all DEP variants since they all use 1-kernel graph
static cudaGraphExec_t g_graph_replay_exec = nullptr;
static nKernel g_graph_replay_cached_func = nullptr;

#define graph_replay_func(DEP) \
void __forceinline__ graph_replay_##DEP(nKernel func, unsigned int blockPerGPU, unsigned int threadPerBlock)\
{\
	if (g_graph_stream == nullptr) {\
		cudaStreamCreate(&g_graph_stream);\
	}\
	if (g_graph_replay_cached_func != func) {\
		if (g_graph_replay_exec != nullptr) {\
			cudaGraphExecDestroy(g_graph_replay_exec);\
			g_graph_replay_exec = nullptr;\
		}\
		g_graph_replay_cached_func = func;\
	}\
	if (g_graph_replay_exec == nullptr) {\
		cudaGraph_t graph;\
		cudaStreamBeginCapture(g_graph_stream, cudaStreamCaptureModeGlobal);\
		CUDA_GRAPH_KERNEL_LAUNCH(func, blockPerGPU, threadPerBlock, g_graph_stream);\
		cudaStreamEndCapture(g_graph_stream, &graph);\
		cudaGraphInstantiate(&g_graph_replay_exec, graph, 0);\
		cudaGraphDestroy(graph);\
	}\
	repeat##DEP(cudaGraphLaunch(g_graph_replay_exec, 0);)\
}

graph_replay_func(1);
graph_replay_func(16);
graph_replay_func(128);

#define DEF_WRAP_LAUNCH_FUNCTION
#endif
