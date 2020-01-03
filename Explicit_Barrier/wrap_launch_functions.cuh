
typedef void(*launchfunction)(fbaseKernel, 
	unsigned int, unsigned int, 
	void**, unsigned int, 
	cudaLaunchParams*);

typedef void (*nKernel)();
typedef void (*fbaseKernel)(float,float,double*,unsigned int*,unsigned int*, unsigned int);

#ifndef DEF_WRAP_LAUNCH_FUNCTION
void __forceinline__ cooperative_launch(fbaseKernel func,
	unsigned int blockPerGPU,unsigned int threadPerBlock, void** KernelArgs, 
	unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
}
void __forceinline__ traditional_launch(fbaseKernel func,
	unsigned int blockPerGPU,unsigned int threadPerBlock, void** KernelArgs, 
	unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	func<<<blockPerGPU,threadPerBlock>>>(((float*)KernelArgs[0])[0],((float*)KernelArgs[1])[0],
		((double**)KernelArgs[2])[0],((unsigned int**)KernelArgs[3])[0],
		((unsigned int**)KernelArgs[4])[0],(( unsigned int*)KernelArgs[5])[0]);
}

void __forceinline__ multi_cooperative_launch(fbaseKernel func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, void** KernelArgs,
	unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	cudaLaunchCooperativeKernelMultiDevice(launchParamsList, GPU_count);
}

#define DEF_WRAP_LAUNCH_FUNCTION
#endif
// #ifndef DEF_WRAP_LAUNCH_FUNCTION
// typedef void(*launchfunction)(nKernel, unsigned int, unsigned int, unsigned int, cudaLaunchParams*);

// void __forceinline__ traditional_launch(nKernel func, unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
// {
// 	func<<<blockPerGPU,threadPerBlock>>>();
// }

// void __forceinline__ cooperative_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
// {
// 	void*KernelArgs[] = {};
// 	cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
// }

// void __forceinline__ multi_cooperative_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
// {
// 	cudaLaunchCooperativeKernelMultiDevice(launchParamsList, GPU_count);
// }

// #define repeatlaunch(fname, DEP) \
// void __forceinline__ fname##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)\
// {\
// 	repeat##DEP(fname(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList););\
// }

// #define gencallfun(callfunc) \
// 	repeatlaunch(callfunc,1); \
// 	repeatlaunch(callfunc,16); \
// 	repeatlaunch(callfunc,128); \

// gencallfun(traditional_launch);
// gencallfun(cooperative_launch);
// gencallfun(multi_cooperative_launch);

// //only for single GPU
// #include<omp.h>

// //The way to repeat omp function is a little different here. Because I found I can not include #pragma statement in MACRO
// void __forceinline__ single_omp_traditional_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL, unsigned int repeat=2)
// {
// 	#pragma omp parallel num_threads(GPU_count)
// 	{
// 		/* code */
// 		int tid = omp_get_thread_num();
// 		cudaSetDevice(tid);
// 		for(int j=0; j<repeat; j++)
// 		{
// 		func<<<blockPerGPU,threadPerBlock,0,launchParamsList[tid].stream>>>();
// 			cudaDeviceSynchronize();
// 			#pragma omp barrier
// 		}
// 	}
	
// }

// //A little different here
// #define repeatomp(DEP) \
// void __forceinline__ omp_traditional_launch##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL)\
// {\
// 	single_omp_traditional_launch(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList,DEP);\
// }

// repeatomp(1);
// repeatomp(16);
// repeatomp(128); 

// #define DEF_WRAP_LAUNCH_FUNCTION
// #endif