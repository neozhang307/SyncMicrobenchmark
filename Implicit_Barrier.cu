#include "repeat.h"
#include "Implicit_Barrier_Kernel.cuh"
#include "statistics.h"
#include "util.h"
#include <stdio.h>
/*
Definition:
* {Kernel Execution Latency:} Total time spent in executing the kernel, excluding any overhead for launching the kernel.
* {Launch Overhead:} Latency that is not related to kernel execution. 
* {Kernel Total Latency:} Total latency to run kernels.
Depends on different situation, the launch overhead could be different:
Situation 1: Launch a single kernel
Situation 2: Launch additional "small kenel" (By "small" we mean the device is not saturate at all, in this experiment in single GPU if each kernel lasts less then 5us it is defined as "small")
Situation 3: Launch additional "big kernel" (By "big" we mean the device is saturate enough while the workload is not severe, in this experiment in single GPU if each kernel lasts longer than 5us, it is defined as "big")

When kernels are "small" or each kernel lasts fewer than 5us, it would not be practical to offload these workloads to GPU at all. So, we only include the launch overhead of "big kernel" in our IPDPS20 paper.

But in this microbenchmark, we include the measurements of launch overhead in all three situations. The detailed information about these measurements are explained in an ICPP19 Poster in this same folder. 

*/


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

//DEP define repeat a function how many times.
#define repeatlaunch(fname, DEP) \
void __forceinline__ fname##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)\
{\
	repeat##DEP(fname(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList););\
}

#define gencallfun(callfunc) \
	repeatlaunch(callfunc,1); \
	repeatlaunch(callfunc,2); \
	repeatlaunch(callfunc,4); \
	repeatlaunch(callfunc,8); \
	repeatlaunch(callfunc,16); \
	repeatlaunch(callfunc,32); \
	repeatlaunch(callfunc,64); \
	repeatlaunch(callfunc,128); \

gencallfun(traditional_launch);
gencallfun(cooperative_launch);
gencallfun(multi_cooperative_launch);

//only for single GPU
#include<omp.h>

//The way to repeat omp function is a little different here. Because I found I can not include #pragma statement in MACRO
void __forceinline__ omp_traditional_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL, unsigned int repeat=2)
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
void __forceinline__ repeat_omp##_##DEP(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, unsigned int GPU_count=2, cudaLaunchParams *launchParamsList=NULL)\
{\
	omp_traditional_launch(func,blockPerGPU,threadPerBlock,GPU_count,launchParamsList,DEP);\
}


repeatomp(1);
repeatomp(2);
repeatomp(4); 
repeatomp(8); 
repeatomp(16); 
repeatomp(32); 
repeatomp(64); 
repeatomp(128); 

#define SIZE 101

 //Test Function to get latency results
struct latencys
{
	double mean_laun;
	double s_laun;
	double mean_clk;
	double s_clk;
	double mean_lat;
	double s_lat;
	double mean_sync;
	double s_sync;
};

void showlatency(latencys* g_result)
{
	double*c_result = (double*)g_result;
	for(int j=0; j<8; j++)
	{
		printf("%f\t",c_result[j]);
	}
	printf("\n");
}

typedef void(*launchfunction)(nKernel, unsigned int, unsigned int, unsigned int, cudaLaunchParams*);
template <int gpu_count>
void getResult(latencys* result, launchfunction run_func, nKernel kernel_func,
	unsigned int blockPerGPU=1, unsigned int threadPerBlock=32)
{
	{
		cudaStream_t *mstream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*gpu_count);
		void***packedKernelArgs = (void***)malloc(sizeof(void**)*gpu_count); 
		cudaLaunchParams *launchParamsList = (cudaLaunchParams *)malloc(
      		sizeof(cudaLaunchParams)*gpu_count);

		for(int deviceid=0; deviceid<gpu_count;deviceid++)
		{
			cudaSetDevice(deviceid);
			packedKernelArgs[deviceid]=(void**)malloc(sizeof(void*));

			cudaStreamCreate(&mstream[deviceid]);

			cudaCheckError();

			packedKernelArgs[deviceid][0]=NULL;
			
			launchParamsList[deviceid].func=(void*)kernel_func;
			launchParamsList[deviceid].gridDim=blockPerGPU;
			launchParamsList[deviceid].blockDim=threadPerBlock;
			launchParamsList[deviceid].sharedMem=32;
			launchParamsList[deviceid].stream=mstream[deviceid];
			launchParamsList[deviceid].args=packedKernelArgs[deviceid];
		}
		cudaCheckError(); 

		timespec ini,tsstart,tsend,tsendop,tsendsync;
		long time_elapsed_ns_laun, time_elapsed_ns_lat ;
		double latency_laun[SIZE];
		double latency_lat[SIZE];
		double latency_clock[SIZE];
		double latency_syncfunc[SIZE];
		
		for(int i=0; i<SIZE; i++)
		{
			//clock
			clock_gettime(CLOCK_REALTIME, &ini);
			clock_gettime(CLOCK_REALTIME, &tsstart);
			//launch
			run_func(kernel_func,blockPerGPU,threadPerBlock,gpu_count,launchParamsList);
			clock_gettime(CLOCK_REALTIME, &tsend);
			//execution
			if(gpu_count==0)
			{
				cudaDeviceSynchronize();
			}
			else
			{
				for(int deviceid=0; deviceid<gpu_count; deviceid++)
				{
					cudaSetDevice(deviceid);
					cudaDeviceSynchronize();
					cudaStreamSynchronize(mstream[deviceid]);
				}
			}
 			clock_gettime(CLOCK_REALTIME, &tsendop);
			if(gpu_count==0)
			{
				cudaDeviceSynchronize();
			}
			else
			{
				for(int deviceid=0; deviceid<gpu_count; deviceid++)
				{
					cudaSetDevice(deviceid);
					cudaDeviceSynchronize();
					cudaStreamSynchronize(mstream[deviceid]);
				}
			}
 			clock_gettime(CLOCK_REALTIME, &tsendsync);
 			//latencys of clock function
	 		time_elapsed_ns_laun = (tsstart.tv_nsec-ini.tv_nsec);
	 		time_elapsed_ns_laun += 1000000000*(tsstart.tv_sec-ini.tv_sec);
	 		latency_clock[i]=time_elapsed_ns_laun;

	 		//latencys of launch functions (no sync here)
	 		time_elapsed_ns_laun = (tsend.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns_laun += 1000000000*(tsend.tv_sec-tsstart.tv_sec);
	 		latency_laun[i]=time_elapsed_ns_laun;

	 		//latencys of total kernel total latency (after sync)
			time_elapsed_ns_lat = (tsendop.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns_lat += 1000000000*(tsendop.tv_sec-tsstart.tv_sec);
	 		latency_lat[i]=time_elapsed_ns_lat;

	 		//latencys of synchronization functions 
			time_elapsed_ns_lat = (tsendsync.tv_nsec-tsendop.tv_nsec);
	 		time_elapsed_ns_lat += 1000000000*(tsendsync.tv_sec-tsendop.tv_sec);
	 		latency_syncfunc[i]=time_elapsed_ns_lat;
		}
		cudaCheckError();

		getStatistics(result->mean_laun, result->s_laun, latency_laun+1, SIZE-1);
		getStatistics(result->mean_clk, result->s_clk, latency_clock+1, SIZE-1);
		getStatistics(result->mean_lat, result->s_lat, latency_lat+1, SIZE-1);
		getStatistics(result->mean_sync, result->s_sync, latency_syncfunc+1, SIZE-1);

		for(int deviceid=0; deviceid<gpu_count;deviceid++)
		{
			cudaSetDevice(deviceid);	
			cudaCheckError();
			cudaStreamDestroy(mstream[deviceid]);
		}

		free(mstream);
		free(packedKernelArgs);
		free(launchParamsList);
	}
}


int main(int argc, char **argv)
{


	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();
   	unsigned int smx_count = deviceProp.multiProcessorCount;
//	double* result=(double*)malloc(sizeof(double)*6*8);
	latencys* result  = (latencys*)malloc(sizeof(latencys));
	//show how total latency is influenced by execution (traditional launch)


	//merge this two situation together
	//launch single null kernel and different features
	//launch additional null kernel and compute the kernel overhead here
	printf("Empty Kernel\n");
	printf("When Calling count is one, the result of total latency\n");
	printf("************traditional_launch***********\n");
	getResult<1>(result, traditional_launch_1, null_kernel,smx_count,1024);
	showlatency(result);
	getResult<1>(result, traditional_launch_128, null_kernel,smx_count,1024);
	showlatency(result);
	printf("************cooperative_launch***********\n");
	getResult<1>(result, cooperative_launch_1, null_kernel,smx_count,1024);
	showlatency(result);
	getResult<1>(result, cooperative_launch_128, null_kernel,smx_count,1024);
	showlatency(result);
	printf("************multi_cooperative_launch***********\n");
	getResult<1>(result, multi_cooperative_launch_1, null_kernel,smx_count,1024);
	showlatency(result);
	getResult<1>(result, multi_cooperative_launch_128, null_kernel,smx_count,1024);
	showlatency(result);
	//launch big kernel and additional big kernel to compute the kernel overhead


}



