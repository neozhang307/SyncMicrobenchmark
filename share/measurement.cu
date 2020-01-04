
#include"measurement.cuh"
#include "wrap_launch_functions.cuh"
#include "../share/util.h"

#include <stdio.h>


//1. measure latencys in cycle (single GPU)
//1.1. measure the latency of instructions in the first SM
//1.2. measure the throughput of instructions in the first SM
int measureIntraSMLatency(latencys* result, 
	launchfunction_rkernel run_func, fbaseKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, 
	float a, float b, unsigned int tile)
{
	{
		int errorcode=1;
		cudaError_t e;

		double * d_out;
		unsigned int totalThreadsPerGPU = blockPerGPU*threadPerBlock;
		cudaMalloc((void **)&d_out, sizeof(double) * totalThreadsPerGPU*1);	
		unsigned int warp_count=blockPerGPU*threadPerBlock/32;

	 	unsigned int * h_time_stamp = (unsigned int*)malloc(sizeof(unsigned int)*warp_count*2); 
	 	unsigned int * d_time_stamp;
	 	cudaMalloc((void**)&d_time_stamp, sizeof(unsigned int)*warp_count*2);

	 	unsigned int * h_idx = (unsigned int*)malloc(sizeof(unsigned int)*warp_count*2); 
	 	unsigned int * d_idx;
	 	cudaMalloc((void**)&d_idx, sizeof(unsigned int)*warp_count*2);

		void*KernelArgs[] = {(void*)&a, 
						(void*)&b,
						(void*)&d_out,
						(void*)&d_time_stamp,
						(void*)&d_idx,
						(void*)&tile};

		cudaCheckError(); 

		timespec tsstart,tsend;
		long time_elapsed_ns_lat ;
		double latency_lat[SIZE];
		double latency_max[SIZE];
		double latency_min[SIZE];
		unsigned int ulatency_max;
		unsigned int ulatency_min;
		for(int i=0; i<SIZE; i++)
		{
			//clock
			clock_gettime(CLOCK_REALTIME, &tsstart);
			//launch
			run_func(kernel_func,blockPerGPU,threadPerBlock,KernelArgs,1,NULL);
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_REALTIME, &tsend);
			//execution

			cudaMemcpy(h_time_stamp, d_time_stamp, sizeof(unsigned int)*warp_count*2, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_idx, d_idx, sizeof(unsigned int)*warp_count*2, cudaMemcpyDeviceToHost);

	 		time_elapsed_ns_lat = (tsend.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns_lat += 1000000000*(tsend.tv_sec-tsstart.tv_sec);
	 		latency_lat[i]=time_elapsed_ns_lat;

	 		GetLatencyOfSM(ulatency_min,ulatency_max,warp_count,h_time_stamp,h_idx,0);
	 		latency_min[i]=ulatency_min;
	 		latency_max[i]=ulatency_max;
			e=cudaGetLastError();                                 
	 		if(e!=cudaSuccess) {                                              
	   			fprintf(stderr,"Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); 
				errorcode=-1;
				break;
	 		}
		}
		cudaCheckError();

		getStatistics(result->mean_lat, result->s_lat, latency_lat+1, SIZE-1);
		getStatistics(result->latency_min, result->s_latency_min, latency_min+1, SIZE-1);
		getStatistics(result->latency_max, result->s_latency_max, latency_max+1, SIZE-1);

		cudaFree(d_out);
		cudaFree(d_time_stamp);
		cudaFree(d_idx);
		free(h_time_stamp);
		free(h_idx);

		// e=cudaGetLastError();                                 
 	// 	if(e!=cudaSuccess) {                                              
  //  			fprintf(stderr,"Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); 
		// 	errorcode=-1;
 	// 	}

 		if(errorcode!=1)
 		{
	 		cudaDeviceReset();
			return -1;
 		}

		return 1;
	}
}
//2. measure latencys in ns (involve several SMs) TODO
int measureInterSMLatency(latencys* result, 
	launchfunction_rkernel run_func, fbaseKernel kernel_func, 
	unsigned int gpu_count,
	unsigned int blockPerGPU, unsigned int threadPerBlock)
{
	{
		int errorcode=1;
		cudaError_t e;

		cudaStream_t *mstream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*gpu_count);
		void***packedKernelArgs = (void***)malloc(sizeof(void**)*gpu_count); 
		cudaLaunchParams *launchParamsList = (cudaLaunchParams *)malloc(
      		sizeof(cudaLaunchParams)*gpu_count);

		float a=2;
		float b=2;
		double **d_out = (double**)malloc(sizeof(double)*gpu_count);
		unsigned int* nptr=NULL;
		unsigned int tile=32;

		for(int deviceid=0; deviceid<gpu_count;deviceid++)
		{
			cudaSetDevice(deviceid);
			packedKernelArgs[deviceid]=(void**)malloc(sizeof(void*)*6);

			cudaStreamCreate(&mstream[deviceid]);

			cudaCheckError();
			cudaMalloc((void**)&d_out[deviceid], sizeof(double));
			packedKernelArgs[deviceid][0]=(void*)&a;
			packedKernelArgs[deviceid][1]=(void*)&b;
			packedKernelArgs[deviceid][2]=(void*)&d_out[deviceid];
			packedKernelArgs[deviceid][3]=(void*)&nptr;
			packedKernelArgs[deviceid][4]=(void*)&nptr;
			packedKernelArgs[deviceid][5]=(void*)&tile;
			
			launchParamsList[deviceid].func=(void*)kernel_func;
			launchParamsList[deviceid].gridDim=blockPerGPU;
			launchParamsList[deviceid].blockDim=threadPerBlock;
			launchParamsList[deviceid].sharedMem=32;
			launchParamsList[deviceid].stream=mstream[deviceid];
			launchParamsList[deviceid].args=packedKernelArgs[deviceid];
		}
		cudaCheckError(); 

		timespec tsstart,tsendop;
		long time_elapsed_ns ;
		double latency_lat[SIZE];
		
		for(int i=0; i<SIZE; i++)
		{

			clock_gettime(CLOCK_REALTIME, &tsstart);
			run_func(kernel_func,blockPerGPU,threadPerBlock,NULL, gpu_count,launchParamsList);
			for(int deviceid=0; deviceid<gpu_count; deviceid++)
			{
				cudaSetDevice(deviceid);
				cudaDeviceSynchronize();
				cudaStreamSynchronize(mstream[deviceid]);
			}
 			clock_gettime(CLOCK_REALTIME, &tsendop);

	 		//latencys of total kernel total latency (after sync)
			time_elapsed_ns = (tsendop.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns += 1000000000*(tsendop.tv_sec-tsstart.tv_sec);
	 		latency_lat[i]=time_elapsed_ns;
	 		e=cudaGetLastError();                                 
 			if(e!=cudaSuccess) {                                              
	   			fprintf(stderr,"Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); 
	   			for(int deviceid=0; deviceid<gpu_count;deviceid++)
				{
					cudaSetDevice(deviceid);	
		 			cudaDeviceReset();
				}
				errorcode=-1;
				break;
	 		}

		}
		// cudaCheckError();
		getStatistics(result->mean_lat, result->s_lat, latency_lat+1, SIZE-1);

		for(int deviceid=0; deviceid<gpu_count;deviceid++)
		{
			cudaSetDevice(deviceid);	
 			cudaStreamDestroy(mstream[deviceid]);
			cudaFree(d_out[deviceid]);
		}

		free(mstream);
		free(packedKernelArgs);
		free(launchParamsList);
		free(d_out);
		// e=cudaGetLastError();                                 
 	// 	if(e!=cudaSuccess) {                                              
  //  			fprintf(stderr,"Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); 
		// 	errorcode=-1;
 	// 	}
 		if(errorcode!=1)
 		{
 			for(int deviceid=0; deviceid<gpu_count;deviceid++)
			{
				cudaSetDevice(deviceid);	
	 			cudaDeviceReset();
			}
			return -1;
 		}
 		return 1;
	}
}
//3. mearsure kernel latency 

template <int gpu_count>
void measureKernelLatency(latencys* result, 
	launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock)
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
		long time_elapsed_ns;
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
	 		time_elapsed_ns = (tsstart.tv_nsec-ini.tv_nsec);
	 		time_elapsed_ns += 1000000000*(tsstart.tv_sec-ini.tv_sec);
	 		latency_clock[i]=time_elapsed_ns;

	 		//latencys of launch functions (no sync here)
	 		time_elapsed_ns = (tsend.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns += 1000000000*(tsend.tv_sec-tsstart.tv_sec);
	 		latency_laun[i]=time_elapsed_ns;

	 		//latencys of total kernel total latency (after sync)
			time_elapsed_ns = (tsendop.tv_nsec-tsstart.tv_nsec);
	 		time_elapsed_ns += 1000000000*(tsendop.tv_sec-tsstart.tv_sec);
	 		latency_lat[i]=time_elapsed_ns;

	 		//latencys of synchronization functions 
			time_elapsed_ns = (tsendsync.tv_nsec-tsendop.tv_nsec);
	 		time_elapsed_ns += 1000000000*(tsendsync.tv_sec-tsendop.tv_sec);
	 		latency_syncfunc[i]=time_elapsed_ns;
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

template void measureKernelLatency<1>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<2>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<3>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<4>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<5>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<6>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<7>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
template void measureKernelLatency<8>(latencys* result, launchfunction_nkernel run_func, nKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock);
