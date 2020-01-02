#include "Explicit_Barrier_Kernel.cuh"
#include "../share/util.h" 
// void __forceinline__ cooperative_launch(nKernel func,unsigned int blockPerGPU,unsigned int threadPerBlock, 
// 	unsigned int GPU_count=1, 
// 	cudaLaunchParams *launchParamsList=NULL)
// {
// 	void*KernelArgs[] = {};
// 	cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
// }
//typedef void (*fbaseKernel)
//(float, float,
//double*, unsigned int*,
//unsigned int*, unsigned int);

	// double* d_out, unsigned int*d_time_stamp,
	//float a=2, float b=2, 
	//unsigned int tile=32

	//void*KernelArgs[] = {(void*)&a, 
	// 					(void*)&b,
	// 					(void*)&d_out,
	// 					(void*)&d_time_stamp,
	// 					(void*)&nptr,
	// 					(void*)&tile};
void __forceinline__ cooperative_launch(fbaseKernel func,
	unsigned int blockPerGPU,unsigned int threadPerBlock, void** KernelArgs, 
	unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
{
	cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
}

typedef void(*launchfunction)(fbaseKernel, 
	unsigned int, unsigned int, 
	void**, unsigned int, 
	cudaLaunchParams*);

//1. measure latencys in cycle (single GPU)
//1.1. measure the latency of instructions in the first SM
//1.2. measure the throughput of instructions in the first SM
void GetLatencyOfSM(
						unsigned int& Min_Latency,//count when last thread start sync first thread finish sync
						unsigned int& Max_Latency,//count when first thread start sync last thread finish sync
						unsigned int totalWarpCount, //warp number in a synchronization group
						unsigned int* time_stamp,	//time stamp from kernel, size=groupCount*warpPerGroup
						unsigned int* idx,	//idx from each warp, size=groupCount*warpPerGroup
						unsigned int smid)//the sm to measure
{
	unsigned int basic_index=0;
	if(smid==0)smid=idx[(basic_index)*2];
	while(smid!=idx[(basic_index)*2])basic_index++;
	unsigned int start_min = time_stamp[basic_index*2];
	unsigned int end_min = time_stamp[basic_index*2+1];
	unsigned int start_max = time_stamp[basic_index*2];
	unsigned int end_max = time_stamp[basic_index*2+1];
	basic_index++;
	for(; basic_index < totalWarpCount; basic_index ++)
	{
		if(smid!=idx[(basic_index)*2])continue;
		start_min=max(start_min,time_stamp[(basic_index)*2]);
		end_min=min(end_min,time_stamp[(basic_index)*2+1]);

		start_max=min(start_max,time_stamp[(basic_index)*2]);
		end_max=max(end_max,time_stamp[(basic_index)*2+1]);
	}
	Min_Latency=end_min-start_min;
	Max_Latency=end_max-start_max;
}

#ifndef SIZE
#define SIZE 101
#endif

void measureLatency_cycle(latencys* result, 
	launchfunction run_func, fbaseKernel kernel_func,
	unsigned int blockPerGPU, unsigned int threadPerBlock, 
	float a=2, float b=2, unsigned int tile=32)
{
	{
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
	}
}

//2. measure latencys in ns (involve several SMs) TODO
void prepare_showRepeatKernelLatency()
{
  printf("method\trep\tblk\tthrd\ttile\tm(cycle)\ts(cycle)\tm(ns)\ts(ns)\tm(ave_cycle)\ts(ave_cycle)\n"); 
}

double computeAvgLatCycle(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return (g_result_more.latency_min-g_result_basic.latency_min)/difference;
}
double computeAvgLatCycles(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return sqrt((g_result_more.s_latency_min*g_result_more.s_latency_min+g_result_basic.s_latency_min*g_result_basic.s_latency_min))/difference;
}

#define single(result,func,DEP,blockPerGPU,threadPerBlock,A,B,TILE) \
	printf("%s\t%s\t%u\t%u\t%u\t",#func, #DEP, blockPerGPU, threadPerBlock, TILE);\
	measureLatency_cycle(result,cooperative_launch,func##_DEP##DEP,blockPerGPU,threadPerBlock,A,B,TILE);\
	showlatency_cycle(result[0]);\
	showlatency_ttl(result[0]);\
	printf("%f\t%f\t", computeAvgLatCycle(clk_result[0],result[0],DEP*2),computeAvgLatCycles(clk_result[0],result[0],DEP*2));\
	nxtline();\

#define thorought(result,func,blockPerGPU,threadPerBlock,A,B,TILE) \
	prepare_showRepeatKernelLatency();\
	single(result,func,1,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,2,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,4,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,8,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,16,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,32,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,64,blockPerGPU,threadPerBlock,A,B,TILE);\
	single(result,func,128,blockPerGPU,threadPerBlock,A,B,TILE);


int main(int argc, char **argv)
{
	//coalesced
	fprintf(stderr, "coalesced latency test\n");

	latencys* result  = (latencys*)malloc(2*sizeof(latencys));
	latencys* clk_result = result+1;
	unsigned int block_count=1;
	unsigned int blockDim=32;

	prepare_showRepeatKernelLatency();
	single(clk_result,k_base_kernel_COM_float_DULL,1,block_count,blockDim,2,2,1);


	thorought(result,k_coalesced_kernel_CCOM_float_DULL,block_count,blockDim,2,2,1);
	thorought(result,k_coalesced_kernel_CCOM_float_DULL,block_count,blockDim,2,2,32);


	//TILE size 
	fprintf(stderr, "tile latency test compared with group size\n");

	thorought(result,k_base_kernel_T1COM_float_DULL,block_count,blockDim,2,2,32);
	thorought(result,k_base_kernel_T2COM_float_DULL,block_count,blockDim,2,2,32);
	thorought(result,k_base_kernel_T4COM_float_DULL,block_count,blockDim,2,2,32);
	thorought(result,k_base_kernel_T8COM_float_DULL,block_count,blockDim,2,2,32);
	thorought(result,k_base_kernel_T16COM_float_DULL,block_count,blockDim,2,2,32);
	thorought(result,k_base_kernel_T32COM_float_DULL,block_count,blockDim,2,2,32);

	fprintf(stderr,"coalesced synchronization\n");


	for(int i=1; i<=32; i*=2)
	{
		printf("shuffle data to the %d thread nearby\n",i);
		thorought(result,k_base_kernel_CSHU_float_EQUAL,block_count,blockDim,2,i,32);
		if(i<=1)
		{
			thorought(result,k_base_kernel_T1SHU_float_EQUAL,block_count,blockDim,2,i,32);
		}
		if(i<=2)
		{
			thorought(result,k_base_kernel_T2SHU_float_EQUAL,block_count,blockDim,2,i,32);
		}
		if(i<=4)
		{
			thorought(result,k_base_kernel_T4SHU_float_EQUAL,block_count,blockDim,2,i,32);		
		}
		if(i<=8)
		{
			thorought(result,k_base_kernel_T8SHU_float_EQUAL,block_count,blockDim,2,i,32);
		}
		if(i<=16)
		{
			thorought(result,k_base_kernel_T16SHU_float_EQUAL,block_count,blockDim,2,i,32);
		}
		if(i<=32)
		{
			thorought(result,k_base_kernel_T32SHU_float_EQUAL,block_count,blockDim,2,i,32);
		}
	}

	thorought(result,k_base_kernel_BCOM_float_DULL,block_count,blockDim,2,2,32);

 	thorought(result,k_base_kernel_GCOM_float_DULL,block_count,blockDim,2,2,32);
 	thorought(result,k_base_kernel_MGCOM_float_DULL,block_count,blockDim,2,2,32);

	free(result);

}

