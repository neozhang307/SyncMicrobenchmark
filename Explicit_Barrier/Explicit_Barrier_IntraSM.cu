#include "Explicit_Barrier_Kernel.cuh"

#include "../share/util.h"
#include "../share/measurement.cuh"


void benchmarkLatencyInSingleSM(fbaseKernel kernel, const char* kernelname)
{
	latencys result;
	latencys basic;//no need to test clk here, because repeat times is large enough
	basic.latency_min=0;
	basic.s_latency_min=0;
	basic.latency_max=0;
	basic.s_latency_max=0;
	unsigned int blockPerGPU=1;
	prepare_showLatencyInSingleSM();
	for(unsigned int threadPerBlock=32; threadPerBlock<=1024; threadPerBlock*=2)
	{
		measureIntraSMLatency(&result,cooperative_launch,kernel,blockPerGPU,threadPerBlock,2,2,32);
		showLatencyInSingleSM(basic, result, kernelname, 1, 128, blockPerGPU, threadPerBlock);
	}
}


void benchmarkThroughputInSingleSM(fbaseKernel kernel, const char* kernelname,
	unsigned int smx_count, unsigned int uplimit)
{
	latencys result;
	//
	prepare_showThroughputInSingleSM();
	for(unsigned int basic=1; basic<=uplimit; basic*=2)
	{
		unsigned int blockPerGPU=smx_count*basic;
		for(unsigned int threadPerBlock=32; threadPerBlock<=1024; threadPerBlock*=2)
		{
			measureIntraSMLatency(&result,traditional_launch,kernel,blockPerGPU,threadPerBlock,2,2,32);
			showThroughputInSingleSM(result, kernelname, 
					1, 128, blockPerGPU, threadPerBlock, smx_count,32);
		}
	}
}




int main(int argc, char **argv)
{
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
  	cudaGetDeviceProperties(&deviceProp, 0);
  	unsigned int	smx_count = deviceProp.multiProcessorCount;
	//test the latency of block through all possible group size
	benchmarkLatencyInSingleSM(k_base_kernel_BCOM_float_DULL_DEP128,"blocksync");
	benchmarkLatencyInSingleSM(k_base_kernel_T32SHU_float_EQUAL_DEP128,"tile_shufl");
	//test the throughput of warp and block through all possible group size
	//warp level
	benchmarkThroughputInSingleSM(k_base_kernel_T32COM_float_DULL_DEP128,"tile_sync",
		smx_count, 64);
	benchmarkThroughputInSingleSM(k_base_kernel_T32SHU_float_EQUAL_DEP128,"tile_shufl",
		smx_count, 64);
	benchmarkThroughputInSingleSM(k_coalesced_kernel_CCOM_float_DULL_DEP128,"coales_sync",
		smx_count, 64);
	benchmarkThroughputInSingleSM(k_base_kernel_CSHU_float_EQUAL_DEP128,"coales_shufl",
		smx_count, 64);
	//block level
	benchmarkThroughputInSingleSM(k_base_kernel_BCOM_float_DULL_DEP128,"blocksync",
		smx_count, 64);
	//test the latency of grid sync// write new measurement code 
}
