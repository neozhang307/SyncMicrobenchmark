#include "Explicit_Barrier_Kernel.cuh"

#include "../share/util.h"
#include "../share/measurement.cuh"



void benchmarkLatencyInterSM(launchfunction_rkernel run_func, 
							fbaseKernel basic_kernel, fbaseKernel more_kernel, 
							const char* kernelname,
							unsigned int basic_dec, unsigned int more_dec,
							unsigned int smx_count, unsigned int uplimit,
							unsigned int gpu_count)
{
	latencys result_basic;
	latencys result_more;
	//
	//prepare_showThroughputInSingleSM();
	int errorcode;

	//warmup call
	errorcode=measureInterSMLatency(&result_basic, 
				run_func, basic_kernel, 
				1,
				1, 32);

	errorcode=measureInterSMLatency(&result_more, 
				run_func, more_kernel, 
				1,
				1, 32);

	prepare_showLatencyInterSM();
	for(unsigned int basic=1; basic<=uplimit; basic*=2)
	{
		unsigned int blockPerGPU=smx_count*basic;
		for(unsigned int threadPerBlock=32; threadPerBlock<=1024; threadPerBlock*=2)
		{
			if(basic*threadPerBlock>2048)continue;//or can not launch the kernel
			errorcode=measureInterSMLatency(&result_basic, 
				run_func, basic_kernel, 
				gpu_count,
				blockPerGPU, threadPerBlock);
			if(errorcode<0)continue;
			errorcode=measureInterSMLatency(&result_more, 
				run_func, more_kernel, 
				gpu_count,
				blockPerGPU, threadPerBlock);
			if(errorcode<0)continue;
			showLatencyInterSM(result_basic, result_more, kernelname, 
							basic_dec, more_dec,
							gpu_count, blockPerGPU, threadPerBlock);



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
	benchmarkLatencyInterSM(cooperative_launch, 
							k_base_kernel_GCOM_float_DULL_DEP256, k_base_kernel_GCOM_float_DULL_DEP2816, 
							"grid_sync",
							256, 2816,
							smx_count, 32,
							1);
	int gpu_count=1;
	if(argc>=2)
	{
		gpu_count=(int)ToUInt(argv[1]);
	}
	for(int i=1; i<=gpu_count; i++)
	{
		benchmarkLatencyInterSM(multi_cooperative_launch, 
							k_base_kernel_MGCOM_float_DULL_DEP256, k_base_kernel_GCOM_float_DULL_DEP2816, 
							"multi_grid_sync",
							256, 2816,
							smx_count, 32,
							1);
	}
}
