#include "Explicit_Barrier_Kernel.cuh"
#include "../share/util.h" 
#include "../share/measurement.cuh"
#include "../share/wrap_launch_functions.cuh"



//easier to write MACRO here

#define single(result,func,DEP,blockPerGPU,threadPerBlock,A,B,TILE) \
	printf("%s\t%s\t%u\t%u\t%u\t",#func, #DEP, blockPerGPU, threadPerBlock, TILE);\
	measureIntraSMLatency(result,cooperative_launch,func##_DEP##DEP,blockPerGPU,threadPerBlock,A,B,TILE);\
	showRepeatKernelLatency(result[0], clk_result[0],DEP*2);\
	

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

