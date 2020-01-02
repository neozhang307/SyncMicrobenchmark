#include "../share/repeat.h"
#include "../share/repeat.h"

#include"cuda_runtime.h"
#include<stdio.h>
#include<math.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARP 32

//Basic Computation 
#define SLEEP(a,b) asm volatile("nanosleep.u32 1000;");
#define AFADD(a,b) asm volatile("add.f32 %0, %0, %1;":"+f"(a):"f"(b));
#define EQUAL(a,b) a=b;
#define DULL(a,b) ;


//Warp level iterate through all possible group size
#define TCOM(OP,a,b) OP(b,a); sync(tg); OP(a,b); sync(tg);
#define T1COM(OP,a,b) OP(b,a); sync(tg1); OP(a,b); sync(tg1);
#define T2COM(OP,a,b) OP(b,a); sync(tg2); OP(a,b); sync(tg2);
#define T4COM(OP,a,b) OP(b,a); sync(tg4); OP(a,b); sync(tg4);
#define T8COM(OP,a,b) OP(b,a); sync(tg8); OP(a,b); sync(tg8);
#define T16COM(OP,a,b) OP(b,a); sync(tg16); OP(a,b); sync(tg16);
#define T32COM(OP,a,b) OP(b,a); sync(tg32); OP(a,b); sync(tg32);
#define CCOM(OP,a,b) OP(b,a); sync(csg); OP(a,b); sync(csg);

#define T1SHU(OP,a,b) OP(a,tg1.shfl_down(a,b)); OP(a,tg1.shfl_down(a,b));
#define T2SHU(OP,a,b) OP(a,tg2.shfl_down(a,b)); OP(a,tg2.shfl_down(a,b));
#define T4SHU(OP,a,b) OP(a,tg4.shfl_down(a,b)); OP(a,tg4.shfl_down(a,b));
#define T8SHU(OP,a,b) OP(a,tg8.shfl_down(a,b)); OP(a,tg8.shfl_down(a,b));
#define T16SHU(OP,a,b) OP(a,tg16.shfl_down(a,b)); OP(a,tg16.shfl_down(a,b));
#define T32SHU(OP,a,b) OP(a,tg32.shfl_down(a,b)); OP(a,tg32.shfl_down(a,b));
#define CSHU(OP,a,b) OP(a,csg.shfl_down(a,b)); OP(a,csg.shfl_down(a,b));

//block level 
#define BCOM(OP,a,b) OP(b,a); sync(bg); OP(a,b); sync(bg); 

//grid level
#define GCOM(OP,a,b) OP(b,a); sync(gg); OP(a,b); sync(gg);
#define MGCOM(OP,a,b) OP(b,a); sync(mgg); OP(a,b); sync(mgg);
#define COM(OP,a,b) OP(b,a); OP(a,b);


#define BASE_KERNEL(COM,TYPE,OP,DEP)\
__global__ void k_base_kernel_##COM##_##TYPE##_##OP##_DEP##DEP (TYPE a, TYPE b, double *out, unsigned int *time_stamp=NULL, unsigned int*idx=NULL, unsigned int tile=32){\
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;\
	thread_group tg = tiled_partition(this_thread_block(), tile);\
	thread_block_tile<1> tg1 = tiled_partition<1>(this_thread_block());\
	thread_block_tile<2> tg2 = tiled_partition<2>(this_thread_block());\
	thread_block_tile<4> tg4 = tiled_partition<4>(this_thread_block());\
	thread_block_tile<8> tg8 = tiled_partition<8>(this_thread_block());\
	thread_block_tile<16> tg16 = tiled_partition<16>(this_thread_block());\
	thread_block_tile<32> tg32 = tiled_partition<32>(this_thread_block());\
	coalesced_group csg = coalesced_threads();\
\
	thread_group bg = this_thread_block();\
	grid_group gg = this_grid();\
	multi_grid_group mgg = this_multi_grid();\
	unsigned int  start,end;\
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");\
	repeat##DEP(COM(OP,a,b));\
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");\
	unsigned int warp_id = id/WARP;\
	if(id%WARP==0)\
	{\
		if(NULL!=time_stamp){\
			time_stamp[warp_id*2]=start;\
			time_stamp[warp_id*2+1]=end;\
		}\
		if(NULL!=idx){\
			idx[warp_id*2]=get_smid();\
			idx[warp_id*2+1]=blockIdx.x;\
		}\
	}\
	out[id]=(double)a;\
}

#define COA_KERNEL(COM,TYPE,OP,DEP)\
__global__ void k_coalesced_kernel_##COM##_##TYPE##_##OP##_DEP##DEP (TYPE a, TYPE b, double *out, unsigned int *time_stamp=NULL,  unsigned int*idx=NULL, unsigned int tile=32){\
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;\
	thread_group bg = this_thread_block();\
	unsigned int start=0;\
	unsigned int end=0;\
	if(id%WARP<=tile)\
	{\
		coalesced_group csg = coalesced_threads();\
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");\
		repeat##DEP(COM(OP,a,b));\
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");\
	}\
	unsigned int warp_id = threadIdx.x/WARP + ((blockDim.x-1)/WARP+1)*blockIdx.x;\
	if(id%WARP==0)\
	{\
		if(NULL!=time_stamp){\
			time_stamp[warp_id*2]=start;\
			time_stamp[warp_id*2+1]=end;\
		}\
		if(NULL!=idx){\
			idx[warp_id*2]=get_smid();\
			idx[warp_id*2+1]=blockIdx.x;\
		}\
	}\
	out[id]=(double)a;\
}

//Declear of kernel function:
typedef void (*fbaseKernel)(float,float,double*,unsigned int*,unsigned int*, unsigned int);

//Declear of all types of kernels
#define FUNC_DDEP(FUNC,type) \
	__global__ void FUNC##1(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##2(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##4(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##8(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##16(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##32(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##64(type,type,double*,unsigned int*,unsigned int*, unsigned int);\
	__global__ void FUNC##128(type,type,double*,unsigned int*,unsigned int*, unsigned int);\


//warp level
FUNC_DDEP(k_base_kernel_TCOM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T1COM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T2COM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T4COM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T8COM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T16COM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_T32COM_float_DULL_DEP,float);

FUNC_DDEP(k_coalesced_kernel_CCOM_float_DULL_DEP,float);//coalesce sync

FUNC_DDEP(k_base_kernel_TSHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T1SHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T2SHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T4SHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T8SHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T16SHU_float_EQUAL_DEP,float);
FUNC_DDEP(k_base_kernel_T32SHU_float_EQUAL_DEP,float);

FUNC_DDEP(k_base_kernel_CSHU_float_EQUAL_DEP,float);//coalesce sync
//block level
FUNC_DDEP(k_base_kernel_BCOM_float_DULL_DEP,float);

//grid level
FUNC_DDEP(k_base_kernel_GCOM_float_DULL_DEP,float);
FUNC_DDEP(k_base_kernel_MGCOM_float_DULL_DEP,float);//multi grid sync

//null used for test gpu cycle latency
__global__ void k_base_kernel_COM_float_DULL_DEP1(float,float,double*,unsigned int*,unsigned int*, unsigned int);

