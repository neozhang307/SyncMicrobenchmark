#include "repeat.h"
#include"cuda_runtime.h"


//sleep instruction is used to control kernel execution latency. We tried other instructions like FADD, the launch overhead tested would be a little larger than using sleep instruction.
#define SLP asm volatile("nanosleep.u32 1000;");

#ifndef repeat0
#define repeat0(S) ;
#endif

#ifndef repeat480
#define repeat480(S)repeat16(repeat30(S));
#endif

#ifndef repeat800
#define repeat800(S)repeat16(repeat50(S));
#endif

#ifndef repeat1680
#define repeat1680(S)repeat16(repeat105(S));
#endif

#ifndef repeat2560
#define repeat2560(S)repeat16(repeat160(S));
#endif

#ifndef repeat3200
#define repeat3200(S)repeat16(repeat200(S));
#endif


#define N_KERNEL(DEP)\
__global__ void null_kernel_##DEP()\
{\
	repeat##DEP(SLP;);\
}

//The kernel should last long enough to study the launch overhead hidden in kernel launch. 
//These kernels are generated for DGX1. 
//1 node
#define DEC_N_KERNEL(DEP) __global__ void null_kernel_##DEP();

DEC_N_KERNEL(5);
	//80
//2 node
DEC_N_KERNEL(15);
	DEC_N_KERNEL(240);
//3 node
DEC_N_KERNEL(30);
	DEC_N_KERNEL(480);
//4 node
DEC_N_KERNEL(50);
	DEC_N_KERNEL(800);
//5 node
DEC_N_KERNEL(80);
	DEC_N_KERNEL(1280);
//6 node
DEC_N_KERNEL(105);
	DEC_N_KERNEL(1680);
//7 node
DEC_N_KERNEL(160);
	DEC_N_KERNEL(2560);
//8 node
DEC_N_KERNEL(200);
	DEC_N_KERNEL(3200);


//to study the relationship between kernel total latency and execution latency
#define SLP2 asm volatile("nanosleep.u32 100;");

#define N2_KERNEL(DEP)\
__global__ void null2_kernel_##DEP()\
{\
	repeat##DEP(SLP2;);\
}

#define DEC_N2_KERNEL(DEP) __global__ void null2_kernel_##DEP();

DEC_N2_KERNEL(0);
DEC_N2_KERNEL(1);
DEC_N2_KERNEL(2);
DEC_N2_KERNEL(4);
DEC_N2_KERNEL(8);
DEC_N2_KERNEL(16);
DEC_N2_KERNEL(32);
DEC_N2_KERNEL(64);
DEC_N2_KERNEL(128);

__global__ void null_kernel();

typedef void (*nKernel)();

