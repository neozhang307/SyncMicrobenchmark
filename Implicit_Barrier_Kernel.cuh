#include "repeat.h"

//sleep instruction is used to control kernel execution latency. We tried other instructions like FADD, the launch overhead tested would be a little larger than using sleep instruction.
#define SLP asm volatile("nanosleep.u32 1000;");

#define repeat0(S) ;
#define repeat480(S)repeat16(repeat30(S));
#define repeat800(S)repeat16(repeat50(S));
#define repeat1680(S)repeat16(repeat105(S));
#define repeat2560(S)repeat16(repeat160(S));
#define repeat3200(S)repeat16(repeat200(S));

__global__ void null_kernel(){}

#define N_KERNEL(DEP)\
__global__ void null_kernel_##DEP()\
{\
	repeat##DEP(SLP;);\
}

//The kernel should last long enough to study the launch overhead hidden in kernel launch. 
//These kernels are generated for DGX1. 
//1 node
N_KERNEL(5);
	//80
//2 node
N_KERNEL(15);
	N_KERNEL(240);
//3 node
N_KERNEL(30);
	N_KERNEL(480);
//4 node
N_KERNEL(50);
	N_KERNEL(800);
//5 node
N_KERNEL(80);
	N_KERNEL(1280);
//6 node
N_KERNEL(105);
	N_KERNEL(1680);
//7 node
N_KERNEL(160);
	N_KERNEL(2560);
//8 node
N_KERNEL(200);
	N_KERNEL(3200);


//to study the relationship between kernel total latency and execution latency
#define SLP2 asm volatile("nanosleep.u32 100;");

#define N2_KERNEL(DEP)\
__global__ void null_kernel_##DEP()\
{\
	repeat##DEP(SLP2;);\
}

N2_KERNEL(0);
N2_KERNEL(1);
N2_KERNEL(2);
N2_KERNEL(4);
N2_KERNEL(8);
N2_KERNEL(16);
N2_KERNEL(32);
N2_KERNEL(64);
N2_KERNEL(128);

typedef void (*nKernel)();

