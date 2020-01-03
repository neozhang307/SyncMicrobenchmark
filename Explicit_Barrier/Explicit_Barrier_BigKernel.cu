#include "Explicit_Barrier_Kernel.cuh"
#include "../share/repeat.h"

#ifndef repeat2816
#define repeat2816(s) repeat11(repeat256(s););
#endif

/*
WHY 256 as a bsic, for grid sync, avg 1us, 256 means 512 calls, last longer than 512us.
In this situation, even 8 GPUs can use the same code to test.
*/
#define MIDDLEKERNEL_DDEP(BASE,COM,TYPE,OP)\
	BASE##_KERNEL(COM,TYPE,OP,256)\

#define BIGKERNEL_DDEP(BASE,COM,TYPE,OP)\
	MIDDLEKERNEL_DDEP(BASE,COM,TYPE,OP)\
	BASE##_KERNEL(COM,TYPE,OP,2816)\

//block level
MIDDLEKERNEL_DDEP(BASE,BCOM,float,DULL)
//grid level
BIGKERNEL_DDEP(BASE,MGCOM,float,DULL)
BIGKERNEL_DDEP(BASE,GCOM,float,DULL) 