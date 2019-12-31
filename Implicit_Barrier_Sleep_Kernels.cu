#include "Implicit_Barrier_Kernel.cuh"
#include "wrap_launch_functions.cuh"

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

N2_KERNEL(0);
N2_KERNEL(1);
N2_KERNEL(2);
N2_KERNEL(4);
N2_KERNEL(8);
N2_KERNEL(16);
N2_KERNEL(32);
N2_KERNEL(64);
N2_KERNEL(128);