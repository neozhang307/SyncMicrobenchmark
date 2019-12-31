#include "repeat.h"

void Null_Kernel(unsigned int block_perGPU, unsigned int thread_perBlock);

template <int gpu_count>
void Null_Kernel_MGPU(unsigned int block_perGPU, unsigned int thread_perBlock);


