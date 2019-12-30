
#include<set>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}
// #ifndef DEBUG
// #define  checkCudaErrors(err)\
// {	\
//   cudaError_t e=err;\
// 	if(e!=cudaSuccess) {        \
// 		char str[100];\
//    		sprintf(str,"Cuda failure %s:%d: '%s'",__FILE__,__LINE__,cudaGetErrorString(e));\
//    		throw str;\
// 	}\
// }

// #endif
// #ifdef DEBUG
// #define  checkCudaErrors(err)\
// {	\
//   cudaError_t e=err;\
// 	if(e!=cudaSuccess) {        \
//     printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
// 	}\
// }
// #endif
unsigned long ToUInt(char* str);
char* genFileName(char* filename, const char *  base,  const char *  deviceName);
// char* addDir(char* baseDir, const char *  addDir);
void getIdenticalGPUs(int num_of_gpus, std::set<int> &identicalGPUs, bool coalaunch=1);
/*
@para
  num_of_gpus gpus insied a Node (large than 2 or stop the program)
  identicalGPUs GPUs with same platform
  coaleanch if true identicalGPUs should be able to do coalanch (i.e. P100 and V100)
*/