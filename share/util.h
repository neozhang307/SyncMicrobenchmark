
#include<set>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

// struct latencys;
#ifndef LATS
struct latencys
{
   double mean_clk;
   double s_clk;

   double mean_sync;
   double s_sync;

   double mean_laun;
   double s_laun;

   double mean_lat;
   double s_lat;

  //block sync, grid sync latency;
  double latency_min;//count when last thread start sync first thread finish sync
  double s_latency_min;
  //computing throughput;
  double latency_max;//count when first thread start sync last thread finish sync
  double s_latency_max;//count when first thread start sync last thread finish sync

};
#define LATS
#endif
// #ifndef DEBUG

unsigned long ToUInt(char* str);
char* genFileName(char* filename, const char *  base,  const char *  deviceName);
// char* addDir(char* baseDir, const char *  addDir);
void getIdenticalGPUs(int num_of_gpus, std::set<int> &identicalGPUs, bool coalaunch);
/*
@para
  num_of_gpus gpus insied a Node (large than 2 or stop the program)
  identicalGPUs GPUs with same platform
  coaleanch if true identicalGPUs should be able to do coalanch (i.e. P100 and V100)
*/
void getStatistics(double &mean, double &s,
  double* list, unsigned int size);


void showlatency(latencys g_result);
void showlatency_ttl(latencys g_result);
void showlatency_cycle(latencys g_result);

void prepare_showAdditionalLatency();
void showAdditionalLatency(latencys result_basic, latencys result_more, const char* funcname, 
  unsigned int gpu_count,
  unsigned int block_perGPU, unsigned int thread_perBlock, 
  unsigned int basicDEP, unsigned int moreDEP);

void prepare_showFusedResult();
void showFusedResult(latencys result_bl_bk, latencys result_bl_mk, latencys result_ml_bk,
  const char* funcname, 
  unsigned int gpu_count,
  unsigned int block_perGPU, unsigned int thread_perBlock, 
  unsigned int basicDEP, unsigned int moreDEP, unsigned int idea_basic_workload);

double computeAddLat(latencys result_basic, latencys result_more, unsigned int difference);
double computeAddLats(latencys result_basic, latencys result_more, unsigned int difference);
void nxtline();