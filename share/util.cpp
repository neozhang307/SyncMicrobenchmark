#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include<string.h>
#include <set>
#include <math.h>
#include<time.h>




unsigned long ToUInt(char* str)
{
    unsigned long mult = 1;
    unsigned long re = 0;
    int len = strlen(str);
    for(int i = len -1 ; i >= 0 ; i--)
    {
        re = re + ((int)str[i] -48)*mult;
        mult = mult*10;
    }
    return re;
}

char* genFileName(char* filename, const char *  base,  const char *  deviceName)
{
  strcpy (filename,base);
    strcat(filename," ");
  strcat(filename,deviceName);

  char *ptr = filename;    
  while (*ptr)
    {
        if (*ptr == ' ')
            *ptr = '_';
        ptr++;
    }

    struct tm *newtime;
    char time_str[128];
    time_t t1;
    t1 = time(NULL); 
    newtime=localtime(&t1);
    strftime( time_str, 128, "_%Y_%m_%d_%H_%M_%S", newtime);
    strcat(filename,time_str);
    strcat(filename,".log");
    return filename;
}

char* addDir(char* baseDir, const char *  addDir)
{
    if(strlen(baseDir)!=0)
      strcat(baseDir,"\\");
    strcpy(baseDir,addDir);
    return baseDir;
}

void getIdenticalGPUs(int num_of_gpus, std::set<int> &identicalGPUs, bool coalaunch) {
  int *major_minor = (int *)malloc(sizeof(int) * num_of_gpus * 2);
  int foundIdenticalGPUs = 0;

  for (int i = 0; i < num_of_gpus; i++) {
    cudaDeviceProp deviceProp;
    
    cudaGetDeviceProperties(&deviceProp, i);
    cudaCheckError();
    major_minor[i * 2] = deviceProp.major;
    major_minor[i * 2 + 1] = deviceProp.minor;
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  int maxMajorMinor[2] = {0, 0};

  for (int i = 0; i < num_of_gpus; i++) {
    for (int j = i + 1; j < num_of_gpus; j++) {
      if ((major_minor[i * 2] == major_minor[j * 2])&&(major_minor[i * 2 + 1] == major_minor[j * 2 + 1])){ 
        identicalGPUs.insert(i);
        identicalGPUs.insert(j);
        foundIdenticalGPUs = 1;
        if (maxMajorMinor[0] < major_minor[i * 2] &&
            maxMajorMinor[1] < major_minor[i * 2 + 1]) {
          maxMajorMinor[0] = major_minor[i * 2];
          maxMajorMinor[1] = major_minor[i * 2 + 1];
        }
      }
    }
  }
  free(major_minor);
  if (!foundIdenticalGPUs) {
    printf(
        "#00 No Two or more GPUs with same architecture found\nWaiving the "
        "sample\n");
    exit(0);
  }

  std::set<int>::iterator it = identicalGPUs.begin();

  // Iterate over all the identical GPUs found
  if(coalaunch==1)
  {
      while (it != identicalGPUs.end()) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, *it);
        cudaCheckError();
        // Remove all the GPUs which are less than the best arch available
        if (deviceProp.major != maxMajorMinor[0] &&
            deviceProp.minor != maxMajorMinor[1]) {
          identicalGPUs.erase(it);
        }
        if (!deviceProp.cooperativeMultiDeviceLaunch ||
            !deviceProp.concurrentManagedAccess) {
          identicalGPUs.erase(it);
        }
        it++;
      }
    }
  return;
}



void nxtline(){printf("\n");};

void showlatency(latencys g_result)
{
  double*c_result = (double*)&g_result;
  for(int j=0; j<8; j++)
  {
    printf("%f\t",c_result[j]);
  }
}

void showlatency_ttl(latencys g_result)
{
    printf("%f\t%f\t",g_result.mean_lat, g_result.s_lat);
}

void showlatency_cycle(latencys g_result)
{
    printf("%f\t%f\t",g_result.latency_min, g_result.s_latency_min);
}


void prepare_showAdditionalLatency()
{
  printf("method\tGPUCount\trep\tblk\tthrd\tm(clk)\ts(clk)\tm(sync)\ts(sync)\tm(laun)\ts(laun)\tm(ttl)\ts(ttl)\tm(avelaun)\ts(avelaun)\tm(addl)\ts(addl)\n"); 
}

void showAdditionalLatency(latencys g_result_basic, latencys g_result_more,const char* funcname, 
  unsigned int gpu_count,
  unsigned int block_perGPU, unsigned int thread_perBlock, 
  unsigned int basicDEP, unsigned int moreDEP)
{
  printf("%s\t%u\t%u\t%u\t%u\t",funcname,gpu_count,basicDEP,block_perGPU,thread_perBlock);
  showlatency(g_result_basic);
  printf("%f\t%f\t",g_result_basic.mean_laun/basicDEP,g_result_basic.s_laun/basicDEP);
  nxtline();

  printf("%s\t%u\t%u\t%u\t%u\t",funcname ,gpu_count,moreDEP,block_perGPU,thread_perBlock);
  showlatency(g_result_more);
  printf("%f\t%f\t",g_result_more.mean_laun/moreDEP,g_result_more.s_laun/moreDEP);
  printf("%f\t%f\t",
    computeAddLat(g_result_basic,g_result_more,moreDEP-basicDEP),
    computeAddLats(g_result_basic,g_result_more,moreDEP-basicDEP));
  nxtline();

}

void prepare_showRepeatKernelLatency()
{
  printf("method\trep\tblk\tthrd\ttile\tm(cycle)\ts(cycle)\tm(ns)\ts(ns)\tm(ave_cycle)\ts(ave_cycle)\n"); 
}
void showRepeatKernelLatency(latencys result, latencys clk_result,unsigned int difference)
{
  showlatency_cycle(result);\
  showlatency_ttl(result);\
  printf("%f\t%f\t", computeAvgLatCycle(clk_result,result,difference),computeAvgLatCycles(clk_result,result,difference));\
  nxtline();\
}
void prepare_showFusedResult()
{
  printf("method\tGPUCount\trep\tblk\tthrd\tidea(wkld)\tm(wkld)\ts(wkld)\tm(ovh)\ts(ovh)\n"); 
}

void showFusedResult(latencys result_bl_bk, latencys result_bl_mk, latencys result_ml_bk,
  const char* funcname, 
  unsigned int gpu_count,
  unsigned int block_perGPU, unsigned int thread_perBlock, 
  unsigned int basicDEP, unsigned int moreDEP, unsigned int idea_basic_workload)
{
  printf("%s\t%u\t%u:%u\t%u\t%u\t",funcname,gpu_count,basicDEP,moreDEP, block_perGPU,thread_perBlock);
  //real workload and theoretical workload
  printf("%u\t%f\t%f\t",idea_basic_workload,
    computeAddLat(result_bl_bk,result_bl_mk,moreDEP-basicDEP),
    computeAddLats(result_bl_bk,result_bl_mk,moreDEP-basicDEP));
  printf("%f\t%f\n",
    computeAddLat(result_bl_mk,result_ml_bk,moreDEP-basicDEP),
    computeAddLats(result_bl_mk,result_ml_bk,moreDEP-basicDEP));
  //deduced launch overhead
}

void prepare_showLatencyInSingleSM()
{
  printf("method\tGPUCount\trep\tblk\tthrd\tm(avgcycle)\ts(avgcycle)\n");
}

void showLatencyInSingleSM(latencys basic, latencys result, const char* kernelname, 
  unsigned int gpu_count, unsigned int rep,
  unsigned int blockPerGPU, unsigned int threadPerBlock)
{
  printf("%s\t%u\t%u\t%u\t%u\t", kernelname,gpu_count,rep,blockPerGPU,threadPerBlock);
  printf("%f\t%f\n", computeAvgLatCycle(basic,result,rep*2),computeAvgLatCycles(basic,result,rep*2));
}

void prepare_showThroughputInSingleSM()
{
  printf("method\tGPUCount\trep\tblk\tthrd\ttile\tm(ttl_latency)\tm(thrput)\n");
}

void showThroughputInSingleSM(latencys result, const char* kernelname, 
  unsigned int gpu_count, unsigned int rep,
  unsigned int blockPerGPU, unsigned int threadPerBlock, 
  unsigned int smx_count, unsigned int tile)
{
  printf("%s\t%u\t%u\t%u\t%u\t%u\t", kernelname,gpu_count,rep,blockPerGPU,threadPerBlock,tile);
  printf("%f\t%f\n", result.latency_max,(blockPerGPU/smx_count)*(threadPerBlock/32)*rep*2/result.latency_max);

}
void prepare_showLatencyInterSM()
{
    printf("method\tGPUCount\tbasicdec\tmoredec\tblk\tthrd\t"); 
    printf("m(basic_ttl)\ts(basic_ttl)\tm(more_ttl)\ts(more_ttl)\t");
    printf("m(avginstr)\ts(avginstr)\n"); 
}

void showLatencyInterSM(latencys result_basic,latencys result_more, const char* kernelname, 
              unsigned int basic_dec, unsigned int more_dec,
              unsigned int gpu_count, unsigned int blockPerGPU, unsigned int threadPerBlock)
{
    printf("%s\t%u\t%u\t%u\t%u\t%u\t", kernelname,gpu_count,basic_dec, more_dec,blockPerGPU,threadPerBlock);
  printf("%f\t%f\t%f\t%f\t",result_basic.mean_lat,result_basic.s_lat,result_more.mean_lat,result_more.s_lat);
  printf("%f\t%f\n",(result_more.mean_lat-result_basic.mean_lat)/(more_dec-basic_dec)/2,sqrt(result_basic.s_lat*result_basic.s_lat+result_more.s_lat*result_more.s_lat)/(more_dec-basic_dec)/2);
}
double computeAddLat(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return (g_result_more.mean_lat-g_result_basic.mean_lat)/difference;
}

double computeAddLats(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return sqrt((g_result_more.s_lat*g_result_more.s_lat+g_result_basic.s_lat*g_result_basic.s_lat))/difference;
}

double computeAvgLatCycle(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return (g_result_more.latency_min-g_result_basic.latency_min)/difference;
}
double computeAvgLatCycles(latencys g_result_basic, latencys g_result_more, unsigned int difference)//size=2
{
  return sqrt((g_result_more.s_latency_min*g_result_more.s_latency_min+g_result_basic.s_latency_min*g_result_basic.s_latency_min))/difference;
}



void GetLatencyOfSM(
            unsigned int& Min_Latency,//count when last thread start sync first thread finish sync
            unsigned int& Max_Latency,//count when first thread start sync last thread finish sync
            unsigned int totalWarpCount, //warp number in a synchronization group
            unsigned int* time_stamp, //time stamp from kernel, size=groupCount*warpPerGroup
            unsigned int* idx,  //idx from each warp, size=groupCount*warpPerGroup
            unsigned int smid)//the sm to measure
{
  unsigned int basic_index=0;
  if(smid==0)smid=idx[(basic_index)*2];
  while(smid!=idx[(basic_index)*2])basic_index++;
  unsigned int start_min = time_stamp[basic_index*2];
  unsigned int end_min = time_stamp[basic_index*2+1];
  unsigned int start_max = time_stamp[basic_index*2];
  unsigned int end_max = time_stamp[basic_index*2+1];
  basic_index++;
  for(; basic_index < totalWarpCount; basic_index ++)
  {
    if(smid!=idx[(basic_index)*2])continue;
    start_min=std::max(start_min,time_stamp[(basic_index)*2]);
    end_min=std::min(end_min,time_stamp[(basic_index)*2+1]);

    start_max=std::min(start_max,time_stamp[(basic_index)*2]);
    end_max=std::max(end_max,time_stamp[(basic_index)*2+1]);
  }
  Min_Latency=end_min-start_min;
  Max_Latency=end_max-start_max;
}


void getStatistics(double &mean, double &s,
  double* list, unsigned int size)
{
  if(size<=0)
  {
    mean=0;
    s=0;
    return;
  }
  double sum = 0; 
  for(int i=0; i<size; i++)
  {
    sum+=list[i];
  }
  mean=sum/size;

  sum=0;
  for(int i=0; i<size; i++)
  {
    sum+=pow(list[i]-mean,2);
  }

  s=sqrt(sum/(size-1)); 
}
