#include "util.h"

#include<string.h>
#include <set>
#include "stdlib.h"
#include "stdio.h"
#include"cuda_runtime.h"

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
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
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
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *it));
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
