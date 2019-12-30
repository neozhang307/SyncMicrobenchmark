
#include<cmath>
#include "statistics.h"
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