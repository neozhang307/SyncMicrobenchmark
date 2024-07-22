

CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 
CODEFLAG70=-arch sm_70
CODEFLAG60=-arch sm_60
GCCFLAG=-lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
OMPFLAG=-Xcompiler -fopenmp
EXECUTABLE=TestRepeat BenchmarkIntraSM BenchmarkInterSM

all: $(EXECUTABLE) 


../share/util.o: ../share/util.cpp
	g++ $(GCCFLAG) -c -o $@ $<

	
../share/measurement.o: ../share/measurement.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

Explicit_Barrier_Kernel.o: Explicit_Barrier_Kernel.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

Explicit_Barrier_BigKernel.o: Explicit_Barrier_BigKernel.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

Test_Repeat_Influence.o: Test_Repeat_Influence.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
Explicit_Barrier_IntraSM.o: Explicit_Barrier_IntraSM.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
Explicit_Barrier_InterSM.o: Explicit_Barrier_InterSM.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

TestRepeat: Test_Repeat_Influence.o Explicit_Barrier_Kernel.o ../share/measurement.o ../share/util.o 
	nvcc $(CODEFLAG) -o $@ $^

BenchmarkIntraSM: Explicit_Barrier_IntraSM.o Explicit_Barrier_BigKernel.o Explicit_Barrier_Kernel.o ../share/measurement.o ../share/util.o 
	nvcc $(CODEFLAG) -o $@ $^

BenchmarkInterSM: Explicit_Barrier_InterSM.o Explicit_Barrier_BigKernel.o Explicit_Barrier_Kernel.o ../share/measurement.o ../share/util.o 
	nvcc $(CODEFLAG) -o $@ $^
