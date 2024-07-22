

CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90
CODEFLAG70=-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90
CODEFLAG60=-arch sm_60
GCCFLAG=-lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
OMPFLAG=-Xcompiler -fopenmp
EXECUTABLE=ImplicitBarrier 

all: $(EXECUTABLE) 

../share/measurement.o: ../share/measurement.cu
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
../share/util.o: ../share/util.cpp
	g++ $(GCCFLAG) -c -o $@ $<
Implicit_Barrier_Null_Kernels.o: Implicit_Barrier_Null_Kernels.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
Implicit_Barrier_Sleep_Kernels.o: Implicit_Barrier_Sleep_Kernels.cu 
	nvcc -x cu $(CODEFLAG70) -dc -o $@ $^
Implicit_Barrier.o: Implicit_Barrier.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

ImplicitBarrier: Implicit_Barrier.o Implicit_Barrier_Sleep_Kernels.o Implicit_Barrier_Null_Kernels.o ../share/measurement.o ../share/util.o
	nvcc $(CODEFLAG70) $(OMPFLAG) -o $@ $^

