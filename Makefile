
CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 
GCCFLAG=-lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
OMPFLAG=-Xcompiler -fopenmp
EXECUTABLE=ImplicitBarrier

all: $(EXECUTABLE) 

measurement.o: measurement.cu
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
util.o: util.cpp
	g++ $(GCCFLAG) -c -o $@ $<
Implicit_Barrier_Null_Kernels.o: Implicit_Barrier_Null_Kernels.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^
Implicit_Barrier.o: Implicit_Barrier.cu 
	nvcc -x cu $(CODEFLAG) -dc -o $@ $^

ImplicitBarrier: Implicit_Barrier.o Implicit_Barrier_Null_Kernels.o measurement.o util.o
	nvcc $(CODEFLAG) $(OMPFLAG) -o $@ $^
