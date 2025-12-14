

CODEFLAG=-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120
CODEFLAG100=-arch sm_100
GCCFLAG=-lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
OMPFLAG=-Xcompiler -fopenmp
EXECUTABLE=ImplicitBarrier

all: $(EXECUTABLE)

../share/measurement.o: ../share/measurement.cu
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $^
../share/util.o: ../share/util.cpp
	g++ $(GCCFLAG) -c -o $@ $<
Implicit_Barrier_Null_Kernels.o: Implicit_Barrier_Null_Kernels.cu
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $^
Implicit_Barrier_Sleep_Kernels.o: Implicit_Barrier_Sleep_Kernels.cu
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $^
Implicit_Barrier.o: Implicit_Barrier.cu
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $^

ImplicitBarrier: Implicit_Barrier.o Implicit_Barrier_Sleep_Kernels.o Implicit_Barrier_Null_Kernels.o ../share/measurement.o ../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

