
EXECUTABLE=greduce

#set use Event to use Event Clock
CODEFLAG=-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120

all: $(EXECUTABLE)

greduce: $(CUOBJS) $(COBJS) grid_reduction.cu
	nvcc $(CODEFLAG) -rdc=true -std=c++17 -Xcompiler -fopenmp -o $@ $^
clean:
	rm  $(EXECUTABLE) 
