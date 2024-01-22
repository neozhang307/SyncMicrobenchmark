
EXECUTABLE=mgreduce greduce 

#set use Event to use Event Clock
CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 

all: $(EXECUTABLE) 


mgreduce: $(CUOBJS) $(COBJS) mgrid_reduction.cu
	nvcc $(CODEFLAG) -std=c++17 -rdc=true -Xcompiler -fopenmp -o $@ $^
greduce: $(CUOBJS) $(COBJS) grid_reduction.cu
	nvcc $(CODEFLAG) -rdc=true -std=c++17 -Xcompiler -fopenmp -o $@ $^
clean:
	rm  $(EXECUTABLE) 
