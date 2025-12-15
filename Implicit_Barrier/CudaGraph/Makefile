
CODEFLAG=-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120
GCCFLAG=-lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
OMPFLAG=-Xcompiler -fopenmp

EXECUTABLES=bench_stream_capture bench_while_conditional bench_child_graph bench_empty_node bench_event_node

all: $(EXECUTABLES)

../../share/util.o: ../../share/util.cpp
	g++ $(GCCFLAG) -c -o $@ $<

CudaGraph_Kernels.o: CudaGraph_Kernels.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

CudaGraph_StreamCapture.o: CudaGraph_StreamCapture.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

CudaGraph_WhileConditional.o: CudaGraph_WhileConditional.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

CudaGraph_ChildGraph.o: CudaGraph_ChildGraph.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

CudaGraph_EmptyNode.o: CudaGraph_EmptyNode.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

CudaGraph_EventNode.o: CudaGraph_EventNode.cu CudaGraph_Kernel.cuh
	nvcc -x cu $(CODEFLAG) -rdc=true -dc -o $@ $<

bench_stream_capture: CudaGraph_StreamCapture.o CudaGraph_Kernels.o ../../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

bench_while_conditional: CudaGraph_WhileConditional.o CudaGraph_Kernels.o ../../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

bench_child_graph: CudaGraph_ChildGraph.o CudaGraph_Kernels.o ../../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

bench_empty_node: CudaGraph_EmptyNode.o CudaGraph_Kernels.o ../../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

bench_event_node: CudaGraph_EventNode.o CudaGraph_Kernels.o ../../share/util.o
	nvcc $(CODEFLAG) -rdc=true $(OMPFLAG) -o $@ $^

clean:
	rm -f *.o $(EXECUTABLES)
