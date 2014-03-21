EXECUTABLE := release/main
NVCC := nvcc
BLOCKTESTARRAYSIZE := 100000000
ONETOHUNDREDSIZE := 56 

all: main.cu xmalloc.cu util.cu
	$(NVCC) -lcudart -arch=sm_20 main.cu -O3 -o $(EXECUTABLE)


