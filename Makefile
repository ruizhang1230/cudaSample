CUDA_INSTALL_PATH=/usr/local/cuda
MPI_INSTALL_PATH=/home/ubuntu/openmpi

MPICC = $(MPI_INSTALL_PATH)/bin/mpicxx 
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc

NVCCFLAGS= -I /usr/local/cuda/include -I /home/ubuntu/openmpi/include

LIBS = -lcudart -lcurand -L$(CUDA_INSTALL_PATH)/lib64

CFILES = test.cpp
CUFILES = test_cuda.cu
OBJECTS = test.o test_cuda.o
EXECNAME = test

all:
	$(MPICC) -g -c $(CFILES) $(NVCCFLAGS) -lrt -L/usr/lib/x86_64-linux-gnu/
	$(NVCC)  -g -c $(CUFILES) $(NVCCFLAGS) $(LIBS)
	$(MPICC) -g -o $(EXECNAME) $(NVCCFLAGS) $(OBJECTS) -lrt -L/usr/lib/x86_64-linux-gnu/  $(LIBS)

clean:
	rm -f *.o $(EXECNAME)
