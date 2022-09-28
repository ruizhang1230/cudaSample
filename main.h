#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#include "mpi.h"
#include "cuda_runtime.h"


#define DATA_SIZE (64ULL << 20ULL)   // 64MB

static const char shmName[] = "simpleIPCshm";

typedef struct shmStruct_st {
    size_t nprocesses;
    int barrier;
    int sense;
    int devices[10];
    cudaIpcMemHandle_t memHandle[10];
    cudaIpcEventHandle_t eventHandle[10];
} shmStruct;

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE shmHandle;
#else
    int shmFd;
#endif
} sharedMemoryInfo;