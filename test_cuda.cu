#include "main.h"

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
  }

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)

__global__ void simpleKernel(char *ptr, int sz, char val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
        ptr[idx] = val;
    }
}

static void barrierWait(volatile int * barrier, volatile int *sense, unsigned int n) {
    int count;

    // check-in
    count = cpu_atomic_add32(barrier, 1);
    if (count == n) *sense = 1;
    while (!*sense);

    count = cpu_atomic_add32(barrier, -1);
    if (count == 0) *sense = 0;
    while(*sense);
}

extern "C" void* test_init(int rank, cudaIpcMemHandle_t& handle, cudaIpcEventHandle_t& ev_handle, cudaEvent_t* event) {
    void* ptr = NULL;
    checkCudaErrors(cudaSetDevice(rank));
    if (rank == 0) {
        checkCudaErrors(cudaDeviceEnablePeerAccess(1, 0));
    } else {
        checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
    }
    checkCudaErrors(cudaMalloc(&ptr, DATA_SIZE));
    checkCudaErrors(cudaIpcGetMemHandle(&handle, ptr));
    checkCudaErrors(cudaEventCreate(event, cudaEventDisableTiming | cudaEventInterprocess));
    checkCudaErrors(cudaIpcGetEventHandle(&ev_handle, *event));
    return ptr;
}

extern "C" void run_test(sharedMemoryInfo& info, volatile shmStruct* shm, int size, cudaIpcMemHandle_t& handle, cudaIpcEventHandle_t& ev_handle, cudaIpcMemHandle_t& get_handle, cudaIpcEventHandle_t& get_ev_handle, int rank, cudaEvent_t ev, void* ptr_t) {
    std::vector<char> verification_buffer(DATA_SIZE);
    cudaStream_t stream;
    int blocks = 0;
    int threads = 128;
    cudaDeviceProp prop;
    if (rank == 0) {
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    } else {
        checkCudaErrors(cudaGetDeviceProperties(&prop, 1));
    }
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, simpleKernel, threads, 0));
    blocks *= prop.multiProcessorCount;

    std::vector<void *> ptrs;
    std::vector<cudaEvent_t> events;
    for (int i = 0; i < size; i++) {
        void* ptr = NULL;
        cudaEvent_t event;
        if (i == rank) {
            ptrs.push_back(ptr_t);
            events.push_back(ev);
            continue;
        } else {
            checkCudaErrors(cudaIpcOpenMemHandle(&ptr, get_handle, cudaIpcMemLazyEnablePeerAccess));
            checkCudaErrors(cudaIpcOpenEventHandle(&event, get_ev_handle));
        }

        ptrs.push_back(ptr);
        events.push_back(event);
    }

    for (int i = 0; i < size; i++) {
        size_t bufferId = (i + rank) % size;
        checkCudaErrors(cudaStreamWaitEvent(stream, events[bufferId], 0));

        simpleKernel<<<blocks, threads, 0, stream>>>((char*)ptrs[bufferId], DATA_SIZE, rank);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaEventRecord(events[bufferId], stream));

        barrierWait(&shm->barrier, &shm->sense, (unsigned int)size);
        if (rank == 0) {
            printf("Step %lld done\n", (unsigned long long)i);
        }
    }

    checkCudaErrors(cudaStreamWaitEvent(stream, events[rank], 0));
    checkCudaErrors(cudaMemcpyAsync(&verification_buffer[0], ptrs[rank], DATA_SIZE, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Process %d: verifying...\n", rank);

    char compareId = (char)((rank + 1) % size);
    for (unsigned long long j = 0; j < DATA_SIZE;j++) {
        if (verification_buffer[j] != compareId) {
            printf("Process %d: Verifying mismatch at %lld : %d != %d\n", rank, j, (int)verification_buffer[j], (int)compareId);
        }
    }

    // for (int i = 0; i < size; i++) {
    //     checkCudaErrors(cudaIpcCloseMemHandle(ptrs[i]));
    //     checkCudaErrors(cudaEventDestroy(events[i]));
    // }

    checkCudaErrors(cudaStreamDestroy(stream));

    printf("Process %d complete\n", rank);

}
