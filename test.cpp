#include "main.h"

extern "C" void* test_init(int rank, cudaIpcMemHandle_t& handle, cudaIpcEventHandle_t& ev_handle, cudaEvent_t* event);
extern "C" void run_test(sharedMemoryInfo& info, volatile shmStruct* shm, int size, cudaIpcMemHandle_t& handle, cudaIpcEventHandle_t& ev_handle, cudaIpcMemHandle_t& get_handle, cudaIpcEventHandle_t& get_ev_handle, int rank, cudaEvent_t ev, void* ptr_t);

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info)
{
    int status = 0;

    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    status = ftruncate(info->shmFd, sz);
    if (status != 0) {
        return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info)
{
    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
}


void sharedMemoryClose(sharedMemoryInfo *info)
{
    if (info->addr) {
        munmap(info->addr, info->size);
    }
    if (info->shmFd) {
        close(info->shmFd);
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sharedMemoryInfo info;
    volatile shmStruct *shm = NULL;

    void* ptr = NULL;
    cudaEvent_t event;
    cudaIpcMemHandle_t handle;
    cudaIpcEventHandle_t ev_handle;
    ptr = test_init(rank, handle, ev_handle, &event);
    if (rank == 0) {

        if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
            printf("failed to create shared memory slab\n");
            exit(-1);
        }
        shm = (volatile shmStruct *)info.addr;
        memset((void*)shm, 0, sizeof(*shm));

    }

    // 发送IpcHandle
    char* tmp_handle = (char*)malloc(64 * sizeof(char));
    cudaIpcMemHandle_t get_handle;
    if (rank == 0) {
        memset((void*)tmp_handle, 0, 64);
        memcpy(tmp_handle, &handle, sizeof(cudaIpcMemHandle_t));
        MPI_Send(tmp_handle, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        memset((void*)tmp_handle, 0, 64);
        MPI_Recv(tmp_handle, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(&get_handle, tmp_handle, sizeof(cudaIpcMemHandle_t));
    } else {
        memset((void*)tmp_handle, 0, 64);
        MPI_Recv(tmp_handle, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(&get_handle, tmp_handle, sizeof(cudaIpcMemHandle_t));

        memset((void*)tmp_handle, 0, 64);
        memcpy(tmp_handle, &handle, sizeof(cudaIpcMemHandle_t));
        MPI_Send(tmp_handle, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 发送event
    cudaIpcEventHandle_t get_ev_handle;
    if (rank == 0) {
        memset((void*)tmp_handle, 0, 64);
        memcpy(tmp_handle, &ev_handle, sizeof(cudaIpcEventHandle_t));
        MPI_Send(tmp_handle, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        memset((void*)tmp_handle, 0, 64);
        MPI_Recv(tmp_handle, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(&get_ev_handle, tmp_handle, sizeof(cudaIpcEventHandle_t));
    } else {
        memset((void*)tmp_handle, 0, 64);
        MPI_Recv(tmp_handle, 64, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(&get_ev_handle, tmp_handle, sizeof(cudaIpcEventHandle_t));

        memset((void*)tmp_handle, 0, 64);
        memcpy(tmp_handle, &ev_handle, sizeof(cudaIpcEventHandle_t));
        MPI_Send(tmp_handle, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
            printf("failed to create shared memory slab\n");
            exit(-1);
        }
        shm = (volatile shmStruct *)info.addr;
    }

    run_test(info, shm, size, handle, ev_handle, get_handle, get_ev_handle, rank, event, ptr);


    if (rank == 0) {
        sharedMemoryClose(&info);
    }

    MPI_Finalize();

}