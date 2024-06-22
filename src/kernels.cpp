#include <kernels.hpp>
#include <hip/hip_runtime.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess));

__global__ void gpuHello()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", tid);
}

int simpleKernel(int n)
{
    gpuHello<<<1,1>>>();
    HIP_ASSERT(hipDeviceSynchronize());
    return n+1;
}