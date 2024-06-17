
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#define SIZE_OF_LONG_INT 64

//максимальная длина массива из длинных целых (для буферного массива я ядре find)
#define N 2
__global__ void test()
{
      cuPrintf("test");
}

unsigned int __device__ gap(int level)
{
    int denom = (int)pow(2, level);
    unsigned int g = (gridDim.x * blockDim.x) / level / denom;
    return g;
}

unsigned int __device__ get_num_thread_to_compare(int level)
{
   
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    return gap(level) + n;
}

int __device__ active_thread(int level)
{
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    return (n < gap(level));
}


//считается что size равен количеству потоков
void __global__ find(unsigned long long* d_v, int size, int* res)
{
    __shared__ int res_by_thread[N];
    int local_1st_nonzero;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    local_1st_nonzero = __ffsll(d_v[n]);  // первая единица в слове
    int levels = int(log((double)N) / log(2.0));
   
    //номер этой первой единицы глобальный (по всему массиву) начиная справа
    res_by_thread[n] = local_1st_nonzero + SIZE_OF_LONG_INT * (size-n-1);
    cuPrintf("n %d res by thread local %d global %d\n",n, 
        local_1st_nonzero,res_by_thread[n]);
    

    cuPrintf("reduction levels %d active %d gap %d \n",levels, active_thread(1),gap(1));
   
    for (int l = 1; l <= levels; l++)
    {
        if (active_thread(l))
        {
            unsigned int m = get_num_thread_to_compare(l);
            cuPrintf("level %d gap %u compare with %u \n", l,gap(l), m);
            res_by_thread[n] = min(res_by_thread[n],
                                        res_by_thread[m]);
            cuPrintf("res_by_thread[n] %d res_by_thread[m] %d \n",
                      res_by_thread[n],   res_by_thread[m]);

        }
    }

   
//    *res = minp;
    cuPrintf("global min %d\n",*res);
}


int main()
{
    unsigned long long h_v[] = {0xABCDABCDABCD0000, 0x0F08000800080070};
    unsigned long long* d_v;
    int *d_res,h_res;

    cudaMalloc(&d_v, N * sizeof(unsigned long long));
    cudaMalloc(&d_res, sizeof(int));
    cudaMemcpy(d_v, h_v, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaPrintfInit();
    find << <1, N >> > (d_v,N,d_res);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);




    return 0;
}
