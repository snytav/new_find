﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <time.h>
#define MAX 999999
#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#define SIZE_OF_LONG_INT 64
//максимальная длина массива из длинных целых (для буферного массива я ядре find)
#define N 1024
__global__ void test()
{
      cuPrintf("test");
}
int  __device__ gap(int level)
{
    //cuPrintf("gap entered level %d\n",level);
    int d2 = 1<<level;
    int g = (gridDim.x * blockDim.x) / d2;
    //cuPrintf("gap %d gridDim.x %d blockDim.x %d denom %d level %d\n",
      //        g,   (int)gridDim.x,   (int)blockDim.x,d2,level);
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
    int levels = int(log((double)(N)) / log(2.0));
   
   
    //номер этой первой единицы глобальный (по всему массиву) начиная справа
    res_by_thread[n] = ((local_1st_nonzero >0) && (local_1st_nonzero != 0xFFFFFFFF))*
                                    (local_1st_nonzero + SIZE_OF_LONG_INT * n) +
                       (local_1st_nonzero == 0)*(N*SIZE_OF_LONG_INT+1);
    cuPrintf("n %d res by thread local %d global %d SIZE_OF_LONG_INT * n %d\n", 
        n,
        local_1st_nonzero,res_by_thread[n],
        SIZE_OF_LONG_INT * n);
    
//    cuPrintf("reduction levels %d active %d gap %d \n",levels, active_thread(levels),gap(levels));
    //return;
    for (int l = 1; l <= levels;l++)
    {
        cuPrintf("l in loop %d activ %d\n",l, active_thread(l));
        __syncthreads();
        if (active_thread(l))
        {
            unsigned int m = get_num_thread_to_compare(l);
            cuPrintf("level %d gap %u compare with %u res_by_thread[n=%d] %d res_by_thread[m=%d] %d \n",
                l,gap(l), m,n, res_by_thread[n],m, res_by_thread[m]);
            res_by_thread[n] = min(res_by_thread[n],
                                        res_by_thread[m]);
            cuPrintf("res_by_thread[n] %d res_by_thread[m] %d \n",
                      res_by_thread[n],   res_by_thread[m]);
        }
    }
    if (n==0)
    {
    	*res = res_by_thread[0];
    // if (*res>length) *res = 0;
    	if (*res>(N*SIZE_OF_LONG_INT)) *res = 0;

    cuPrintf("global min %d\n",*res);
    }
}

void __global__ find_simple(unsigned long long* d_v, unsigned int length, int* res)
{
    __shared__ unsigned int res_by_thread[N];
    unsigned int local_1st_nonzero;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    local_1st_nonzero = __ffsll(d_v[n]);  // первая единица в слове
    unsigned int active_threads=gridDim.x * blockDim.x<<1;
    //номер этой первой единицы глобальный (по всему массиву) начиная справа
     res_by_thread[n]=(local_1st_nonzero!=0)?(local_1st_nonzero+n*SIZE_OF_LONG_INT): (SIZE_OF_LONG_INT*N+1);

   while(active_threads>0)
    {
        __syncthreads();
        if (n < active_threads)
        {
            res_by_thread[n] = min(res_by_thread[n], res_by_thread[active_threads+n]);
        }
        active_threads=active_threads<<1;
    }
    if (n==0)
    {
    	*res = res_by_thread[0];
        if (*res>length) *res = 0;
    }
}

/* В дальнейшем метод класса Slice
 * внутри функции доступны
 * unsigned int length - длина слайса в битах
 * unsigned int N - длина слайса в 64-х разрядных целых
 * unsigned int IT - количество элементов, обрабатываемых одним потоком (для N>1024)
 */
unsigned int FND(unsigned long long *d_v)
{
	// вычислить число конфигурацию для find_simple
}

int main()
{
    unsigned long long h_v[N];/* = {0xABCDABCDABCD0000, 0x0F08000800080070,
                                0xABCDABCDAB900000, 0x0F08000800080700,
                                0xABCDABCDABC80000, 0x0F08000807000000,
                                0xABCDABCDAB001000, 0x0F08000800080500 };*/
    unsigned long long* d_v;
    int *d_res,h_res;
    for (int i = 0; i < N; i++)
    {
        h_v[i] = (i == (2)) ? 0x8000000000000000 : 0;//rand() % MAX + 1;
        int sh = rand() % 32 + 1;
        //h_v[i] <<= sh;
        //printf("%d %30lx shift %d \n",i,h_v[i],sh);
    }
    cudaMalloc(&d_v, N * sizeof(unsigned long long));
    cudaMalloc(&d_res, sizeof(int));
    cudaMemcpy(d_v, h_v, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaPrintfInit();
//    gap << <1, N >> > (2);
    find << <1, N >> > (d_v,N,d_res);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);


    printf("%d \n",h_res);
    find_simple << <1, N >> > (d_v,64*N,1,d_res);
    printf("Simple FND %d \n",h_res);
    return 0;
}
