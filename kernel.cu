
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
#define N 100000

/* В дальнейшем метод класса Slice
 * внутри функции доступны
 * unsigned int length - длина слайса в битах
 * unsigned int N - длина слайса в 64-х разрядных целых
 * unsigned int IT - количество элементов, обрабатываемых одним потоком (для N>1024)
 *
 * константы
 *  #define MAX_THREADS 1024
 */
#define MAX_THREADS 1024
void __global__ find_kernel(unsigned long long* d_v, unsigned int length,unsigned int N1,unsigned int it, int* res)
{
    __shared__ unsigned int res_by_thread[MAX_THREADS];
    unsigned int local_1st_nonzero,local_it_1st_nonzero,tmp;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int active_threads=gridDim.x * blockDim.x;

    // res_by_thread[n] для it элементов:
    local_1st_nonzero=SIZE_OF_LONG_INT*N+1;
    for(int i=0;i<it;i++){
    	local_it_1st_nonzero = __ffsll(d_v[n*it+i]);  // первая единица в слове
    //номер этой первой единицы глобальный (по всему массиву) начиная справа
    	tmp=(local_it_1st_nonzero!=0)?(local_it_1st_nonzero+(n*it+i)*SIZE_OF_LONG_INT): (SIZE_OF_LONG_INT*N+1);
    	local_1st_nonzero=min(local_1st_nonzero,tmp);
    }
    res_by_thread[n]=local_1st_nonzero;

   while(active_threads>0)
    {
        __syncthreads();
        active_threads=active_threads>>1;
        if (n < active_threads)
        {
            res_by_thread[n] = min(res_by_thread[n], res_by_thread[active_threads+n]);
        }
    }
    if (n==0)
    {
    	*res = res_by_thread[0];
        if (*res>length) *res = 0;
    }
}
unsigned int FND(unsigned long long *d_v)
{
	// вычислить конфигурацию для find_simple
	unsigned int N1 =N,threads,it;
	int *d_res,h_res;
	cudaMalloc(&d_res, sizeof(int));

//	for (N1=512;N1<1000000;N1=N1<<1)
	{
	   threads = min(MAX_THREADS,N1);
	   it=(N1-1)/threads+1;
	   printf("N=%d threads=%d,IT=%d \n",N1,threads,it);
	}
	find_kernel<<<1,threads>>>(d_v,N*64,N1,it,d_res);
	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	return h_res;
}

int main()
{
    unsigned long long h_v[N];/* = {0xABCDABCDABCD0000, 0x0F08000800080070,
                                0xABCDABCDAB900000, 0x0F08000800080700,
                                0xABCDABCDABC80000, 0x0F08000807000000,
                                0xABCDABCDAB001000, 0x0F08000800080500 };*/
    unsigned long long* d_v;
//  int *d_res,h_res;
//   cudaMalloc(&d_res, sizeof(int));

    for (int i = 0; i < N; i++)
    {
        h_v[i] = (i == (800)) ? 0x8000000000000000 : 0;//rand() % MAX + 1;
        int sh = rand() % 32 + 1;
        //h_v[i] <<= sh;
        //printf("%d %30lx shift %d \n",i,h_v[i],sh);
    }
    cudaMalloc(&d_v, N * sizeof(unsigned long long));

    cudaMemcpy(d_v, h_v, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
 /* печать с устройства
    cudaPrintfInit();

    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
*/

    printf("FND= %d \n",FND(d_v));
    return 0;
}
