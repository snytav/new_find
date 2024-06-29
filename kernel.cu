
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
//#include <time.h>
#include "kernel.h"


//максимальная длина массива из длинных целых (для буферного массива я ядре find)


/* В дальнейшем метод класса Slice
 * внутри функции доступны
 * unsigned int length - длина слайса в битах
 * unsigned int N - длина слайса в 64-х разрядных целых
 * unsigned int IT - количество элементов, обрабатываемых одним потоком (для N>1024)
 *
 * константы
 *  #define MAX_THREADS 1024
 */

void __global__ find_kernel(unsigned long long* d_v, unsigned int length,unsigned int N1,unsigned int it, int* res)
{
    __shared__ unsigned int res_by_thread[MAX_THREADS];
    unsigned int local_1st_nonzero,local_it_1st_nonzero,tmp;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int active_threads=gridDim.x * blockDim.x;

    // res_by_thread[n] для it элементов:
    local_1st_nonzero=SIZE_OF_LONG_INT*N1+1;
    for(int i=0;i<it;i++){
    	local_it_1st_nonzero = __ffsll(d_v[n*it+i]);  // первая единица в слове
    //номер этой первой единицы глобальный (по всему массиву) начиная справа
    	tmp=(local_it_1st_nonzero!=0)?(local_it_1st_nonzero+(n*it+i)*SIZE_OF_LONG_INT): (SIZE_OF_LONG_INT*N1+1);
    	local_1st_nonzero=min(local_1st_nonzero,tmp);
    }
    res_by_thread[n]=local_1st_nonzero;

   while(active_threads>1)
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

void __global__ some_kernel(unsigned long long* d_v,unsigned int N1,unsigned int it, int*res)
{
	__shared__ unsigned int tmp;
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n==0) tmp=0;
	__syncthreads();

	int i=0;
	while((!tmp)&&(i<it))
	{
		if(d_v[n*it+i]>0) tmp=1;
		i++;
	}
	__syncthreads();
	 if (n==0)
		 *res = tmp;
}

void __global__ zero_kernel(unsigned long long* d_v,unsigned int N1,unsigned int it, int*res)
{
	__shared__ unsigned int tmp;
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n==0) tmp=1;
	__syncthreads();

	int i=0;
	while((tmp)&&(i<it))
	{
		if(d_v[n*it+i]>0) tmp=0;
		i++;
	}
	__syncthreads();
	 if (n==0)
		 *res = tmp;
}

void __global__ numb_kernel(unsigned long long* d_v, unsigned int length,unsigned int N1,unsigned int it, int* res)
{
    __shared__ unsigned int res_by_thread[MAX_THREADS];
    unsigned int tmp;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int active_threads=gridDim.x * blockDim.x;

    unsigned long long int tail;


    // res_by_thread[n] для it элементов:
    tmp=0;
    for(int i=n*it;i<(n+1)*it;i++){
    	if(i==(N1-1))
    	{   tail=1;
    		tail=(tail<<(length%SIZE_OF_LONG_INT))-1;
 //   		printf("length=%d,остаток= %d \t",length,length%SIZE_OF_LONG_INT);
 //   		printf("%lx %d \n",tail,tail==0);
            if (tail==0)
            {
            	tail=~0;
 //           	printf("tail2=%llu \t",tail);
            }
 //   		printf("%d:\t d_v[%d]=%lx tail=%lx ",N1,i,d_v[i],tail);
    		d_v[i]=d_v[i]&tail;
//    		printf(":%lx\n",d_v[i]);
    	}
    	tmp += __popcll(d_v[i]);  // первая единица в слове
    }
    res_by_thread[n]=tmp;

   while(active_threads>1)
    {
        __syncthreads();
        active_threads=active_threads>>1;
        if (n < active_threads)
        {
            res_by_thread[n] = res_by_thread[n]+ res_by_thread[active_threads+n];
        }
    }
    if (n==0)
    {
    	*res = res_by_thread[0];

    }
}


