
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

void __global__ find_kernel(LongPointer d_v, unsigned int length,unsigned int N1,unsigned int it, int* res)
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

void __global__ some_kernel(LongPointer d_v,unsigned int N1,unsigned int it, int*res)
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

void __global__ tail_kernel(LongPointer d_v, unsigned int length,unsigned int N1)
{
	unsigned long long int tail=1;
	tail=(tail<<(length%SIZE_OF_LONG_INT))-1;
	 //   		printf("length=%d,остаток= %d \t",length,length%SIZE_OF_LONG_INT);
	 //   		printf("%lx %d \n",tail,tail==0);
    if (tail==0) tail=~0;
//	printf("%d:\t d_v[%d]=%lx tail=%lx ",N1,d_v[N1-1],tail);
 		d_v[N1-1]=d_v[N1-1]&tail;

}


void __global__ zero_kernel(LongPointer d_v,unsigned int N1,unsigned int it, int*res)
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

void __global__ numb_kernel(LongPointer d_v, unsigned int length,unsigned int N1,unsigned int it, int* res)
{
    __shared__ unsigned int res_by_thread[MAX_THREADS];
    unsigned int tmp;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int active_threads=gridDim.x * blockDim.x;

    // res_by_thread[n] для it элементов:
    tmp=0;
    for(int i=n*it;i<(n+1)*it;i++)
    {
    	tmp += __popcll(d_v[i]);  // количество единиц в слове
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

__device__ void _and(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] &= d_v1[index+i];
}

__global__ void and_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1){   _and(d_v,d_v1,NN,IT);}


__device__ void _or(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] |= d_v1[index+i];
}

__global__ void or_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1){   _or(d_v,d_v1,NN,IT);}

__device__ void _xor(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] ^= d_v1[index+i];
}

__global__ void xor_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1){   _xor(d_v,d_v1,NN,IT);}

__device__ void _assign(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] = d_v1[index+i];
}

__global__ void assign_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1){   _assign(d_v,d_v1,NN,IT);}

__device__ void _set(LongPointer d_v, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] =0xFFFFFFFFFFFFFFFF;
}

__global__ void set_kernel(LongPointer d_v, unsigned int NN,
unsigned int IT=1){   _set(d_v,NN,IT);}

__device__ void _clr(LongPointer d_v, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] =0;
}

__global__ void clr_kernel(LongPointer d_v, unsigned int NN,
unsigned int IT=1){   _clr(d_v,NN,IT);}

__device__ void _not(LongPointer d_v, unsigned int NN,
unsigned int it)
{
    unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
    for(int i=0; i<it;i++)
	if (index+i<NN) d_v[index+i] =~d_v[index+i];
}

__global__ void not_kernel(LongPointer d_v, unsigned int NN,
unsigned int IT=1){   _not(d_v,NN,IT);}


__global__ void setbit_kernel(LongPointer d_v,unsigned int n, int bit)
{// нумерация с 0, поэтому не добавляется
	unsigned int n_el=(n-1)/SIZE_OF_LONG_INT;
	unsigned int n_i=(n-1)%SIZE_OF_LONG_INT;

	unsigned long long int tmp=1;
	tmp=tmp<<n_i;

	if (bit==1)
		d_v[n_el]|=tmp;
	else
		d_v[n_el]&=~tmp;
}

void __global__ getbit_kernel(LongPointer d_v, unsigned int n,int *d_res)
{ //нумерация с 0, поэтому не добавляется
	unsigned int n_el=(n-1)/SIZE_OF_LONG_INT;
	unsigned int n_i=(n-1)%SIZE_OF_LONG_INT;

	unsigned long long int tmp=1;
	tmp=tmp<<n_i;
	tmp=tmp&d_v[n_el];
	*d_res=(tmp==0)?0:1;
//	printf("tmp=%lx bit=%d\n",tmp,*d_res);
}

__global__ void mask_kernel(LongPointer d_v,int num, unsigned int NN,
unsigned int IT=1){   _mask(d_v,num,NN,IT);}

__device__ void _mask(LongPointer d_v, int num,unsigned int NN,
		unsigned int it)
{ unsigned long long int zero=1;
  int num_el=num>>6; // номер элемента, содержащий переход от 0 к 1;
  int el=num % SIZE_OF_LONG_INT;
//  printf("%i in %i \n", num,num_el);

  unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
  for(int i=0; i<it;i++)
	  if((index+i)<NN)
	  {
		  if ((index+i)==num_el)
		  {
			  zero=(el==0)?0:(zero<<(el-1))-1;
			  zero=~zero;
		  }
		  else
		  {
			  zero=0;
			  if ((index+i)>num_el)
			  {
				  zero=~zero;
			  }
		  }
		  d_v[index+i]=zero;
	  }
}

void __global__ shiftup_kernel(LongPointer d_v, LongPointer d_v_in,int i,unsigned int NN,
		unsigned int it=1)
{
	unsigned long long int teal, head;
	int num_el=i>>6;//номер элемента в большем слайсе
//int num_el1=(i+h)>>6;
	int num_bit_first= i % SIZE_OF_LONG_INT ; // номер бита в элементе, который станет первым в маленьком слайсе
//int num_bit_last = h % SIZE_OF_LONG_INT;
//printf("num_els %i (%i) bits from %i  \n",blockIdx.x +num_el,gridDim.x,num_bit_first);
	unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
	for(int i=0; i<it;i++)
	{
		head=d_v_in[index+num_el]>>(num_bit_first);
		if (index +num_el<NN)//?????????
		{
			teal = (index+1+num_el<NN)?(d_v_in[index+1+num_el]<<(SIZE_OF_LONG_INT-num_bit_first)):0;
			d_v[index]=head | teal;
		}
		else // обрезать последние биты от num_bit_last
		{
//	   teal=(1<<num_bit_last) -1;
			d_v[index]=0;//head & teal;
		}
	index++;
 }
}

void __global__ shiftdown_kernel(LongPointer d_v, LongPointer d_v_in,int i,unsigned int NN,
		unsigned int it=1)
{
	int num_el=i>>6;//номер элемента в большем слайсе
	int num_bit_first= i % SIZE_OF_LONG_INT ;
	unsigned long long int teal,head;
	unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
	for(int i=0; i<it;i++)
	{
		if (index >num_el)//?????????
		{      head =d_v_in[index-num_el]<<(num_bit_first);
			   teal =d_v_in[index-1-num_el]>>(SIZE_OF_LONG_INT-num_bit_first);
			   d_v[index]=head | teal;
		}
		else // обрезать последние биты от num_bit_last
		{
			   d_v[index]=(index==num_el)? (d_v_in[0]<<(num_bit_first)):0;//head & teal;
		}
		index++;
	}
}

void __global__ trim_kernel(LongPointer d_v, LongPointer d_v_in,int i,int h,unsigned int NN,
		unsigned int it=1)
{
	int num_el=(i-1)>>6;//номер первого элемента в большем слайсе
	int num_el1=(h-1)>>6; // номер последнего элемента в маленьком
	int num_el2=(i+h-1)>>6;// номер последнего элемента в большом слайсе
	int num_bit_first= i % SIZE_OF_LONG_INT -1; // номер бита в элементе, который станет первым в маленьком слайсе
	int num_bit_last = h % SIZE_OF_LONG_INT;

	unsigned long long int teal,head;
	unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
	for(int i=0; i<it;i++)
	{
		head =d_v_in[index+num_el]>>(num_bit_first);
		if (index +num_el< num_el2)
		{      head =d_v_in[index+num_el]>>(num_bit_first);
			   teal =d_v_in[index+1+num_el]<<(SIZE_OF_LONG_INT-num_bit_first);
			   d_v[index]=head | teal;
		}
		if (index==num_el1)
		{
			teal=1;
			teal=(num_bit_last==0)? ~0:((teal<<num_bit_last)-1);
				//	   long_to_binary(teal,prb,64);
				//	   printf("\n teal_up (%i):",num_bit_last);printf(prb);
			d_v[index]&=teal;
		}
		index++;
	}
}
