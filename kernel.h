/*
 *  kernel.h
 *
 *  библиотека содержит реализацию функций на устройстве
 */
#define MAX_THREADS 1024
#define SIZE_OF_LONG_INT 64

typedef unsigned long long int *LongPointer;
/*
* критичные к синхронизации,
* <<<1,threads>>>, где threads вычисляется в соответствующей процедуре класса Slice
* 		threads = min(MAX_THREADS,N1);
* 		it=(N1-1)/threads+1;
*
* 		перед ними нужно вызвать tail_kernel<<<1,1>>>(d_v,length,N1)
*/
void __global__ find_kernel(LongPointer d_v, unsigned int length,unsigned int N1,unsigned int it, int* res);
void __global__ some_kernel(LongPointer d_v,unsigned int N1,unsigned int it, int*res);
void __global__ zero_kernel(LongPointer d_v,unsigned int N1,unsigned int it, int*res);
void __global__ numb_kernel(LongPointer d_v, unsigned int length,unsigned int N1,unsigned int it, int* res);

//shiftup(k), shiftdown(k)
//getbit(k) и setbit(k,bit)
void __global__ tail_kernel(LongPointer d_v, unsigned int length,unsigned int N1);


/*
 * некритичные к синхронизации, параметры блока вычисляются в соответствующей процедуре класса Slice
 * <<<blocks,1>>>
 */
__device__ void _assign(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void assign_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);

__device__ void _set( LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void set_kernel(LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__device__ void _clr( LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void clr_kernel(LongPointer d_v1, unsigned int NN,
unsigned int IT=1);

__device__ void _not( LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void not_kernel(LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__device__ void _and(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void and_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__device__ void _or(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void or_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__device__ void _xor(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
__global__ void xor_kernel(LongPointer d_v, LongPointer d_v1, unsigned int NN,
unsigned int IT=1);
