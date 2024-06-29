/*
 *  kernel.h
 *
 *  библиотека содержит реализацию функций на устройстве
 */
#define MAX_THREADS 1024
#define SIZE_OF_LONG_INT 64

void __global__ find_kernel(unsigned long long* d_v, unsigned int length,unsigned int N1,unsigned int it, int* res);
void __global__ some_kernel(unsigned long long* d_v,unsigned int N1,unsigned int it, int*res);
void __global__ zero_kernel(unsigned long long* d_v,unsigned int N1,unsigned int it, int*res);
void __global__ numb_kernel(unsigned long long* d_v, unsigned int length,unsigned int N1,unsigned int it, int* res);
