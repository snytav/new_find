#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <time.h>
#include "slice.h"
#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#define MAX 999999
#define N 100000
#define L  N*64-32

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
	numb_kernel<<<1,threads>>>(d_v,N*64-32,N1,it,d_res);
	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	printf("NUMB=%d\n",h_res);

	find_kernel<<<1,threads>>>(d_v,(N*64),N1,it,d_res);
	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	return h_res;
}


int main()
{
 /*   unsigned long long h_v[N];/* = {0xABCDABCDABCD0000, 0x0F08000800080070,
                                0xABCDABCDAB900000, 0x0F08000800080700,
                                0xABCDABCDABC80000, 0x0F08000807000000,
                                0xABCDABCDAB001000, 0x0F08000800080500 };*/
/*
    unsigned long long* d_v;
//  int *d_res,h_res;
//   cudaMalloc(&d_res, sizeof(int));

    for (int i = 0; i < N; i++)
    {
        h_v[i] = (i >0) ? 0x8000000000000008 : 0;//rand() % MAX + 1;
        int sh = rand() % 32 + 1;
        //h_v[i] <<= sh;
        //printf("%d %30lx shift %d \n",i,h_v[i],sh);
    }
    cudaMalloc(&d_v, N * sizeof(unsigned long long));

    cudaMemcpy(d_v, h_v, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
 // печать с устройства
    cudaPrintfInit();

    printf("FND= %d \n",FND(d_v));

    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
*/
    Slice X(L);
    X.fprint("X1");
    X.SET();
    X.fprint("X2");
    printf("%d %d",X.length,X.NUMB());
    return 0;
}
 
