
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
//#include <time.h>

#include "slice.h"

 Slice::Slice(unsigned int k)
   {
	length=k;
	NN=(k-1)/SIZE_OF_LONG_INT +1;
	blocks=min(NN,MAX_BLOCK);
    IT=(NN-1)/blocks+1;

    cudaMalloc(&d_v,NN*sizeof(unsigned long long int));
    }

 void Slice::ASSIGN(Slice *X)
 {
	 assign_kernel<<<blocks,1>>>(d_v, X->get_device_pointer(),NN,IT);
 }

 void Slice::AND(Slice *X)
  {
 	 and_kernel<<<blocks,1>>>(d_v, X->get_device_pointer(),NN,IT);
  }

 void Slice::OR(Slice *X)
   {
  	 or_kernel<<<blocks,1>>>(d_v, X->get_device_pointer(),NN,IT);
   }

 void Slice::XOR(Slice *X)
   {
  	 xor_kernel<<<blocks,1>>>(d_v, X->get_device_pointer(),NN,IT);
   }

 void Slice::NOT()
   {
  	 not_kernel<<<blocks,1>>>(d_v,NN,IT);
   }

 void Slice::SET()
   {
  	 set_kernel<<<blocks,1>>>(d_v,NN,IT);
   }

 void Slice::CLR()
   {
  	 clr_kernel<<<blocks,1>>>(d_v,NN,IT);
   }

 unsigned int Slice::FND()
 {
 	// вычислить конфигурацию
 	unsigned int threads,it;
 	int *d_res,h_res;
 	cudaMalloc(&d_res, sizeof(int));

 	{
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;
 //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
 	}

 	find_kernel<<<1,threads>>>(d_v,length,NN,it,d_res);
 	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
 	return h_res;
 }

 unsigned int Slice::NUMB()
 {
 	// вычислить конфигурацию
 	unsigned int threads,it;
 	int *d_res,h_res;
 	cudaMalloc(&d_res, sizeof(int));

 	{
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;
 //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
 	}

 	tail_kernel<<<1,1>>>(d_v,length,NN);
 	numb_kernel<<<1,threads>>>(d_v,length,NN,it,d_res);
 	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
 	return h_res;
 }

 bool Slice::SOME()
 {
 	// вычислить конфигурацию
 	unsigned int threads,it;
 	int *d_res,h_res;
 	cudaMalloc(&d_res, sizeof(int));

 	{
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;
 //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
 	}

 	tail_kernel<<<1,1>>>(d_v,length,NN);
 	some_kernel<<<1,threads>>>(d_v,NN,it,d_res);
 	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
 	return h_res==1;
 }

 bool Slice::ZERO()
  {
  	// вычислить конфигурацию
  	unsigned int threads,it;
  	int *d_res,h_res;
  	cudaMalloc(&d_res, sizeof(int));

  	{
  	   threads = min(MAX_THREADS,NN);
  	   it=(NN-1)/threads+1;
  //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
  	}

  	tail_kernel<<<1,1>>>(d_v,length,NN);
  	zero_kernel<<<1,threads>>>(d_v,NN,it,d_res);
  	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
  	return h_res==1;
  }
