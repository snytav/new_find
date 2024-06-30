
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

 void __global__ digit_kernel(unsigned long long *w, unsigned long long *dig)
 {
 	dig[0]=__brevll(w[0]);
 //	dig[0]=w[0];
 }

 unsigned long long int Slice::ToDigit()
 { unsigned long long *d_dig1,res=0;
 	if (NN==1)
 	{  cudaMalloc(&d_dig1,sizeof(unsigned long long));
 		digit_kernel<<<1,1>>>(d_v,d_dig1);
 		cudaMemcpy(&res,d_dig1,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
 		res>>=(64-length);
 	}
 	return res;
 }
 void Slice::FromDigit(unsigned long long dig)
 {	 unsigned long long *d_dig1;
 		if (NN==1)
 		{   dig<<=(64-length);
 			cudaMalloc(&d_dig1,sizeof(unsigned long long));
 			cudaMemcpy(d_dig1,&dig,sizeof(unsigned long long),cudaMemcpyHostToDevice);
 			digit_kernel<<<1,1>>>(d_dig1,d_v);
 		}
 }


 void __global__ print_kernel(LongPointer d_v, char* d_str, unsigned int length,unsigned int N1,unsigned int it)
 {
	 unsigned long long int tmp,one=1;
	 unsigned int index=(blockIdx.x*blockDim.x+threadIdx.x)*it;
	 for(int i=0; i<it;i++)
		if (index+i<N1)
		{
			tmp=d_v[index+i];
			for (int j=0;j< SIZE_OF_LONG_INT;j++)
			{
				d_str[(index+i)*SIZE_OF_LONG_INT+j] =(tmp&one)?'1':'0';
				tmp=tmp>>1;
				if(((index+i)*SIZE_OF_LONG_INT+j)==length) d_str[length]=0;
			}
		}
 }

 void Slice::print(char *label)
 { char *d_str, *str;
 	 cudaMalloc(&d_str,NN*SIZE_OF_LONG_INT*sizeof(char));
 	 str=new char[NN*SIZE_OF_LONG_INT];
 	 print_kernel<<<blocks,1>>>(d_v,d_str,length,NN,IT);
 	cudaMemcpy(str,d_str,NN*SIZE_OF_LONG_INT*sizeof(char),cudaMemcpyDeviceToHost);
 	printf("%s \n%s\n",label,str);

 }

 void Slice::fprint(char *label)
  { char *d_str, *str;
  	 cudaMalloc(&d_str,NN*SIZE_OF_LONG_INT*sizeof(char));
  	 str=new char[NN*SIZE_OF_LONG_INT];
  	 print_kernel<<<blocks,1>>>(d_v,d_str,length,NN,IT);
  	cudaMemcpy(str,d_str,NN*SIZE_OF_LONG_INT*sizeof(char),cudaMemcpyDeviceToHost);

  	FILE * pFile;
  	char fname[30]{0};
  	strcat(fname,label);
  	strcat(fname,".dat");
  	pFile = fopen (fname,"w");
  	fprintf(pFile,"%s (%d)\n%s\n",label,length,str);
  	fclose (pFile);
  }
