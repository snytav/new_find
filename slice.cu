
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"
#include <stdio.h>
#include <string>
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
 Slice::~Slice()
   {
//	 if(d_v!=NULL) cudaFree(d_v);
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
 	int static *d_res=NULL,h_res;
 	if (d_res==NULL)cudaMalloc(&d_res, sizeof(int));
//cudaError_t err = cudaGetLastError();
// 	if (err>0) printf("\n errors FND malloc %d %s \n",err,cudaGetErrorString(err));
 	{
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;
// 	   printf("FND: N=%d threads=%d,IT=%d \n",NN,threads,it);
 	}

 	find_kernel<<<1,threads>>>(d_v,length,NN,it,d_res);
//err = cudaGetLastError();
 //	 	if (err>0) printf("\n errors FND count %d %s \n",err,cudaGetErrorString(err));
 	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
//err = cudaGetLastError();
// 	 	if (err>0) printf("\n errors FND copy memory %d %s \n",err,cudaGetErrorString(err));
 	return h_res;
 }

 unsigned int Slice::STEP()
 {
 	// вычислить конфигурацию
 	unsigned int threads,it;
 	int static *d_res=NULL,h_res;
 	if (d_res==NULL) cudaMalloc(&d_res, sizeof(int));

 	{
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;
 //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
 	}

 	find_kernel<<<1,threads>>>(d_v,length,NN,it,d_res);
 	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

 	if (h_res>0) setbit(h_res,0);
 	return h_res;
 }

 unsigned int Slice::NUMB()
 {
 	// вычислить конфигурацию
 	unsigned int threads,it;
 	int static *d_res=NULL,h_res;
 	if (d_res==NULL)cudaMalloc(&d_res, sizeof(int));

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
 	int static *d_res=NULL,h_res;
 	if (d_res==NULL) cudaMalloc(&d_res, sizeof(int));

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
 /* 	// вычислить конфигурацию
  	unsigned int threads,it;
  	static int *d_res=NULL;
  	int h_res;
  	if (d_res==NULL) cudaMalloc(&d_res, sizeof(int));

  	{
  	   threads = min(MAX_THREADS,NN);
  	   it=(NN-1)/threads+1;
  //	   printf("N=%d threads=%d,IT=%d \n",NN,threads,it);
  	}

/*  	tail_kernel<<<1,1>>>(d_v,length,NN);
  	zero_kernel<<<1,threads>>>(d_v,NN,it,d_res);
  	cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
  	*/
//  	printf("ZERO: FND %d",FND());
  	return FND()==0;
  }

 void __global__ digit_kernel(unsigned long long *w, unsigned long long *dig)
 {
 	dig[0]=__brevll(w[0]);
 //	dig[0]=w[0];
 }

 unsigned long long int Slice::ToDigit()
 { unsigned long long static *d_dig1;
   unsigned long long res=0;
 	if (NN==1)
 	{ if (d_dig1==NULL) cudaMalloc(&d_dig1,sizeof(unsigned long long));
 		digit_kernel<<<1,1>>>(d_v,d_dig1);
 		cudaMemcpy(&res,d_dig1,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
 		res>>=(64-length);
 	}
 	return res;
 }
 void Slice::FromDigit(unsigned long long dig)
 {	 unsigned long long static *d_dig1;
 		if (NN==1)
 		{   dig<<=(64-length);
 		if (d_dig1==NULL) cudaMalloc(&d_dig1,sizeof(unsigned long long));
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

 void Slice::print(const char *label)
 { 	 //static
	 char *d_str=NULL;
 	 char *str;
 	if (d_str==NULL)cudaMalloc(&d_str,(NN*SIZE_OF_LONG_INT+1)*sizeof(char));
 	 str=new char[NN*SIZE_OF_LONG_INT];
 	 print_kernel<<<blocks,1>>>(d_v,d_str,length,NN,IT);
 	cudaMemcpy(str,d_str,(NN*SIZE_OF_LONG_INT+1)*sizeof(char),cudaMemcpyDeviceToHost);
 	printf("%s \n%s\n",label,str);
 	cudaFree(d_str);
 }

 void Slice::fprint(const char *label)
  { static char *d_str=NULL;
	 char *str;
	if (d_str==NULL)cudaMalloc(&d_str,NN*SIZE_OF_LONG_INT*sizeof(char));
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
 // 	cudaFree(d_str);
  }

 void Slice::setbit(unsigned int n, int bit)
 {
	if((n>0)&& (n<=length)) setbit_kernel<<<1,1>>>(d_v,n,bit);
	else printf("setbit: incorrect position %d \n",n);
 }

 int Slice::getbit(unsigned int n)
  {
	 static int *d_res=NULL,h_res;
	  if (d_res==NULL)cudaMalloc(&d_res, sizeof(int));

	 if((n>0)&& (n<=length))getbit_kernel<<<1,1>>>(d_v,n,d_res);
	 else printf("getbit: incorrect position %d \n",n);
 	 cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
 	 return h_res;
  }

 void Slice::MASK(int i)
 {
 	mask_kernel<<<blocks,1>>>(d_v,i,NN,IT);
 }
 void Slice::MASK1(int i)
  {
  	mask1_kernel<<<blocks,1>>>(d_v,i,NN,IT);
  }
 void Slice::shift_up(int i,Slice *s)
 {
 	unsigned long long int *d_v_in;

	// вычислить конфигурацию
 	unsigned int threads,it;
 	   threads = min(MAX_THREADS,NN);
 	   it=(NN-1)/threads+1;

 	d_v_in= s->get_device_pointer();
	tail_kernel<<<1,1>>>(d_v_in,length,NN);
 	shiftup_kernel<<<1,threads>>>(d_v,d_v_in,i,NN,it);
 }

 void Slice::shift_down(int i,Slice *s)
  {
  	unsigned long long int *d_v_in;

 	// вычислить конфигурацию
  	unsigned int threads,it;
  	   threads = min(MAX_THREADS,NN);
  	   it=(NN-1)/threads+1;

  	d_v_in= s->get_device_pointer();
 	tail_kernel<<<1,1>>>(d_v_in,length,NN);
  	shiftdown_kernel<<<1,threads>>>(d_v,d_v_in,i,NN,it);
  }

 void Slice::trim(int i,int h, Slice *s)
  {
  	unsigned long long int *d_v_in;

 	// вычислить конфигурацию
  	unsigned int threads,it;
  	   threads = min(MAX_THREADS,NN);
  	   it=(NN-1)/threads+1;

  	d_v_in= s->get_device_pointer();
 	tail_kernel<<<1,1>>>(d_v_in,length,NN);
  	trim_kernel<<<1,threads>>>(d_v,d_v_in,i,h,NN,it);
  }
