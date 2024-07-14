#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
//#include <time.h>

#include "table.h"

Table::Table(unsigned int k,unsigned int s)
   {
	length=k;
	NN=(k-1)/SIZE_OF_LONG_INT +1;
	blocks=min(NN,MAX_BLOCK);
    IT=(NN-1)/blocks+1;
    size=s;
    cudaMalloc(&d_v,s*NN*sizeof(unsigned long long int));
    }
void Table::GetCol(Slice* X,unsigned int i)
{
	assign_kernel<<<blocks,1>>>( X->get_device_pointer(),&(d_v[(i-1)*NN]),NN,IT);
}

void Table::SetCol(Slice* X,unsigned int i)
{
	assign_kernel<<<blocks,1>>>(&(d_v[(i-1)*NN]), X->get_device_pointer(),NN,IT);
}

__device__ void _getCol(LongPointer d_table, LongPointer d_slice,unsigned int i,
		unsigned int NN,unsigned int IT){_assign(d_slice,&(d_table[(i-1)*NN]),NN,IT); }

__device__ void _setCol(LongPointer d_table, LongPointer d_slice,unsigned int i,
		unsigned int NN,unsigned int IT){_assign(&(d_table[(i-1)*NN]),d_slice,NN,IT); }

__global__ void setRow_kernel(LongPointer p,int n,LongPointer d_v, int size,unsigned NN,unsigned int IT)
{
	int index=(threadIdx.x + blockIdx.x*blockDim.x)*IT;
	if (index>size-1) return;
	LongPointer d_rhs;//для каждой нити свой столбец

	int bit;
	unsigned int n_el1,n_i1;
	unsigned int n_el=(n-1)/SIZE_OF_LONG_INT;
	unsigned int n_i=(n-1)%SIZE_OF_LONG_INT;

	unsigned long long int tmp;
//#ifdef ttt
//	printf("threadIdx.x %d %d n %d \n",blockIdx.x,i,n);
//#endif
//	long_to_binary(*d_rhs,s);
	for(int i=0; i<IT;i++)
	{
		if (index<size)
		{
		//вычислить bit из строки d_v
			d_rhs = &p[index*NN];
			n_el1=(index)/SIZE_OF_LONG_INT;
			n_i1=(index)%SIZE_OF_LONG_INT;

			tmp=1;
			tmp=tmp<<n_i1;
			tmp=tmp&d_v[n_el1];
			bit=(tmp==0)?0:1;
		// нумерация с 0, поэтому не добавляется
		tmp=1;
		tmp=tmp<<n_i;
		if (bit==1)
			d_rhs[n_el]|=tmp;
		else
			d_rhs[n_el]&=~tmp;
		}
		index++;
//		printf("index=%d n_el1=%d (it=%d size=%d NN=%d %d)\n",index,n_el1,IT,size,NN, size*NN);
	}
}

void Table::SetRow(Slice* X,unsigned int i)
{
	// вычислить конфигурацию
 	unsigned int threads,it;
    threads = min(MAX_THREADS,size);
    it=(size-1)/threads+1;
	setRow_kernel<<<1,threads>>>(d_v,i,X->get_device_pointer(),size,NN,it);
}

__global__ void getRow_kernel1(LongPointer p,int n,LongPointer d_v, int size,unsigned NN,unsigned int IT)
{
	int index=(threadIdx.x + blockIdx.x*blockDim.x)*IT;
	if (index>size-1) return;
	LongPointer d_rhs;//для каждой нити свой столбец

	int bit;
	unsigned int n_el1,n_el=(n-1)/SIZE_OF_LONG_INT;
	unsigned int n_i1,n_i=(n-1)%SIZE_OF_LONG_INT;

	unsigned long long int tmp;
//#ifdef ttt
//	printf("threadIdx.x %d %d n %d \n",blockIdx.x,i,n);
//#endif
//	long_to_binary(*d_rhs,s);
	for(int i=0; i<IT;i++)
	{
		if (index<size)
		{   d_rhs = &p[index*NN];
			n_el1=(index)/SIZE_OF_LONG_INT;
			n_i1=(index)%SIZE_OF_LONG_INT;

			//вычислить bit из столбца d_rhs
			tmp=1;
			tmp=tmp<<n_i;
			tmp=tmp&d_rhs[n_el];
			bit=(tmp==0)?0:1;
//			printf("(%d,%d) ",index,bit);
		// нумерация с 0, поэтому не добавляется
		// изменяет только один бит, но теоретически так нельзя, лучше через reduce
		tmp=1;
		tmp=tmp<<n_i1;
		if (bit==1)
			d_v[n_el1]|=tmp;
		else
			d_v[n_el1]&=~tmp;
		}
		index++;
	}
}

void Table::GetRow1(Slice* X,unsigned int i)
{
 	unsigned int threads,it;
    threads = min(MAX_THREADS,NN);
    it=(size-1)/threads+1;
	getRow_kernel1<<<1,threads>>>(d_v,i,X->get_device_pointer(),size,NN,it);
}

__global__ void getRow_kernel(LongPointer p,int n,LongPointer d_v, int size,unsigned NN,unsigned int IT)
{   __shared__ unsigned long long int tmp[SIZE_OF_LONG_INT];
	int index=(threadIdx.x + blockIdx.x*blockDim.x)*IT;
	if (index>size-1) return;
	LongPointer d_rhs;//для каждой нити свой столбец

	int bit;
	unsigned int n_el1,n_el=(n-1)/SIZE_OF_LONG_INT;
	unsigned int n_i1,n_i=(n-1)%SIZE_OF_LONG_INT;


//#ifdef ttt
//	printf("threadIdx.x %d %d n %d \n",blockIdx.x,i,n);
//#endif
//	long_to_binary(*d_rhs,s);
	for(int i=0; i<IT;i++)
	{   tmp[threadIdx.x]=0;
		if (index<size)
		{   d_rhs = &p[index*NN];
			n_el1=(index)/SIZE_OF_LONG_INT;
			n_i1=(index)%SIZE_OF_LONG_INT;

			//вычислить bit из столбца d_rhs
			tmp[threadIdx.x]=1;
			tmp[threadIdx.x]=tmp[threadIdx.x]<<n_i;
			tmp[threadIdx.x]=tmp[threadIdx.x]&d_rhs[n_el];
			bit=(tmp[threadIdx.x]==0)?0:1;

		// нумерация с 0, поэтому не добавляется
		// изменяет только один бит, но теоретически так нельзя, лучше через reduce
			tmp[threadIdx.x]=(bit==1)?1:0;
			tmp[threadIdx.x]=tmp[threadIdx.x]<<n_i1;

//			printf("(%d,%d,%lx) ",index,bit,tmp[threadIdx.x]);
		__syncthreads();
		if (threadIdx.x<8){
			tmp[threadIdx.x*8]=tmp[threadIdx.x*8]|tmp[1+threadIdx.x*8]|tmp[2+threadIdx.x*8]
			                    |tmp[3+threadIdx.x*8]|tmp[4+threadIdx.x*8]|tmp[5+threadIdx.x*8]
			                    |tmp[6+threadIdx.x*8]|tmp[7+threadIdx.x*8];
//			printf("\n =%d:[%d,%d](<%d,%d>,%lx) ",index,blockIdx.x,threadIdx.x,threadIdx.x*8,7+threadIdx.x*8,tmp[threadIdx.x*8]);
			}
		__syncthreads();
		if (threadIdx.x==0)
			d_v[n_el1]=tmp[0]|tmp[8]|tmp[16]|tmp[24]|tmp[32]|tmp[40]|tmp[48]|tmp[56];
		}
		index++;
	}
}

void Table::GetRow(Slice* X,unsigned int i)
{
 	unsigned int blocks,it,NN1;
 	NN1=X->NN;
    blocks = min(MAX_BLOCK,NN1);
    it=(size-1)/(blocks*SIZE_OF_LONG_INT)+1;
//    printf("GetRow1 %d %d it=%d\n",size,blocks,it);
	getRow_kernel<<<blocks,SIZE_OF_LONG_INT>>>(d_v,i,X->get_device_pointer(),size,NN,it);
}

void Table::fprint(char *label)
  {
  	FILE * pFile;
  	char fname[30]{0};
  	strcat(fname,label);
  	strcat(fname,".dat");
  	pFile = fopen (fname,"w");
  	fprintf(pFile,"%s (%dx%d)\n%s\n",label,length,size);
// печать  64 строк слайса
  	char *d_str, *str;
  	  	 cudaMalloc(&d_str,SIZE_OF_LONG_INT*(size+1)*sizeof(char));
  	  	 str=new char[SIZE_OF_LONG_INT*(size+1)];
  	  for(int i=0;i<NN;i++)
  	  {
 // 	  	 print_block_kernel<<<blocks,1>>>(d_v,d_str,length,size,IT);
  	  	cudaMemcpy(str,d_str,SIZE_OF_LONG_INT*(size+1)*sizeof(char),cudaMemcpyDeviceToHost);
  	    fprintf(pFile,"%s\n",str);//есть вероятность, что не все '0x0A'- перенос строки
  	  }
//
  	fclose (pFile);
  	cudaFree(d_str);
  	delete[] str;
  }
