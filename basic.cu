#include "basic.h"
#include <stdio.h>
#include <string.h>

LongPointer d_aux_slice=NULL;

int InitAuxSlices(unsigned  int NN)
{
	if (d_aux_slice==NULL) cudaMalloc(&d_aux_slice,AUX_COUNT*NN*sizeof(unsigned long long int));
	cudaError_t err = cudaGetLastError();
	return err;
}

void MATCH(Table *tab, Slice *X, Slice *w, Slice *Z)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
//	cudaFuncSetAttribute(match_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NN*AUX_COUNT*sizeof(unsigned long long int));
	match_kernel<<<blocks,1>>>(tab->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Z->get_device_pointer(),tab->size,NN,IT,d_aux_slice);
	//<<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>(tab->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Z->get_device_pointer(),tab->size,NN,IT);
//    cudaError_t err = cudaGetLastError();
 //   if (err>0) printf("errors after MATCH %d %s\n",err,cudaGetErrorString(err));
}

__global__ void match_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{  // if(blockIdx.x==0)printf("MATCH start \n");
	match(d_tab,d_x,d_w,d_z,size,NN,IT,aux_slice);
}


__device__ void match(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_y=&aux_slice[0];
	unsigned int i;

	_assign(d_z, d_x,NN,IT);
	for(i=1;i<=size;i++)
	{
		_getCol(d_tab,d_y,i,NN,IT);
		if (_getbit(d_w, i)==1)
		{
			_and(d_z,d_y,NN,IT);

		}
		else
		{
			_not(d_y,NN,IT);
			_and(d_z,d_y,NN,IT);
		}
	}
}

void GEL(Table *tab, Slice *w, Slice *X, Slice *Y)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	gel_kernel<<<blocks,1>>>(tab->get_device_pointer(),w->get_device_pointer(),X->get_device_pointer(),Y->get_device_pointer(),tab->size,NN,IT,d_aux_slice);

	cudaError_t err = cudaGetLastError();
	if (err>0) printf("errors after GEL %d %s\n",err,cudaGetErrorString(err));
}
__global__ void gel_kernel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	gel(d_tab,d_w,d_x,d_y,size,NN,IT,aux_slice);
}

__device__ void gel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
      LongPointer d_z=&aux_slice[0];
      LongPointer d_b=&aux_slice[NN];

      _clr(d_x,NN,IT);
	  _clr(d_y,NN,IT);
	  _set(d_z,NN,IT);
      for(int i=1;i<=size;i++)
      {
    	  _getCol(d_tab,d_b,i,NN,IT);

		 if (_getbit(d_w, i)==1)
		 {
//	(* In the slice Y we accumulate position of those i-th rows for which row(i,T)<w. *)
		   _not(d_b,NN,IT);
		   _and(d_b,d_z,NN,IT);
		   _or(d_y,d_b,NN,IT);
//	printf("%i:l=%llu \n",blockIdx.x,b);
		 }
		 else{
			_and(d_b,d_z,NN,IT);
			_or(d_x,d_b,NN,IT);
//	printf("%i:g=%llu \n",blockIdx.x,b);
//	(* In the slice X we accumulate position of those i-th rows for which row(i,T)>w. *)
		 }
		 _not(d_b,NN,IT);
		 _and(d_z,d_b,NN,IT);
//	(* Positions of the selected rows are deleting from the slice Z. *)
      }
}

void LESS(Table *T, Slice *X, Slice *w,Slice *Y)
{
	unsigned int NN, IT,blocks;
		NN=X->NN;
		IT=X->IT;
		blocks=X->blocks;
		less_kernel<<<blocks,1>>>(T->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Y->get_device_pointer(),T->size,NN,IT,d_aux_slice);

//		 cudaError_t err = cudaGetLastError();
//		 if (err>0) printf("errors after LESS %d %s\n",err,cudaGetErrorString(err));
}
__global__ void less_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	less(d_tab,d_x,d_w,d_y,size,NN,IT,aux_slice);
}
__device__ void less(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	 LongPointer d_y1=&aux_slice[0];
	 LongPointer d_b=&aux_slice[NN];
	 LongPointer d_yy=&aux_slice[2*NN];

	 _assign(d_y1, d_x,NN,IT);
	 _clr(d_yy,NN,IT);
	 for(int i=1;i<=size;i++)
	 {
	    _getCol(d_tab,d_b,i,NN,IT);
		 if (_getbit(d_w, i)==1)
		 {
			 _not(d_b,NN,IT);
			 _and(d_b,d_y1,NN,IT);
			 _or(d_yy,d_b,NN,IT);
		 }
		 _not(d_b,NN,IT);
		 _and(d_y1,d_b,NN,IT);
	 }
	 _assign(d_y, d_yy,NN,IT);
}

void GREAT(Table *T, Slice *X, Slice *w,Slice *Y)
{
	unsigned int NN, IT,blocks;
		NN=X->NN;
		IT=X->IT;
		blocks=X->blocks;
		great_kernel<<<blocks,1>>>(T->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Y->get_device_pointer(),T->size,NN,IT,d_aux_slice);
//		cudaError_t err = cudaGetLastError();
//		if (err>0) printf("errors after GREAT %d %s\n",err,cudaGetErrorString(err));
}
__global__ void great_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	great(d_tab,d_x,d_w,d_y,size,NN,IT,aux_slice);
}
__device__ void great(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	 LongPointer d_c=&aux_slice[0];
	 LongPointer d_b=&aux_slice[NN];

	 _assign(d_b, d_x,NN,IT);
	 _clr(d_y,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	 {
		 _getCol(d_tab,d_c,i,NN,IT);
		 if (_getbit(d_w, i)==0)
		 {
			 _and(d_c,d_b,NN,IT);
			 _xor(d_y,d_c,NN,IT);
			 _not(d_c,NN,IT);
		 }
		 _and(d_b,d_c,NN,IT);
	 }
}

__device__ bool some(LongPointer d_y,unsigned int length,unsigned int NN, unsigned int it)
{
	__shared__ unsigned int tmp;
	unsigned int n = threadIdx.x;
	__syncthreads();
	if (n==0)
	{
		tmp=0;
	//------------tail---------------------
		unsigned long long int tail=1;
		tail=(tail<<(length%SIZE_OF_LONG_INT))-1;
		if (tail==0) tail=~0;
		d_y[NN-1]=d_y[NN-1]&tail;
	//------------tail end---------------------
	}
	__syncthreads();
 	int i=0;
 	while((!tmp)&&(i<it))
 	{
 		if(d_y[n*it+i]>0) tmp=1;
 		i++;
 	}
 	__syncthreads();
	//-----------------------------------------
 	return tmp>0;
}

__global__ void min_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_z,
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	 LongPointer d_y=&aux_slice[0];

	 _assign(d_z, d_x,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	{
		 _getCol(d_tab,d_y,i,NN,IT);
		 _not(d_y,NN,IT);
		 _and(d_y,d_z,NN,IT);
		 if (some(d_y,length,NN,IT)) _assign(d_z, d_y,NN,IT);
	 }
}

void MIN(Table *T, Slice *X, Slice*Z)
{
	unsigned int NN, IT,threads;

	NN=X->NN;
	threads = min(MAX_THREADS,NN);
	IT=(NN-1)/threads+1;
	min_kernel<<<1,threads>>>(T->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->length,T->size,NN,IT,d_aux_slice);
	//<<<1,threads,NN*sizeof(unsigned long long int)>>>
	cudaError_t err = cudaGetLastError();
	if (err>0) printf("errors after MIN %d %s\n",err,cudaGetErrorString(err));
}

__global__ void max_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_z,
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	 LongPointer d_y=&aux_slice[0];

	 _assign(d_z, d_x,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	{
		 _getCol(d_tab,d_y,i,NN,IT);
		 _and(d_y,d_z,NN,IT);

		 if (some(d_y,length,NN,IT)) _assign(d_z, d_y,NN,IT);
	 }
}

void MAX(Table *T, Slice *X, Slice*Z)
{
	unsigned int NN, IT,threads;

	NN=X->NN;
	threads = min(MAX_THREADS,NN);
	IT=(NN-1)/threads+1;
	max_kernel<<<1,threads>>>(T->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->length,T->size,NN,IT,d_aux_slice);

	cudaError_t err = cudaGetLastError();
	    if (err>0) printf("errors after MAX %d %s\n",err,cudaGetErrorString(err));
}

void SETMIN(Table *T, Table *F, Slice *X, Slice *Z)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	setmin_kernel<<<blocks,1>>>(T->get_device_pointer(),F->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->size,NN,IT,d_aux_slice);

	cudaError_t err = cudaGetLastError();
	    if (err>0) printf("errors after SETMIN %d\n",err);
}

__global__ void setmin_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)//<<<NN,1>>>
{
	setmin(d_t,d_f,d_x,d_z,size,NN,IT,aux_slice);
}

__device__ void setmin(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_b=&aux_slice[0];
	LongPointer d_m=&aux_slice[NN];
	LongPointer d_y=&aux_slice[2*NN];

	_clr(d_z,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	{
	  _getCol(d_t,d_b,i,NN,IT);
	  _getCol(d_f,d_y,i,NN,IT);
	  _assign(d_m,d_b,NN,IT);
	  _xor(d_m,d_y,NN,IT);
	  _and(d_m,d_x,NN,IT);
	  // In the slice M , we save positions of the rows
	  //where ROW(j, T ) != ROW(j, F ) and X(j) = 1 .
	  _not(d_b,NN,IT);
	  _and(d_b,d_y,NN,IT);
	  _and(d_b,d_x,NN,IT);
	  // In the slice B, we save positions of the rows
	  //where ROW(j, T ) < ROW(j, F ) and X(j) = 1 .
	  _or(d_z,d_b,NN,IT);
	  _not(d_m,NN,IT);
	  _and(d_x,d_m,NN,IT);
	}
}

void SETMAX(Table *T, Table *F, Slice *X, Slice *Z)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	setmax_kernel<<<blocks,1>>>(T->get_device_pointer(),F->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->size,NN,IT,d_aux_slice);

	cudaError_t err = cudaGetLastError();
	    if (err>0) printf("errors after SETMAX %d %s\n",err,cudaGetErrorString(err));
}

__global__ void setmax_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)//<<<NN,1>>>
{
	setmax(d_t,d_f,d_x,d_z,size,NN,IT,aux_slice);
}

__device__ void setmax(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_b=&aux_slice[0];
	LongPointer d_m=&aux_slice[NN];
	LongPointer d_y=&aux_slice[2*NN];

	_clr(d_z,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	{
	  _getCol(d_f,d_b,i,NN,IT);
	  _getCol(d_t,d_y,i,NN,IT);
	  _assign(d_m,d_b,NN,IT);
	  _xor(d_m,d_y,NN,IT);
	  _and(d_m,d_x,NN,IT);
	  // In the slice M , we save positions of the rows
	  //where ROW(j, T ) != ROW(j, F ) and X(j) = 1 .
	  _not(d_b,NN,IT);
	  _and(d_b,d_y,NN,IT);
	  _and(d_b,d_x,NN,IT);
	  // In the slice B, we save positions of the rows
	  //where ROW(j, T ) < ROW(j, F ) and X(j) = 1 .
	  _or(d_z,d_b,NN,IT);
	  _not(d_m,NN,IT);
	  _and(d_x,d_m,NN,IT);
	}
}


void HIT(Table *T, Table *F, Slice *X, Slice *Z)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	hit_kernel<<<blocks,1>>>(T->get_device_pointer(),F->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->size,NN,IT,d_aux_slice);
	cudaError_t err = cudaGetLastError();
		    if (err>0) printf("\n errors after HIT %d %s\n",err,cudaGetErrorString(err));
}
__global__ void hit_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)//<<<NN,1>>>
{
	hit(d_t,d_f,d_x,d_z,size,NN,IT,aux_slice);
}

__device__ void hit(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_b=&aux_slice[0];
	LongPointer d_y=&aux_slice[NN];

	_assign(d_z,d_x,NN,IT);
	 for(unsigned int i=1;i<=size;i++)
	{
	  _getCol(d_t,d_b,i,NN,IT);
	  _getCol(d_f,d_y,i,NN,IT);
	  _xor(d_y,d_b,NN,IT);
	  _not(d_y,NN,IT);
	  _and(d_z,d_y,NN,IT);
	}
}

void ADDV(Table *T, Table *F, Slice *X, Table *S)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	addv_kernel<<<blocks,1>>>(T->get_device_pointer(),F->get_device_pointer(),X->get_device_pointer(),S->get_device_pointer(),T->size,NN,IT,d_aux_slice);
}
__global__ void addv_kernel(LongPointer d_t,LongPointer d_r,LongPointer d_x,LongPointer d_s,
					unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)//<<<NN,1>>>
{
	addv(d_t,d_r,d_x,d_s,size,NN,IT,aux_slice);
}

__device__ void addv(LongPointer d_t,LongPointer d_f,LongPointer d_x,LongPointer d_s,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_y=&aux_slice[0];
	LongPointer d_z=&aux_slice[NN];
	LongPointer d_m=&aux_slice[2*NN];
	LongPointer d_b=&aux_slice[3*NN];
	_clr(d_m,NN,IT);
	 for(unsigned int i=size;i>0;i--)
	{
		 _getCol(d_t,d_y,i,NN,IT);
		 _and(d_y,d_x,NN,IT);
		 _getCol(d_f,d_z,i,NN,IT);
		 _and(d_z,d_x,NN,IT);
		 _assign(d_b,d_y,NN,IT);
		 _and(d_b,d_z,NN,IT);
		 _xor(d_z,d_y,NN,IT);
		 _xor(d_m,d_z,NN,IT);
		 _setCol(d_s,d_z,i,NN,IT);
		 _assign(d_y,d_z,NN,IT);
		 _and(d_y,d_m,NN,IT);
		 _or(d_b,d_y,NN,IT);
		 _assign(d_m,d_b,NN,IT);
	}
	//если SOME(B) нужно встваить в результирующую таблицу, но просто передаем наверх.
}

void ADDC(Table *T, Slice *w, Slice *X, Table *S)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
//	Slice B(X->length);
	addc_kernel<<<blocks,1>>>(T->get_device_pointer(),w->get_device_pointer(),X->get_device_pointer(),S->get_device_pointer(),T->size,NN,IT,d_aux_slice);
//	if(B->SOME()) puts("ADDC: size error");
//	cudaError_t err = cudaGetLastError();
//				    if (err>0) printf("\n errors after ADDC %d %s\n",err,cudaGetErrorString(err));
}
__global__ void addc_kernel(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_s,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	addc(d_t,d_w,d_x,d_s,size,NN,IT,aux_slice);
}

__device__ void addc(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_s,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_y=&aux_slice[0];
	LongPointer d_m=&aux_slice[NN];
	LongPointer d_b=&aux_slice[2*NN];
	  _clr(d_b,NN,IT);
	  for (unsigned int i=size;i>0;i--)
	  {
		  _getCol(d_t,d_y,i,NN,IT);
		  _assign(d_m,d_b,NN,IT);
		  _xor(d_m,d_y,NN,IT);

	      if (_getbit(d_w,i)==0)
	      {
	    	  _and(d_b,d_y,NN,IT);
	      }
	      else
	      {
	    	  _not(d_m,NN,IT);
	    	  _or(d_b,d_y,NN,IT);
	      }
	      _and(d_m,d_x,NN,IT);
	      _setCol(d_s,d_m,i,NN,IT);
	      _and(d_b,d_x,NN,IT);
	   }
}

void ADDC1(Table *T, Slice *w, Slice *X)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
//	Slice B1(X->length);
//	printf("addc1 \n sz=%i NN=%i lg=%i\n",T->size,NN,T->length);

		addc1_kernel<<<blocks,1>>>(T->get_device_pointer(),w->get_device_pointer(),X->get_device_pointer(),T->size,NN,IT,d_aux_slice);

	/*	if(B.SOME())
	{
		puts("ADDC1: size error");
//		B.print("ADDC1");
	}
	*/
	cudaError_t err = cudaGetLastError();
				    if (err>0) printf("\n errors after ADDC1 %d %s\n",err,cudaGetErrorString(err));
}
__global__ void addc1_kernel(LongPointer d_t,LongPointer d_w,LongPointer d_x,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	addc1(d_t,d_w,d_x,size,NN,IT,aux_slice);
}

__device__ void addc1(LongPointer d_t,LongPointer d_w,LongPointer d_x,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_y=&aux_slice[0];
	LongPointer d_m=&aux_slice[NN];
	LongPointer d_col=&aux_slice[2*NN];
	LongPointer d_b=&aux_slice[3*NN];
	  _clr(d_b,NN,IT);
	  for (unsigned int i=size;i>0;i--)
	  {
		  _getCol(d_t,d_y,i,NN,IT);
		  _assign(d_m,d_b,NN,IT);
		  _xor(d_m,d_y,NN,IT);

	      if (_getbit(d_w,i)==0)
	      {
	    	  _and(d_b,d_y,NN,IT);
	      }
	      else
	      {
	    	  _not(d_m,NN,IT);
	    	  _or(d_b,d_y,NN,IT);
	      }
	      _and(d_m,d_x,NN,IT);
	      _assign(d_col,d_x,NN,IT);
	      _not(d_col,NN,IT);
	      _and(d_col,d_y,NN,IT);
	      _or(d_col,d_m,NN,IT);
	      _setCol(d_t,d_col,i,NN,IT);
	      _and(d_b,d_x,NN,IT);
	   }
}


void ADDROW(Table *T, Table *R, unsigned int i, Slice *X)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
//	Slice B(X->length);
//	printf("addc1 \n sz=%i NN=%i lg=%i\n",T->size,NN,T->length);
	addrow_kernel<<<blocks,1>>>(T->get_device_pointer(),R->get_device_pointer(),i,X->get_device_pointer(),T->size,NN,R->NN,IT,d_aux_slice);
/*	if(B.SOME())
	{
		puts("ADDC1: size error");
//		B.print("ADDC1");
	}
	*/
	cudaError_t err = cudaGetLastError();
				    if (err>0) printf("\n errors ADDROW i=%i %d %s \n",i,err,cudaGetErrorString(err));
}

__global__ void addrow_kernel(LongPointer d_t,LongPointer d_r,unsigned int i,LongPointer d_x,
				unsigned int size,unsigned int NN,unsigned int NN2, unsigned int IT,LongPointer aux_slice)
{
	addrow(d_t,d_r,i,d_x,size,NN,NN2,IT,aux_slice);
}
//d_b перенос на предыдущий разряд
__device__ void addrow(LongPointer d_t,LongPointer d_r,unsigned int j,LongPointer d_x,
		unsigned int size,unsigned int NN,unsigned int NN2, unsigned int IT,LongPointer aux_slice)
{
	LongPointer d_y=&aux_slice[0];
	LongPointer d_m=&aux_slice[NN];
	LongPointer d_col=&aux_slice[2*NN];
	LongPointer d_b=&aux_slice[3*NN];
	LongPointer d_col1;
//	unsigned int vol;

	  _clr(d_b,NN,IT);
	  for (unsigned int i=size;i>0;i--)
	  {
		  _getCol(d_t,d_y,i,NN,IT);
		  _assign(d_m,d_b,NN,IT);
		  _xor(d_m,d_y,NN,IT);
		  d_col1=&(d_r[(i-1)*NN2]);
	      if (_getbit(d_col1,j)==0)
	      {
	    	  _and(d_b,d_y,NN,IT);
	      }
	      else
	      {
	    	  _not(d_m,NN,IT);
	    	  _or(d_b,d_y,NN,IT);
	      }
	      _and(d_m,d_x,NN,IT);
	      _assign(d_col,d_x,NN,IT);
	      _not(d_col,NN,IT);
	      _and(d_col,d_y,NN,IT);
	      _or(d_col,d_m,NN,IT);
	      _setCol(d_t,d_col,i,NN,IT);
	      _and(d_b,d_x,NN,IT);
	   }
}
void CLEAR(Table *T)
{
	unsigned int NN, IT,blocks,sz;
	NN=T->NN;
	IT=T->IT;
	blocks=T->blocks;
	sz=min(MAX_THREADS,T->size);
	clear_kernel<<<blocks,sz>>>(T->get_device_pointer(),T->size,NN,IT);
}
__global__ void  clear_kernel(LongPointer d_v,unsigned int size,unsigned int NN, unsigned int IT)
{
	clear(d_v,size,NN,IT);
}
__device__ void clear(LongPointer d_v,unsigned int size,unsigned int NN, unsigned int it)
{
	unsigned int index=blockIdx.x*it;
	unsigned int col_numb=threadIdx.x;
	while(size>= blockDim.x)
	{
		for(int i=0; i<it;i++)
			if ((index+i<NN)&&(col_numb<size)) d_v[col_numb*NN+index+i] =0;
		size-=blockDim.x;
		col_numb+=blockDim.x;
	}
}
void TCOPY(Table *T, Table *F)
{
	unsigned int NN, IT,blocks,sz;
	NN=T->NN;
	IT=T->IT;
	blocks=T->blocks;
	sz=min(MAX_THREADS,T->size);
	tcopy_kernel<<<blocks,sz>>>(T->get_device_pointer(),F->get_device_pointer(),T->size,NN,IT);
}
__global__ void  tcopy_kernel(LongPointer d_t, LongPointer d_f,unsigned int size,unsigned int NN, unsigned int IT)
{
	tcopy(d_t,d_f,size,NN,IT);
}
__device__ void tcopy(LongPointer d_t, LongPointer d_f,unsigned int size,unsigned int NN, unsigned int it)
{
	unsigned int index=blockIdx.x*it;
	unsigned int col_numb=threadIdx.x;
	while(size>= blockDim.x)
	{
		for(int i=0; i<it;i++)
			if ((index+i<NN)&&(col_numb<size)) d_f[col_numb*NN+index+i] =d_t[col_numb*NN+index+i];
		size-=blockDim.x;
		col_numb+=blockDim.x;
	}
}
