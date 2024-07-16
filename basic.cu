#include "basic.h"
#include <stdio.h>

// память для AUX_CONT вспомогательных слайсов
extern __shared__ unsigned long long int aux_slice[];

void MATCH(Table *tab, Slice *X, Slice *w, Slice *Z)
{
	unsigned int NN, IT,blocks;
	NN=X->NN;
	IT=X->IT;
	blocks=X->blocks;
	match_kernel<<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>(tab->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Z->get_device_pointer(),tab->size,NN,IT);
}

__global__ void match_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT)
{
	match(d_tab,d_x,d_w,d_z,size,NN,IT);
}


__device__ void match(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT)
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
	gel_kernel<<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>(tab->get_device_pointer(),w->get_device_pointer(),X->get_device_pointer(),Y->get_device_pointer(),tab->size,NN,IT);
}
__global__ void gel_kernel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
{
	gel(d_tab,d_w,d_x,d_y,size,NN,IT);
}

__device__ void gel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
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
		less_kernel<<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>(T->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Y->get_device_pointer(),T->size,NN,IT);

}
__global__ void less_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
{
	less(d_tab,d_x,d_w,d_y,size,NN,IT);
}
__device__ void less(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
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
		great_kernel<<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>(T->get_device_pointer(),X->get_device_pointer(),w->get_device_pointer(),Y->get_device_pointer(),T->size,NN,IT);
}
__global__ void great_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
{
	great(d_tab,d_x,d_w,d_y,size,NN,IT);
}
__device__ void great(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT)
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

/*
procedure MIN(T: table; X: slice; var Z: slice);
var Y: slice; i,k: integer;
Begin Z:=X; k:=SIZE(T);
for i:=1 to k do
begin Y:=COL(i,T); Y:=Z and ( not Y);
if SOME(Y) then Z:=Y
end;
End;
*/

__device__ bool some(LongPointer d_y,unsigned int length,unsigned int NN, unsigned int it)
{
	//---------SOME-----------------------
			 __shared__ unsigned int tmp;
			 	unsigned int n = threadIdx.x;
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
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT)
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
	IT=X->IT;
	threads = min(MAX_THREADS,NN);
	min_kernel<<<1,threads,NN*sizeof(unsigned long long int)>>>(T->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->length,T->size,NN,IT);

}

__global__ void max_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_z,
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT)
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
	IT=X->IT;
	threads = min(MAX_THREADS,NN);
	max_kernel<<<1,threads,NN*sizeof(unsigned long long int)>>>(T->get_device_pointer(),X->get_device_pointer(),Z->get_device_pointer(),T->length,T->size,NN,IT);

}
