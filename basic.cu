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
	printf("%i %i _match: Y %p\n",blockIdx.x,threadIdx.x, d_y);
	_assign(d_z, d_x,NN,IT);
	for(i=1;i<=size;i++)
	{
		_getCol(d_tab,d_y,i,NN,IT);
		if (_getbit(d_w, i)==1)
		{ printf("%i ",i);
			_and(d_z,d_y,NN,IT);
		}
		else
		{
			_not(d_y,NN,IT);
			_and(d_z,d_y,NN,IT);
		}
	}
}
