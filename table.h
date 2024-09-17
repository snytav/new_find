#ifndef TABLE_H_
#define TABLE_H_

#include "kernel.h"
#include "slice.h"

class Table{
// поля
    LongPointer d_v;
public:
    unsigned int length, NN,size, blocks,IT;
// методы
   LongPointer get_device_pointer(){return d_v;}
//  unsigned int get_block_count(){return blocks;}
   Table(unsigned int l,unsigned int s);
   ~Table();
   void GetCol(Slice* X,unsigned int i);
   void SetCol(Slice* X,unsigned int i);
   void GetRow(Slice* X,unsigned int i);
   void GetRow1(Slice* X,unsigned int i);
   void SetRow(Slice* X,unsigned int i);
// печать в двоичном виде
   void fprint(const char *label);
};

__device__ void _getCol(LongPointer d_table, LongPointer d_slice,unsigned int i,
		unsigned int NN,unsigned int IT);

__device__ void _setCol(LongPointer d_table, LongPointer d_slice,unsigned int i,
		unsigned int NN,unsigned int IT);

#endif /* TABLE_H_ */
