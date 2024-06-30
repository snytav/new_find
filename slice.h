/*
 *  slice.h
 *
 *  библиотека содержит реализацию функций на устройстве
 */
#include "kernel.h"

#define MAX_BLOCK 2024

class Slice{
// поля
    LongPointer d_v;
public:
    unsigned int length, NN, blocks,IT;
// методы
   LongPointer get_device_pointer(){return d_v;}
   unsigned int get_block_count(){return blocks;}
   Slice(unsigned int k);

   void ASSIGN(Slice *X);
   void AND(Slice *X);
   void OR(Slice *X);
   void XOR(Slice *X);
   void NOT();
   void SET();
   void CLR();

   unsigned int FND();
   unsigned int NUMB();
   bool ZERO();
   bool SOME();
};
