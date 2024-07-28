/*
 *  slice.h
 *
 *  библиотека содержит реализацию класса Slice
 */
#include "kernel.h"


class Slice{
// поля
    LongPointer d_v;
public:
    unsigned int length, NN, blocks,IT;
// методы
   LongPointer get_device_pointer(){return d_v;}
//  unsigned int get_block_count(){return blocks;}
   Slice(unsigned int k);
   ~Slice();
   void ASSIGN(Slice *X);
   void AND(Slice *X);
   void OR(Slice *X);
   void XOR(Slice *X);
   void NOT();
   void SET();
   void CLR();
   void MASK(int i);

   unsigned int FND();
   unsigned int STEP();
   unsigned int NUMB();
   bool ZERO();
   bool SOME();

   void shift_up(int i,Slice *s);
   void shift_down(int i,Slice *s);
   void trim(int i,int h,Slice *s);

   void setbit(unsigned int n, int bit);
   int getbit(unsigned int n);

   unsigned long long int ToDigit();
   void FromDigit(unsigned long long dig);

   void print(char *label);
   void fprint(char *label);
};
