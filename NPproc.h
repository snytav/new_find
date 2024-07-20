#include "table.h"

#define WMAX 100
#define CMAX 1000

__global__ void init_stable(LongPointer d_T,unsigned int NN1,unsigned int size);
// для заданного числа n генерирует масив весов от 1 до 100 и массив стоимости от 1 до 1000

void knapsack_exp();
/*
 * knapsack problem

void problem_generate(int n, int *w, int *c);

void branch_cut(int n,int *w, int W, Slice *T,Slice *B);
void branch_cut(int n,int *w, int W, int &k_t ,int &k_b);

*/
