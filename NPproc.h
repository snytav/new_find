#include "table.h"

#define WMAX 100
#define CMAX 1000

void init_stable(Table T);
__global__ void init_stable(LongPointer d_T,unsigned int NN1,unsigned int size);
// для заданного числа n генерирует масив весов от 1 до 100 и массив стоимости от 1 до 1000

void knapsack_exp();
void knapsack_experiment(unsigned int sz, unsigned lth);

void problem_generate(int n, int *w, int *c);
void complete_search(Table T, Slice ST, Slice SB);

void sortdown(int *first,int *second, unsigned int size);
void sortup(int *first,int *second, unsigned int size);

void branch_cut(int n,int *w, int W, Slice *T,Slice *B);
void branch_cut(int n,int *w, int W, int &k_t ,int &k_b);

void knapsack_optim(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res);
void knapsack_bound(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res);
