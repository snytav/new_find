#ifndef BASIC_H_
#define BASIC_H_

#include "table.h"

#define AUX_COUNT 3
int InitAuxSlices(unsigned  int NN);

// для использования в I группе, конфигурация ядра <<<1,threads>>>
// нужна синхронизация по всем потокам
__device__ bool some(LongPointer d_y,unsigned int length,unsigned int NN, unsigned int it);
/*
*---------------------------- I группа алгоритмов--------------------------------
* конфигурация ядра <<<1,threads,NN*sizeof(unsigned long long int)>>>
* нужна синхронизация по всем потокам
* использует вспомогательный слайс
*/
/*
void MIN(Table *T, Slice *X, Slice*Z);
__global__ void min_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_z,
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT);
*/
void MAX(Table *T, Slice *X, Slice*Z);
__global__ void max_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_z,
		unsigned int length, unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
/*
*---------------------------- II группа алгоритмов--------------------------------
* конфигурация ядра <<<blocks,1,NN*AUX_COUNT*sizeof(unsigned long long int)>>>
* допустима <<<1,threads,NN*AUX_COUNT*sizeof(unsigned long long int)>>>
*/
void MATCH(Table *tab, Slice *X, Slice *w, Slice *Z);
__global__ void match_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
__device__ void match(LongPointer d_tab,LongPointer d_x,LongPointer d_w,LongPointer d_z,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void GEL(Table *tab, Slice *w, Slice *X, Slice *Y);
__global__ void gel_kernel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
__device__ void gel(LongPointer d_tab, LongPointer d_w,LongPointer d_x,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void LESS(Table *T, Slice *X, Slice *v,Slice *Y);
__global__ void less_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_v,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
__device__ void less(LongPointer d_tab,LongPointer d_x,LongPointer d_v,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void GREAT(Table *T, Slice *X, Slice *v,Slice *Y);
__global__ void great_kernel(LongPointer d_tab,LongPointer d_x,LongPointer d_v,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
__device__ void great(LongPointer d_tab,LongPointer d_x,LongPointer d_v,LongPointer d_y,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void SETMIN(Table *T, Table *F, Slice *X, Slice *Z);
__global__ void setmin_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
__device__ void setmin(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void SETMAX(Table *T, Table *F, Slice *X, Slice *Z);
__global__ void setmax_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
__device__ void setmax(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void HIT(Table *T, Table *F, Slice *X, Slice *Z);
__global__ void hit_kernel(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
__device__ void hit(LongPointer d_t, LongPointer d_f,LongPointer d_x,LongPointer d_z ,
		unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

//---------------------------- IV группа алгоритмов--------------------------------
void CLEAR(Table *T);
__global__ void clear_kernel(LongPointer d_v,unsigned int size,unsigned int NN, unsigned int IT);
__device__ void clear(LongPointer d_v,unsigned int size,unsigned int NN, unsigned int IT);
/*
void TMERGE(Table *T,  Slice *X, Table *F, int k=1);
__global__ void tmerge_kernel(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f,int size);//<<<NN,k>>> k=1,...,M
__device__ void tmerge(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f,int size);
__device__ void tmerge_par(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f); // blockDim.y=M

void WMERGE(Slice *v,  Slice *X, Table *F, int k=1);
//size не вставлен может ли выступать в этой роли k нужно смотреть
__global__ void wmerge_kernel(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f, int k);//<<<NN,k>>> k=1,...,M
__device__ void wmerge(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f, int k);

void WCOPY(Slice *v,  Slice *X, Table *F,int k=1);
__global__ void wcopy_kernel(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f,int k,int size);//<<<NN,k>>> k=1,...,M
__device__ void wcopy(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f,int k,int size);

void TCOPY(Table *T, Table *F,int k=1);
__global__ void tcopy_kernel(LongPointer *d_t, LongPointer *d_f,int size);//<<<NN,k>>> k=1,...,r r=SIZE(T)
__device__ void tcopy(LongPointer *d_t, LongPointer *d_f,int size);

void TCOPY1(Table *T,int j, int h, Table *F,int k=1);// Копирует вертикальную полосу из T в F
__global__ void tcopy1_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f);//<<<NN,k>>> k=1,...,h
__device__ void tcopy1(LongPointer *d_t, int j, int h, LongPointer *d_f);

void TCOPY2(Table *T,int j, int h, Table *F,int k=1);// Копирует T как вертикальную полосу в F
__global__ void tcopy2_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f);//<<<NN,k>>> k=1,...,h
__device__ void tcopy2(LongPointer *d_t, int j, int h, LongPointer *d_f);

//нужно проверять
void TCOPY3(Table *T,int j, int h, Table *F,int k=1);// Копирует горизонтальную полосу из T в F
__global__ void tcopy3_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f);//!<<<NN,k>>> k=1,...,f.size
__device__ void tcopy3(LongPointer *d_t, int j, int h, LongPointer *d_f);

void TCOPY4(Table *T,int j, int h, Table *F,int k=1);// Копирует T как горизонтальную полосу в F
__global__ void tcopy4_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f);//!<<<NN,k>>> k=1,...,h
__device__ void tcopy4(LongPointer *d_t, int j, int h, LongPointer *d_f);
*/
//---------------------------- III группа алгоритмов--------------------------------

void ADDV(Table *T, Table *R, Slice *X, Table *S);
__global__ void addv_kernel(LongPointer d_t,LongPointer d_r,LongPointer d_x,LongPointer d_s,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);//<<<NN,1>>>
//d_b перенос на предыдущий разряд
__device__ void addv(LongPointer d_t,LongPointer d_r,LongPointer d_x,LongPointer d_s,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void ADDC(Table *T, Slice *w, Slice *X, Table *S);
__global__ void addc_kernel(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_s,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
__device__ void addc(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_s,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);

void ADDC1(Table *T, Slice *w, Slice *X);
__global__ void addc1_kernel(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
__device__ void addc1(LongPointer d_t,LongPointer d_w,LongPointer d_x,LongPointer d_b,
				unsigned int size,unsigned int NN, unsigned int IT,LongPointer aux_slice);
/*
void SUBTV(Table *T, Table *R, Slice *X,Table *S);
__global__ void subtv_kernel(LongPointer *d_t, LongPointer *d_r,int k, unsigned long long int *d_x, LongPointer *d_s,unsigned long long int *d_m);
__device__ void subtv(LongPointer *d_t, LongPointer *d_r,int k, unsigned long long int *d_x, LongPointer *d_s,unsigned long long int *d_m);

void SUBTC(Table *T, Slice *X, Slice *w, Table *S);
__global__ void subtc_kernel(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m);
__device__ void subtc(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m);

void SUBTC1(Table *T, Slice *X, Slice *w, Table *S);
__global__ void subtc1_kernel(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m);
__device__ void subtc1(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m);
//---------------------------------------------------------------------------
void WTRANS(Slice *w, int h, Table *R);//<<<NN,64>>>
__global__ void wtrans_kernel(unsigned long long int *d_w, int h,int length, LongPointer *d_r);
__device__ void wtrans(unsigned long long int *d_w, int h, int length, LongPointer *d_r);

// <<<NN,VER>>>
// слайс X объединяется с теми столбцами из T_k, если w(k)=1
__global__ void x_w_table_or_kernel(LongPointer *d_tab,  unsigned long long int *d_x, unsigned long long int *d_w,int size);
__global__ void x_w_table_and_kernel(LongPointer *d_tab,  unsigned long long int *d_x, unsigned long long int *d_w, int size);
 
*/
#endif /* BASIC_H_ */
