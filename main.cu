#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"
#include <stdio.h>
#include <time.h>
//#include "slice.h"
//#include "table.h"
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"

#include <cmath>
//#include "basic.h"
#include "NPproc.h"

#define NN_MAX 4096
// 1536
// AUX_COUNT 4 при большем числе попытка вызвать базовую процедуру выдает ошибку 11
int main()
{
/*

	unsigned int N1,sz=500,lth=64*2048;//powl(2,sz);


    Slice X(lth),Y(sz),Z(lth);
    Table T(lth,sz);
    N1=X.NN;

    printf("Размеры: size=%u length =%u, NN=%u IT=%u\n",sz,lth,N1,X.IT);
 //   Y.SET();
 //   Y.NOT();
    X.SET();

    init_stable<<<sz,1>>>(T.get_device_pointer(),N1,sz);
    int i=521;
    T.GetRow(&Y,i);
//    T.GetCol(&X,1);
    Y.print("row_521");

//    printf(" ZERO %d\n", X.ZERO());
 //   printf("%d %d ZERO %d\n",i,X.STEP(), X.SOME()); // так нельзя, Х не успевает обновляться
 //   printf("%d STEP=%d\n",i,X.STEP());
 //   Y.trim(15,L,&X);
 //   Y.fprint("tream");
    MATCH(&T,&X,&Y,&Z);
    cudaError_t err = cudaGetLastError();
    printf("errors after MATCH %d\n",err);
    Z.fprint("MATCH_res");
    printf("MATCH i=%i res=%i\n",i,Z.FND());
    T.GetRow(&Y,lth);
    Y.print("row");
    ADDC1(&T,&Y,&X);//последнюю строку добавляем ко всем

    T.GetRow(&Y,lth);
    Y.print("row_last");
 /*   for(int i=1;i<=sz;i++)
    {
    	T.GetCol(&X,i);
    	X.print("col");
    }
    */
	unsigned int N1,sz,lth;
	sz=M;//13;//ограничение по второй таблице
/*	MAX_BLOCK=32;//1024
	N1=4096;//pow(2,sz-6); //2048;
*/
	if (M<12){
	MAX_BLOCK=pow(2,sz-6);N1=MAX_BLOCK; }
	else
	{
		MAX_BLOCK=32;
		if(M<24)
		N1=pow(2,sz-6);
		else N1=16384;
	}
	lth=64*N1;
    knapsack_experiment(M,lth,sz);
   // knapsack_experiment1();
//    MAX_BLOCK=MAX_BLOCK/32;
//	}
	cudaError_t err = cudaGetLastError();
	if (err>0) printf("errors after knapsack_exp %d %s\n",err,cudaGetErrorString(err));
    return 0;
}
 
