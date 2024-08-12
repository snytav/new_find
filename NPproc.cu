#include "NPproc.h"
#include "basic.h"
#include <ctime>
#include <stdio.h>
#include <sys/time.h>

__global__ void init_stable(LongPointer d_T,unsigned int NN1,unsigned int size)
{	LongPointer d_t;
	unsigned long long int init_x[]={0xAAAAAAAAAAAAAAAA,0xCCCCCCCCCCCCCCCC,
		   0xF0F0F0F0F0F0F0F0,0xFF00FF00FF00FF00,
		   0xFFFF0000FFFF0000,0xFFFFFFFF00000000,
		   0xFFFFFFFFFFFFFFFF,0};
    unsigned long long int i,j=1,k;
    i=blockIdx.x;
    		{   d_t=&(d_T[(size-i-1)*NN1]);
    			if (i<6)
    			{
    				for (k=0;k<NN1;k++) d_t[k]=init_x[i];

    			}
    			else
    			if (i<70)//там сдвиг уже не сработает, но длины таблицы <2:70
    			{
    				j=j<<(i-6);
    				for (k=0;k<NN1;k++)
    				   d_t[k]=((k&j)==j)?init_x[6]:init_x[7];
    			}
    		}
}
// для заданного числа n генерирует масив весов от 1 до WMAX и массив стоимости от 1 до CMAX
void problem_generate(int n, int *w, int *c)
{   srand( time(NULL));
	int i;
	for (i=0;i<n; i++)
	{
		w[i]=rand()%WMAX+1;
		c[i]=rand()%CMAX+1;
		printf("<%i:%d,%d> ",i+1,w[i],c[i]);
	}
}

// vjue
void branch_cut(int n,int *w, int W, Slice *T,Slice *B)
{   int i;
int w_t=W;
int w_b=W;

for (i=0;i<n;i++)
{
	if (w[i]<=w_b)
	{   B->setbit(i+1,1);//нумерация с 1
		w_b-=w[i];

	}
}

for (i=1;i<n+1;i++)
	{
	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				T->setbit(n-i+1,1);//нумерация с 1
			}
}
}

// подряд идущие, без пропусков
void branch_cut(int n, int *w, int W, int &k_t, int &k_b)
{   int i,k_tmp=0;
	int w_t=W;
	int w_b=W;
	k_t=n;
	k_b=0;
	for (i=0;i<n;i++)
	{
		if (w[i]<=w_b)
		{//   printf("%i",1);
			w_b-=w[i];
			k_b++;

		}else
		{
		//	printf("%i",0);
			if (k_tmp==0) k_tmp=k_b;
		}

	}
   // puts("");
	k_b=(k_tmp==0)?k_b:k_tmp+1;
	i=1; k_tmp=0;
//	puts("Mirrored");
	for (i=1;i<n+1;i++)
		{
		if(w[n-i]<=w_t)
				{ //printf("%i",1);
					w_t-=w[n-i];
					k_t--;
				}
		else
				{
					//printf("%i",0);
					if (k_tmp==0) k_tmp=k_t;
				}
	}
		k_t=(k_tmp==0)?k_t:k_tmp;
		printf("   %i:%i %i ",k_b,k_t ,w_t);
   	 puts("");

}

void sort(int *first,int *second, unsigned int size)
{	int tmp;
	for(unsigned int i=0;i<size-1;i++)
		for(unsigned int j=i+1;j<size;j++)
		 if (first[i]>first[j])
		 {
			tmp=first[i];
			first[i]=first[j];
			first[j]=tmp;

			tmp=second[i];
			second[i]=second[j];
			second[j]=tmp;
		 }
}

//количество предметов
#define M 32
//количество бит для максимального веса и цены
#define N_K 16
// МЕДЛЕННЕЕ В 2 РАЗА!
__global__ void weightssumm(LongPointer d_t,LongPointer d_y, LongPointer d_wt,LongPointer d_ww,LongPointer d_ct, LongPointer d_cc, LongPointer d_wmax, LongPointer d_z3, LongPointer d_z,unsigned int size,unsigned int NNT,unsigned int NNR, unsigned int IT,LongPointer aux_slice)
{unsigned int i;
for(i=1;i<=size;i++)
	{

  // T.GetCol(&Y,i);
   _getCol(d_t,d_y,i,NNT,IT);
//    ADDROW(&WT,&WW,i,&Y);
    addrow(d_wt,d_ww,i,d_y,d_z,size,NNT,NNR,IT,aux_slice);
//     ADDROW(&CT,&CC,i,&Y);
    addrow(d_ct,d_cc,i,d_y,d_z,size,NNT,NNR,IT,aux_slice);
	}
//LESS(&WT,&Z3,&w_max,&Z);
less(d_wt,d_z3,d_wmax,d_z,size,NNT,IT,aux_slice);
}

void knapsack_exp()
{
	unsigned int sz=M,lth;
//	lth=powl(2,sz);
	lth=64*2048;//powl(2,sz);
	int
	w[M],c[M],W,k_b,k_t,i,j,nn;//,W_t=0,W_b=0;

	Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

	Table WW(sz,N_K),CC(sz,N_K),T(lth,sz);
	unsigned long long int nj,ns;//tmp,dig,
	if (InitAuxSlices(T.NN)>0){puts(" InitAuxSlices error");}
	double tt;
    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);
	Table WT(lth,N_K), CT(lth,N_K);

	Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
	FILE *f = fopen("res/time.txt", "w");
	fprintf(f,"N\t W \t unsorted \t sorted by weight \t count only \t I \t II \t III \n");

 	Table WLast(lth,N_K), CLast(lth,N_K);
 	unsigned int klst;
// ST Верхняя строка проверяемого
// SB Нижняя строка проверяемого

//	Для проверки границ
	// ST1 Верхняя строка проверяемого
	// SB1 Нижняя строка проверяемого
//
 for (nn=0;nn<10;nn++)
 {
	printf("\n %d \n",nn);
	problem_generate(M, w, c);
	W=rand()%(WMAX*M/2)+1;
	branch_cut(M, w,W,k_t,k_b);
	branch_cut(M, w,W, &ST,&SB);
//	ST.print("st");
//	SB.print("sb");

	w_max.FromDigit(W);
//		w_max->print("w_max",0);
	printf("W=%i \n",W);

	printf("Размеры: size=%u length =%u, Blocks=%u IT=%u\n",T.size,T.length,T.blocks,T.IT);
	// инициализация таблицы перебора
	// появление в ней sb - условие окончания перебора
	init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);

//	T->writeToFile("log/init");
	cudaError_t err = cudaGetLastError();
	 printf("errors after init_stable %d\n",err);

    //вставить проверку, что ST<>SB, иначе тривиально.
    SN.ASSIGN(&ST);
    SN.NOT();
    SN.AND(&SB);
    if (SN.ZERO())
    {
    	ST.print("SOLVE");
    	return;
    }

    T.GetRow(&SN,lth);
//    SN.print("last");
    i=SN.FND(); SN.CLR();
    if (i>1) SN.setbit(i-1,1);
//    SN.print("будем складывать");
    Z1.CLR();
    Z2.SET();
    Z3.SET();
    ADDC1(&T,&ST,&Z3);
    c_lmax.CLR();


//    Z2.setbit(1,0);// в этом варианте мах храним не в таблице

    ST.print("unsorted ST");
 //   SN.print("log/SN");
    SB.print("insorted SB");
    ns=SN.ToDigit();
    nj=ST.ToDigit();
    fprintf(f,"%d\t W=%d \t %llu \t",nn, W,SB.ToDigit()-nj);
 /*/ сортировка по стоимости
    sort(c,w,sz);
    branch_cut(M, w,W, &ST,&SB);
    ST.print("log/ST");
    SB.print("log/SB");
    nj=ST.ToDigit();
    printf("Sorted by cost W=%d SN=%llu, ST=%llu, SB=%llu: %llu \n",W,ns,nj,SB.ToDigit(),SB.ToDigit()-nj);
 */
    // сортировка по весу
    sort(w,c,sz);
    branch_cut(M, w,W, &ST,&SB);
    ST.print("sorted ST");
    SB.print("sorted SB");
    nj=ST.ToDigit();
    fprintf(f," %llu \t",SB.ToDigit()-nj);

    // print results branch_cut
    	for (i=0; i<sz;i++){
    		w_max.FromDigit(w[i]);
    		WW.SetRow(&w_max,i+1);
    		w_max.FromDigit(c[i]);
    		CC.SetRow(&w_max,i+1);
    	}
    	w_max.FromDigit(W);
    //вставить проверку, что ST<>SB, иначе тривиально.


//    printf("Перебор без вычислений \n");
    gettimeofday(&tv1,NULL);
    j=0;
    while(Z1.ZERO())
    {

        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
       j++;
       nj+=ns;
    	   ADDC1(&T,&SN,&Z3); // к первой строке не добавляется, там максимум
    }

     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 fprintf(f,"%f.2 \t", tt);
	 /****************************************************************************/

	 gettimeofday(&tv1,NULL);
	init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
    Z1.CLR();
    Z2.SET();
    Z3.SET();
    ADDC1(&T,&ST,&Z3);
    c_lmax.CLR();
    j=0;

//Z1->print("Z1",0);
    while(Z1.ZERO())//&&(j<10))
    {
        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
    	CLEAR(&WT); // тут должен быть h=N_K
    	CLEAR(&CT); // тут должен быть h=N_K
       for(i=1;i<=sz;i++)
    	{
           T.GetCol(&Y,i);
          ADDROW(&WT,&WW,i,&Y);
          ADDROW(&CT,&CC,i,&Y);
     	  err = cudaGetLastError();
          if (err>0)
          {
              printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
              return;
   		  }
        }

        LESS(&WT,&Z3,&w_max,&Z);
        GREAT(&CT,&Z,&c_lmax,&Z2);

    	err = cudaGetLastError();
    	if (err>0){
    	  printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
    	  return;
    			}
       MAX(&CT,&Z2,&X);
       i=X.FND();
//       printf("max=%i\n",i);
       if (i>0)// смена максимума
       {
          T.GetRow(&t_lmax,i);
          CT.GetRow(&c_lmax,i);
          WT.GetRow(&w_lmax,i);
 //         t_lmax.print("local");
 //         printf("j=%i i=%i C=%llu W=%llu\n",j,i,c_lmax.ToDigit(),w_lmax.ToDigit());
       }
       j++;
       nj+=ns;

   	   ADDC1(&T,&SN,&Z3);

       err = cudaGetLastError();
       if (err>0){
            printf("after all %d , %s \n",err,cudaGetErrorString(err));
            return;
       }
    }

     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 t_lmax.print("MAX");

//	 printf("Max=%llu, NextSB=%llu \n",ST1.ToDigit(),SB1.ToDigit());
	 printf("j=%i W=%d weight sum= %llu, cost summ = %llu \n",j,W, w_lmax.ToDigit(),c_lmax.ToDigit());

	 fprintf(f,"%f.2 \t", tt);
/****************************************************************/

	 	gettimeofday(&tv1,NULL);
		init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();
	    ADDC1(&T,&ST,&Z3);
	    c_lmax.CLR();
	    t_lmax.CLR();
	//Z1->print("Z1",0);
	    CLEAR(&WLast); // тут должен быть h=N_K
	    CLEAR(&CLast);
	    klst=SN.FND()+1;
	    for(i=klst;i<=sz;i++)
	    {
	       T.GetCol(&Y,i);
	       ADDROW(&WLast,&WW,i,&Y);
	       ADDROW(&CLast,&CC,i,&Y);
	       err = cudaGetLastError();
	       if (err>0)
	    	{
	    	   printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
	    	   return;
	    	}
	     }
	    j=0;

	    while (Z1.ZERO())//&&(j<10))
	    {
	        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
	        TCOPY(&WLast,&WT); // тут должен быть h=N_K
	        TCOPY(&CLast,&CT); // тут должен быть h=N_K

	       for(unsigned int i1=1;i1<klst;i1++)
	    	{
	           T.GetCol(&Y,i1);
	          ADDROW(&WT,&WW,i1,&Y);
	          ADDROW(&CT,&CC,i1,&Y);
	     	  err = cudaGetLastError();
	          if (err>0)
	          {
	              printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
	              return;
	   		  }
	        }

	        LESS(&WT,&Z3,&w_max,&Z);
	        GREAT(&CT,&Z,&c_lmax,&Z2);
	    	err = cudaGetLastError();
	    	if (err>0){
	    	  printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
	    	  return;
	    			}
	       MAX(&CT,&Z2,&X);
	       i=X.FND();
//	       printf("max=%i\n",i);
	       if (i>0)// смена максимума
	       {
	          T.GetRow(&t_lmax,i);
	          CT.GetRow(&c_lmax,i);
	          WT.GetRow(&w_lmax,i);
//	          t_lmax.print("local");
	 //         printf("j=%i i=%i C=%llu W=%llu\n",j,i,c_lmax.ToDigit(),w_lmax.ToDigit());
	       }
	       j++;
	       nj+=ns;

	   	   ADDC1(&T,&SN,&Z3);

	       err = cudaGetLastError();
	       if (err>0){
	            printf("after all %d , %s \n",err,cudaGetErrorString(err));
	            return;
	       }
	    }

	     gettimeofday(&tv2,NULL);
		 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);

		 t_lmax.print("MAX");

//		 printf("Max=%llu, NextSB=%llu \n",ST1.ToDigit(),SB1.ToDigit());
		 printf("j=%i W=%i weight sum= %llu, cost summ = %llu \n",j,W, w_lmax.ToDigit(),c_lmax.ToDigit());

		 fprintf(f,"%f.2 \t", tt);
/********************************************************************/

		 	gettimeofday(&tv1,NULL);
			init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
		    Z1.CLR();
		    Z2.SET();
		    Z3.SET();
		    ADDC1(&T,&ST,&Z3);
		    c_lmax.CLR();
		    t_lmax.CLR();
		    CLEAR(&WLast); // тут должен быть h=N_K
		    CLEAR(&CLast);

		    klst=SN.FND()+1;
		    for(i=klst;i<=sz;i++)
		    {
		       T.GetCol(&Y,i);
		       ADDROW(&WLast,&WW,i,&Y);
		       ADDROW(&CLast,&CC,i,&Y);
		       err = cudaGetLastError();
		       if (err>0)
		    	{
		    	   printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
		    	   return;
		    	}
		     }
		    j=0;
//		    printf("klst=%d\n",klst);
		    unsigned long long int cloc,wloc;
		    while (Z1.ZERO())//&&(j<10))
		    {
		        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
		        TCOPY(&WLast,&WT); // тут должен быть h=N_K
		        TCOPY(&CLast,&CT); // тут должен быть h=N_K

		        T.GetRow(&ST1,1);
		        T.GetRow(&SB1,lth);
//		        ST1.print("ST");
//		        SB1.print("SB");
		        TMP.ASSIGN(&ST1);
		        TMP.XOR(&SB1);
		        unsigned int i1=TMP.FND();
		        cloc=0;wloc=0;
		        TMP.ASSIGN(&ST1);
  //  	        TMP.print("2 part");

		        i=TMP.STEP();
		        while ((i>0)&&(i<i1))
		        {

		        	cloc+=c[i-1];
		        	wloc+=w[i-1];
  //  	        	printf("<%i,%i> %d->%d %d->%d",j,i,c[i-1],cloc,w[i-1],wloc);
		        	i=TMP.STEP();
		        }
		        u.FromDigit(cloc);
		        ADDC1(&CT, &u,&Z3);
		        u.FromDigit(wloc);
		        ADDC1(&WT, &u,&Z3);

		        i=i1;
//		        printf("j=%i from %i to %i \n",j,i,klst-1);
		       for(unsigned int i1=i;i1<klst;i1++)
		    	{
		           T.GetCol(&Y,i1);
		          ADDROW(&WT,&WW,i1,&Y);
		          ADDROW(&CT,&CC,i1,&Y);
		     	  err = cudaGetLastError();
		          if (err>0)
		          {
		              printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
		              return;
		   		  }
		        }

		        LESS(&WT,&Z3,&w_max,&Z);
		        GREAT(&CT,&Z,&c_lmax,&Z2);
		    	err = cudaGetLastError();
		    	if (err>0){
		    	  printf("after for j=%i i=%i %d , %s \n",j,i,err,cudaGetErrorString(err));
		    	  return;
		    			}
		       MAX(&CT,&Z2,&X);
		       i=X.FND();
	//	       printf("max=%i\n",i);
		       if (i>0)// смена максимума
		       {
		          T.GetRow(&t_lmax,i);
		          CT.GetRow(&c_lmax,i);
		          WT.GetRow(&w_lmax,i);
	//	          t_lmax.print("local");
	//	          printf("j=%i i=%i C=%llu W=%llu\n",j,i,c_lmax.ToDigit(),w_lmax.ToDigit());
		       }
		       j++;
		       nj+=ns;

		   	   ADDC1(&T,&SN,&Z3);

		       err = cudaGetLastError();
		       if (err>0){
		            printf("after all %d , %s \n",err,cudaGetErrorString(err));
		            return;
		       }
		    }

		     gettimeofday(&tv2,NULL);
			 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);

			 t_lmax.print("MAX");

			 printf("j=%i W=%i weight sum= %llu, cost summ = %llu \n",j,W, w_lmax.ToDigit(),c_lmax.ToDigit());

			 fprintf(f,"%f.2  \n", tt);
/*************************************************************/
 }
			 fclose(f);
}
