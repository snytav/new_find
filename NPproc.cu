#include "NPproc.h"
#include "basic.h"
#include <ctime>
#include <stdio.h>
#include <string>
#include <sys/time.h>

void init_stable(Table T)
{
    unsigned int sz=T.size;
    unsigned int lth=T.length;
    Slice X(lth);
    for(int i=1;i<=sz;i++)
    {
//    	printf("%d ",i);
        X.MASK1(sz-i);
        T.SetCol(&X,i);
    }
    puts("INIT");
    cudaError_t err = cudaGetLastError();
       if (err>0) printf("errors after init_stable %d:%s\n",err, cudaGetErrorString(err));
}

__global__ void init_stable(LongPointer d_T,unsigned int NN1,unsigned int size)
{	LongPointer d_t;
	unsigned long long int init_x[]={0xAAAAAAAAAAAAAAAA,0xCCCCCCCCCCCCCCCC,
		   0xF0F0F0F0F0F0F0F0,0xFF00FF00FF00FF00,
		   0xFFFF0000FFFF0000,0xFFFFFFFF00000000,
		   0xFFFFFFFFFFFFFFFF,0};
    unsigned long long int i, k,j=1;
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
//		printf("<%i:%d,%d> ",i+1,w[i],c[i]);
	}
	printf("w={%d",w[0]);
	for (i=1;i<n; i++)
		{
			printf(", %d",w[i]);
		}
	printf("}\n");
	printf("c={%d",c[0]);
		for (i=1;i<n; i++)
			{
				printf(", %d",c[i]);
			}
		printf("}");
}

// vjue
void branch_cut(int n,int *w, int W, Slice *T,Slice *B)
{   int i;
int w_t=W;
int w_b=W;
char str[n+1];
for (i=1;i<n+1;i++)
	{
	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				T->setbit(n-i+1,1);//нумерация с 1
				str[n-i]='1';
			}
	else{ str[n-i]='0'; T->setbit(n-i+1,0);}
}
str[n]=0;
printf("Top \t %s \n",str);
for (i=0;i<n;i++)
{
	if (w[i]<=w_b)
	{   B->setbit(i+1,1);//нумерация с 1
		w_b-=w[i];
		str[i]='1';
	}
	else {str[i]='0';B->setbit(i+1,0);}
}
str[n]=0;
printf("Bottom \t %s \n",str);
//T->print("bc_T");B->print("bc_B");
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

void sortdown(int *first,int *second, unsigned int size)
{	int tmp;
	for(unsigned int i=0;i<size-1;i++)
		for(unsigned int j=i+1;j<size;j++)
		 if (first[i]<first[j])
		 {
			tmp=first[i];
			first[i]=first[j];
			first[j]=tmp;

			tmp=second[i];
			second[i]=second[j];
			second[j]=tmp;
		 }
}

void sortup(int *first,int *second, unsigned int size)
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
#define M 20
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
    addrow(d_wt,d_ww,i,d_y,size,NNT,NNR,IT,aux_slice);
//     ADDROW(&CT,&CC,i,&Y);
    addrow(d_ct,d_cc,i,d_y,size,NNT,NNR,IT,aux_slice);
	}
//LESS(&WT,&Z3,&w_max,&Z);
less(d_wt,d_z3,d_wmax,d_z,size,NNT,IT,aux_slice);
}

void complete_search(Table T, Slice ST,Slice SB)
{   cudaError_t err;
	unsigned int sz=T.size;
	unsigned int lth=T.length;
	//unsigned long long int iter_max;
	unsigned long long int j;
	Slice Z1(lth),Z2(lth), Z3(lth),Y(lth);
	Slice ST1(sz),SB1(sz),TMP(sz),SN(sz);
	unsigned int i,m,l;

//	ST.print("ST before init");
//	SB.print("SB before init");
	init_stable(T);

//	int a[sz];
//	for(int i=0;i<sz;i++) a[i]=0;
 //   init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
    err = cudaGetLastError();
         if (err>0) printf("errors after init_stable %d:%s\n",err, cudaGetErrorString(err));
	printf("lth=%i ",lth);
    T.GetRow(&SB1,lth);
// puts("Row last");
    err = cudaGetLastError();
             if (err>0) printf("errors after GetRow lth %d:%s\n",err, cudaGetErrorString(err));
    l=SB1.FND();
// puts("SB1.FND");
    printf("l=%i \n",l);
    SB1.print("SB1");
    SN.CLR();
    if (l>1) SN.setbit(l-1,1);
    else
    {
    	printf("перебор за одну итерацию l=%d \n",l);
    }
	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();
	    ST1.ASSIGN(&ST);
	    SB1.NOT();
	    ST1.AND(&SB1);
ST.print("ST before ADDC1"); ST1.print("ST corrected ");
	    ADDC1(&T,&ST1,&Z3);
/*
 *  инициализация переменных задачи
 */
	    for(i=l;i<=sz;i++)
	    {
/*
 *  вычисление данных по неизменным столбцам
 */
	     }
//	    iter_max=10;
	    j=0;
	    while (Z1.ZERO())
	    {
//
	        MATCH(&T,&Z3,&SB,&Z1);
/*
 *  копирование данных по неизменным столбцам
 */
	        T.GetRow(&ST1,1);
	        T.GetRow(&SB1,lth);
//	        printf("j=%llu ",j); ST1.print("ST1"); SB1.print("SB1");

	        TMP.ASSIGN(&ST1);
	        TMP.XOR(&SB1);
	        m=TMP.FND();
	        TMP.ASSIGN(&ST1);
/*	        if(m!=l)
	        {
	        	char name[100];
	        	sprintf(name,"m= %i l=%i",m,l);
	        	ST1.print(name);
	        	SB1.print(name);
	        }
*/
	        i=TMP.STEP();
//	        a[m]++;
	        while ((i>0)&&(i<m))
	        {
/*
 *   расчет по элементам, одинаковым во всех строках, на CPU
 */
	        	i=TMP.STEP();
	        }

/*
 *   обновить данные на GPU
 */

//	       for(i=m;i<l;i++)
//	    	{
//	           T.GetCol(&Y,i);
/*
 *   расчет по оставшимся столбцам
 */
//	        }

/*
 *  обновление локального минимума/максимума
 */
	   	   ADDC1(&T,&SN,&Z3);
	   	   j++;
	    }
//	    for(int i=0;i<sz;i++) printf("a[%i]=%i ",i,a[i]);
}

void knapsack_optim_new(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{  unsigned int sz=T.size,lth=T.length;
//unsigned long long int w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
unsigned long long int i,j;
unsigned long long int cloc,wloc;//,wlast,wlmin;
Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

Table WW(sz,N_K),CC(sz,N_K);
unsigned long long int nj,ns;//tmp,dig,
Table WT(lth,N_K), CT(lth,N_K);
	Table WLast(lth,N_K), CLast(lth,N_K);
	unsigned int klst,l;
// bool change;

Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
cudaError_t err;
    branch_cut(sz, w,W, &ST,&SB);
//    ST.print("sorted ST");
//    SB.print("sorted SB");


    // print results branch_cut
    	for (i=0; i<sz;i++){
    		w_max.FromDigit(w[i]);
    		WW.SetRow(&w_max,i+1);
    		w_max.FromDigit(c[i]);
    		CC.SetRow(&w_max,i+1);
//	    		printf("<%i:%i,%i> ",i+1,w[i],c[i]);
    	}
//	    	printf("\n");
    	w_max.FromDigit(W);
    	init_stable(T);
//		init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
err = cudaGetLastError();
	if (err>0) printf("errors after init_stable %d:%s\n",err, cudaGetErrorString(err));
		T.GetRow(&SB1,lth);
err = cudaGetLastError();
	if (err>0) printf("errors after getrow lst %d:%s\n",err, cudaGetErrorString(err));
		l=SB1.FND();
		    SN.CLR();
		    if (l>1) SN.setbit(l-1,1);
		    else
		    {
		    	printf("перебор за одну итерацию l=%d \n",l);
		    }

	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();

	    ST1.ASSIGN(&ST);
	    SB1.NOT();
	    ST1.AND(&SB1);
	    nj=ST1.ToDigit();
//	    printf("count  %llu : j<%llu \n",SB.ToDigit()-nj,(SB.ToDigit()-nj)/T.length);

	    ADDC1(&T,&ST1,&Z3);
	    c_lmax.CLR();
	    t_lmax.CLR();
	    CLEAR(&WLast); // тут должен быть h=N_K
	    CLEAR(&CLast);

	    klst=l;//SN.FND()+1;
	    for(i=klst;i<=sz;i++)
	    {
	       T.GetCol(&Y,i);
	       ADDROW(&WLast,&WW,i,&Y);
	       ADDROW(&CLast,&CC,i,&Y);
	       err = cudaGetLastError();
	       if (err>0)
	    	{
	    	   printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
	    	   return;
	    	}
	     }
	    j=0;
//		    printf("klst=%d\n",klst);

	    while (Z1.ZERO())//&&(j<10))
	    {
	        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
//puts("MATCH");
//	        TCOPY(&WLast,&WT); // тут должен быть h=N_K
//	        TCOPY(&CLast,&CT); // тут должен быть h=N_K

	        T.GetRow(&ST1,1);
	        T.GetRow(&SB1,lth);
//		        ST1.print("ST");
//		        SB1.print("SB");
//	if(j%1000==0)//||(j>15000))
//		printf("%llu:%llu ",j, ST1.ToDigit());
	        TMP.ASSIGN(&ST1);
	        TMP.XOR(&SB1);
	        unsigned int i1=TMP.FND();
	        cloc=0;wloc=0;
	        TMP.ASSIGN(&ST1);
//  	        TMP.print("2 part");

	        i=TMP.STEP();
	        while ((i>0)&&(i<i1))
	        {
//printf(" %i ",i);
	        	cloc+=c[i-1];
	        	wloc+=w[i-1];
//  	        	printf("<%i,%i> %d->%d %d->%d",j,i,c[i-1],cloc,w[i-1],wloc);
	        	i=TMP.STEP();
	        }
	        u.FromDigit(cloc);
	        ADDC(&CLast, &u,&Z3,&CT);
	        u.FromDigit(wloc);
	        ADDC(&WLast, &u,&Z3,&WT);

/*	        i=i1;
//		        printf("j=%i from %i to %i \n",j,i,klst-1);
	       for(unsigned int i1=i;i1<klst;i1++)
	    	{
	           T.GetCol(&Y,i1);
	          ADDROW(&WT,&WW,i1,&Y);
	          ADDROW(&CT,&CC,i1,&Y);
	     	  err = cudaGetLastError();
	          if (err>0)
	          {
	              printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
	              return;
	   		  }
	        }
*/
	        LESS(&WT,&Z3,&w_max,&Z);
	        GREAT(&CT,&Z,&c_lmax,&Z2);
	    	err = cudaGetLastError();
	    	if (err>0){
	    	  printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
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
//	       printf(" %llu: %llu ",j, SB1.ToDigit());
//	       if(j%1000==0) printf(" %llu ",j);
	    }
	    res.ASSIGN(&t_lmax);
}
/////////////////////////////////////////////////////////////////////////////////

void knapsack_optim(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{  unsigned int sz=T.size,lth=T.length;
//unsigned long long int w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
unsigned long long int i,j;
unsigned long long int cloc,wloc;//,wlast,wlmin;
Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

Table WW(sz,N_K),CC(sz,N_K);
unsigned long long int nj,ns;//tmp,dig,
Table WT(lth,N_K), CT(lth,N_K);
	Table WLast(lth,N_K), CLast(lth,N_K);
	unsigned int klst,l;
// bool change;

Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
cudaError_t err;
    branch_cut(sz, w,W, &ST,&SB);
 //   ST.print("sorted ST");
 //   SB.print("sorted SB");
    nj=ST.ToDigit();
    printf("count  %llu : j<%llu \n",SB.ToDigit()-nj,(SB.ToDigit()-nj)/T.length);

    // print results branch_cut
    	for (i=0; i<sz;i++){
    		w_max.FromDigit(w[i]);
    		WW.SetRow(&w_max,i+1);
    		w_max.FromDigit(c[i]);
    		CC.SetRow(&w_max,i+1);
//	    		printf("<%i:%i,%i> ",i+1,w[i],c[i]);
    	}
//	    	printf("\n");
    	w_max.FromDigit(W);
    	init_stable(T);
//		init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
	err = cudaGetLastError();
	if (err>0) printf("errors after init_stable %d:%s\n",err, cudaGetErrorString(err));
		T.GetRow(&SB1,lth); l=SB1.FND();
		    SN.CLR();
		    if (l>1) SN.setbit(l-1,1);
		    else
		    {
		    	printf("перебор за одну итерацию l=%d \n",l);
		    }

	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();

	    ADDC1(&T,&ST,&Z3);
	    c_lmax.CLR();
	    t_lmax.CLR();
	    CLEAR(&WLast); // тут должен быть h=N_K
	    CLEAR(&CLast);

	    klst=l;//SN.FND()+1;
	    for(i=klst;i<=sz;i++)
	    {
	       T.GetCol(&Y,i);
	       ADDROW(&WLast,&WW,i,&Y);
	       ADDROW(&CLast,&CC,i,&Y);
	       err = cudaGetLastError();
	       if (err>0)
	    	{
	    	   printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
	    	   return;
	    	}
	     }
	    j=0;
//		    printf("klst=%d\n",klst);

	    while (Z1.ZERO())//&&(j<10))
	    {
	        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
	        TCOPY(&WLast,&WT); // тут должен быть h=N_K
	        TCOPY(&CLast,&CT); // тут должен быть h=N_K

	        T.GetRow(&ST1,1);
	        T.GetRow(&SB1,lth);
//     if(j%1000==0)//||(j>15000))
 //   	 printf("%llu:%llu ",j, ST1.ToDigit());

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
	              printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
	              return;
	   		  }
	        }

	        LESS(&WT,&Z3,&w_max,&Z);
	        GREAT(&CT,&Z,&c_lmax,&Z2);
	    	err = cudaGetLastError();
	    	if (err>0){
	    	  printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
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
	            exit(0);
	       }
//	       if(j%1000==0)printf(" %llu ",j);
	  //     printf(" %llu: %llu ",j, SB1.ToDigit());
	    }
	    res.ASSIGN(&t_lmax);
}

void knapsack_bound(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{   unsigned int sz=T.size,lth=T.length;
	unsigned long long int i,j;//,w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
	unsigned long long int cloc,wloc,wlmin;//,wlast,clmax;
	Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),t_lmax(sz),w_lmax(N_K);

	Table WW(sz,N_K),CC(sz,N_K);
//	unsigned long long int ns;//tmp,dig,nj,
	Table WT(lth,N_K), CT(lth,N_K);
 	Table WLast(lth,N_K), CLast(lth,N_K);
 	unsigned int klst;
//	bool change;

	Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
	cudaError_t err;
	    branch_cut(sz, w,W, &ST,&SB);
//	    ST.print("sorted ST");
//	    SB.print("sorted SB");
//	    nj=ST.ToDigit();
//	    printf("count  %llu \n",SB.ToDigit()-nj);

	    // print results branch_cut
	    	for (i=0; i<sz;i++){
	    		w_max.FromDigit(w[i]);
	    		WW.SetRow(&w_max,i+1);
	    		w_max.FromDigit(c[i]);
	    		CC.SetRow(&w_max,i+1);
//	    		printf("<%i:%i,%i> ",i+1,w[i],c[i]);
	    	}
//	    	printf("\n");
	    	w_max.FromDigit(W);

//	init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
	init_stable(T);
	T.GetRow(&ST1,1);
//	ST1.print("ST1");
	T.GetRow(&SB1,lth);
//	SB1.print("SB1");
	klst=SB1.FND();
printf("*********************l=%i \n",klst);
/* сдвигаем диапазон, чтобы таблица делилась только на две части*/
	ST1.ASSIGN(&ST);
	SB1.NOT();
	ST1.AND(&SB1);
	ST1.print("ST corrected");

	SN.CLR();  if (klst>1) SN.setbit(klst-1,1);
	SN.print("SN");
    Z1.CLR();
    Z2.SET();
    Z3.SET();


    ADDC1(&T,&ST1,&Z3);
    c_lmax.CLR();
    t_lmax.CLR();
    CLEAR(&WLast); // тут должен быть h=N_K
    CLEAR(&CLast);
// puts("CLEAR");
    for(i=klst;i<=sz;i++)
    {
//    printf("klst=%i,i=%i\n", klst,i);
       T.GetCol(&Y,i);
 //      puts("getCOL");
       err = cudaGetLastError();
          if (err>0)
          {
             printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
             exit(0);
          }
       ADDROW(&WLast,&WW,i,&Y);
       err = cudaGetLastError();
          if (err>0)
          {
             printf("ADDROW WLast i=%llu %d , %s \n",i,err,cudaGetErrorString(err));
             return;
          }
       ADDROW(&CLast,&CC,i,&Y);
       err = cudaGetLastError();
       if (err>0)
    	{
    	   printf("ADDROW CLast i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
    	   exit(0);
    	}
     }

/*
    MIN(&WLast,&Z3,&Y);
//    puts("MIN");
    i=Y.FND();
//    printf("min row i=%i \n",i);
    WLast.GetRow(&u,i);
    wlmin=u.ToDigit();
    MAX(&CLast,&Z3,&Y);
//    puts("MAX");
    i=Y.FND();
    CLast.GetRow(&u,i);
    =u.ToDigit();
*/  wlmin=0;//потому что первая подстрока нулевая

    j=0;
//		    printf("klst=%d\n",klst);
    while (Z1.ZERO())//&&(j<10))
    {
        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
 ///////////   можно опустить и ниже использовать ADDC(&WLast,&Z3, u, &WT)
        TCOPY(&WLast,&WT); // тут должен быть h=N_K
        TCOPY(&CLast,&CT); // тут должен быть h=N_K
///////////
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
 //       printf("%i<>%i",i1,klst);
 // проверить на for if TMP(i)=1
        while ((i>0)&&(i<i1))
        {

        	cloc+=c[i-1];
        	wloc+=w[i-1];
//  	        	printf("<%i,%i> %d->%d %d->%d",j,i,c[i-1],cloc,w[i-1],wloc);
        	i=TMP.STEP();
        }
   if(wloc+wlmin<W){

	   	u.FromDigit(cloc);
        ADDC1(&CT, &u,&Z3);
        u.FromDigit(wloc);
        ADDC1(&WT, &u,&Z3);

        LESS(&WT,&Z3,&w_max,&Z);
        GREAT(&CT,&Z,&c_lmax,&Z2);
    	err = cudaGetLastError();
    	if (err>0){
    	  printf("after for j=%llu i=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
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
   }
//			   else {ST1.print("not");}
       j++;
//       SN.print("SN");
   	   ADDC1(&T,&SN,&Z3);

       err = cudaGetLastError();
       if (err>0){
            printf("after all %d , %s \n",err,cudaGetErrorString(err));
            exit(0);
       }
//    if(j%10==0)
//    	printf(" %llu ",j);
    }

	res.ASSIGN(&t_lmax);
	c_res=c_lmax.ToDigit();
}

void knapsack_experiment(unsigned int sz, unsigned lth)
{
	int//w[sz],c[sz],
		W,w_res,c_res;
	Slice res(sz);
	Table T(lth,sz);
	Slice ST(sz),SB(sz);
	ST.FromDigit(371617);SB.SET();
	if (InitAuxSlices(T.NN)>0){puts(" InitAuxSlices error");}
	double tt;
	struct timeval tv1,tv2;

	FILE *f = fopen("res/time.txt", "w");
	fprintf(f,"\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	fprintf(f,"count only \t optim_new \t optim \n");

//	problem_generate(sz, w, c);
//	W=rand()%(WMAX*sz/2)+1;

//	int w[sz]{63, 86, 33, 65, 75, 64, 49, 49, 35, 32, 29, 35, 95, 95, 48, 49, 2, 85, 23, 89, 1, 10, 69, 52, 97, 74, 40, 40, 51, 51, 77, 60};
//	int c[sz]{359, 724, 37, 504, 600, 62, 666, 403, 740, 719, 105, 263, 844, 566, 692, 610, 34, 334, 602, 397, 152, 549, 758, 555, 483, 578, 8, 234, 834, 751, 351, 630};
//	 W=137;

	//M=30
	int w[sz]={82, 90, 34, 24, 19, 61, 43, 53, 96, 42, 2, 63, 37, 40, 83, 56, 17, 46, 51, 89, 45, 76, 43, 87, 49, 21, 33, 68, 10, 13, 65, 59};
	int c[sz]={111, 531, 355, 188, 25, 955, 143, 201, 670, 680, 702, 466, 227, 292, 707, 916, 97, 902, 969, 669, 401, 187, 880, 490, 380, 603, 384, 170, 959, 344, 368, 833};
	W=416;
	printf("\n W=%i \n",W);
	branch_cut(sz, w,W, &ST,&SB);
	printf("\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	gettimeofday(&tv1,NULL);
//	knapsack_optim(w,c,W,T,w_res,c_res,res);
	complete_search(T, ST, SB);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	fprintf(f,"%.5f \t", tt);
	printf(" t=%.5f \n",tt);

	printf("knapsack_optim_new:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	gettimeofday(&tv1,NULL);
	knapsack_optim_new(w,c,W,T,w_res,c_res,res);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	fprintf(f,"%.2f \t", tt);
	printf(" t=%.2f \n",tt);
	res.print("result");

	res.CLR();
	printf("knapsack_optim:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	gettimeofday(&tv1,NULL);
	knapsack_optim(w,c,W,T,w_res,c_res,res);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	fprintf(f,"%.2f \t", tt);
	printf(" t=%.2f \n",tt);
	res.print("result");

	res.CLR();
		printf("knapsack_bound:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
		gettimeofday(&tv1,NULL);
		knapsack_bound(w,c,W,T,w_res,c_res,res);
		gettimeofday(&tv2,NULL);
		tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
		fprintf(f,"%.2f \t", tt);
		printf(" t=%.2f \n",tt);
		res.print("result");


	fclose(f);
}
void knapsack_exp()
{
	unsigned int sz=M,lth;
//	lth=powl(2,sz);
	lth=64*2048;//powl(2,sz);
	int
	w[sz],c[sz],W,w0,w01,i,j;//,nn;//,W_t=0,W_b=0,k_b,k_t;
    unsigned long long int cloc,wloc,wlast,wlmin;
	Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

	Table WW(sz,N_K),CC(sz,N_K),T(lth,sz);
	unsigned long long int nj,ns;//tmp,dig,
	if (InitAuxSlices(T.NN)>0){puts(" InitAuxSlices error");}
	double tt;
    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);
	Table WT(lth,N_K), CT(lth,N_K);
//	bool change;

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
// for (nn=0;nn<2;nn++)
 {
//	printf("\n %d \n",nn);
	problem_generate(M, w, c);
	W=rand()%(WMAX*M/2)+1;
//	branch_cut(M, w,W,k_t,k_b);
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
    fprintf(f," W=%d \t %llu \t",W,SB.ToDigit()-nj);
 /*/ сортировка по стоимости
    sort(c,w,sz);
    branch_cut(M, w,W, &ST,&SB);
    ST.print("log/ST");
    SB.print("log/SB");
    nj=ST.ToDigit();
    printf("Sorted by cost W=%d SN=%llu, ST=%llu, SB=%llu: %llu \n",W,ns,nj,SB.ToDigit(),SB.ToDigit()-nj);
 */
    // сортировка по весу
    sortdown(w,c,sz);
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
    		printf("<%i:%i,%i> ",i+1,w[i],c[i]);
    	}
    	printf("\n");
    	w_max.FromDigit(W);
    //вставить проверку, что ST<>SB, иначе тривиально.

/*
    printf("only count\n");
    gettimeofday(&tv1,NULL);
    j=0;
    while(Z1.ZERO())
    {

        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
    	ADDC1(&T,&SN,&Z3,&B); // к первой строке не добавляется, там максимум
    }

     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 fprintf(f,"%.2f \t", tt);
	 /****************************************************************************/
	 /*
	 printf("first\n");
	 gettimeofday(&tv1,NULL);
	init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
    Z1.CLR();
    Z2.SET();
    Z3.SET();
    ADDC1(&T,&ST,&Z3,&B);
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

   	   ADDC1(&T,&SN,&Z3,&B);

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

	 fprintf(f,"%.2f \t", tt);
/****************************************************************/
	 /*
	   printf("second\n");
	 	gettimeofday(&tv1,NULL);
		init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();
	    ADDC1(&T,&ST,&Z3,&B);
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

	   	   ADDC1(&T,&SN,&Z3,&B);

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

		 fprintf(f,"%.2f \t", tt);
/********************************************************************/
/*		 printf("third\n");
		 	gettimeofday(&tv1,NULL);
			init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);
		    Z1.CLR();
		    Z2.SET();
		    Z3.SET();
		    ADDC1(&T,&ST,&Z3,&B);
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
		        ADDC1(&CT, &u,&Z3,&B);
		        u.FromDigit(wloc);
		        ADDC1(&WT, &u,&Z3,&B);

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

		   	   ADDC1(&T,&SN,&Z3,&B);

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

			 fprintf(f,"%.2f  \n", tt);
/*************************************************************/
			 printf("fourth\n");
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
			    MIN(&WLast,&Z3,&Y);
			    i=Y.FND();
			    WLast.GetRow(&u,i);
			    wlmin=u.ToDigit();
			    MAX(&WLast,&Z3,&Y);
			    i=Y.FND();
			    WLast.GetRow(&u,i);
			    wlast=u.ToDigit();
			    j=0;
	//		    printf("klst=%d\n",klst);
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
			   if(wloc+wlmin<W){
				   w01=0;
				   for(i=klst-1;i>=i1;i--) w01+=w[i-1];
			        w0=W-wloc-wlast-w01;
//			    change=0;
			        i=i1;
			        while((i>0)&&(w0>0))
			        {
			        	if((ST1.getbit(i)==0)&&(w[i-1]<w0))
			        	{
			        		wloc+=w[i-1];
			        		cloc+=c[i-1];
			        		T.SetCol(&Z3,i);
			        		w0-=w[i-1];
//			        		printf("less 2^%d w[i] wloc=%llu w0=%d %d \n",i,wloc,w0,w01);
//			     change=1;
			        	}
			        	i--;
			        }

/*			        if (change)
			        { ST1.print("ST");
			          SB1.print("SB");
			        	T.GetRow(&TMP,1);
			        	TMP.print("new");
			        	change=0;
			        }
*/
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
			   }
//			   else {ST1.print("not");}
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

				 fprintf(f,"%.2f  \n", tt);
	/*************************************************************/
 }
			 fclose(f);
}
