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
//    puts("INIT");
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
void problem_generate(int n, int *w, int *c,int &w_all)
{   srand( time(NULL));
	int i;
	w_all=0;
	for (i=0;i<n; i++)
	{
		w[i]=rand()%WMAX+1;
		w_all+=w[i];
		c[i]=rand()%CMAX+1;
//		printf("<%i:%d,%d> ",i+1,w[i],c[i]);
	}
/*printf("w={%d",w[0]);
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
		*/
}

// vjue
void branch_cut(int n,int *w,int *c, int W, Slice *T,Slice *B)
{   int i;
int w_t=W;
int w_b=W;
int w_loc=0;
int c_loc=0;
//char str[n+1];
for (i=1;i<n+1;i++)
	{
	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				w_loc+=w[n-i];
				c_loc+=c[n-i];
				T->setbit(n-i+1,1);//нумерация с 1
				//str[n-i]='1';
			}
	else{// str[n-i]='0';
	 T->setbit(n-i+1,0);}
}
printf("TOP: w=%i c=%i  (%i)\n", w_loc,c_loc,W-w_loc);
w_loc=0;c_loc=0;
//str[n]=0;
//printf("Top \t %s \n",str);
for (i=0;i<n;i++)
{
	if (w[i]<=w_b)
	{   B->setbit(i+1,1);//нумерация с 1
		w_b-=w[i];
		w_loc+=w[i];
		c_loc+=c[i];
		//str[i]='1';
	}
	else {//str[i]='0';
		B->setbit(i+1,0);}
}
//str[n]=0;
printf("BOTTOM: w=%i c=%i (%i)\n", w_loc,c_loc,W-w_loc);
//printf("Bottom \t %s \n",str);
//T->print("bc_T");B->print("bc_B");
}

void branch_cut(int n,int *w,int *c, int W, Slice *T,Slice *B,int & C )
{   int i;
int w_t=W;
int w_b=W;
int w_loc=0;
int c_loc=0;
//char str[n+1];
for (i=1;i<n+1;i++)
	{
	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				w_loc+=w[n-i];
				c_loc+=c[n-i];
				T->setbit(n-i+1,1);//нумерация с 1
				//str[n-i]='1';
			}
	else{ //str[n-i]='0';
	T->setbit(n-i+1,0);}
}
printf("TOP: w=%i c=%i  (%i)\n", w_loc,c_loc,W-w_loc);
C=c_loc;
w_loc=0;c_loc=0;
//str[n]=0;
//printf("Top \t %s \n",str);
for (i=0;i<n;i++)
{
	if (w[i]<=w_b)
	{   B->setbit(i+1,1);//нумерация с 1
		w_b-=w[i];
		w_loc+=w[i];
		c_loc+=c[i];
		//str[i]='1';
	}
	else {//str[i]='0';
	B->setbit(i+1,0);}
}
if(C<c_loc) C=c_loc;
//str[n]=0;
printf("BOTTOM: w=%i c=%i (%i)\n", w_loc,c_loc,W-w_loc);
//printf("Bottom \t %s \n",str);
//T->print("bc_T");B->print("bc_B");
}

void branch_cut1(int n,int *w,int *c, int W, Slice *T,Slice *B)
{   int i;
int w_t=W;
int w_b=W;
int w_loc=0;
int w_max=0;
int c_loc=0;
//char str[n+1];
i=1;
T->CLR();
	while((i<n+1)&&(w[n-i]<=w_t))
//for (i=1;i<n+1;i++)
	{
//	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				w_loc+=w[n-i];
				c_loc+=c[n-i];
				T->setbit(n-i+1,1);//нумерация с 1
			//	str[n-i]='1';
			}
//	else{ str[n-i]='0'; T->setbit(n-i+1,0);}
	i++;
}
printf("TOP: w=%i c=%i (%i)\n", w_loc,c_loc,W-w_loc);
w_loc=0;c_loc=0;
//str[n]=0;
//printf("Top \t %s \n",str);
B->CLR();
i=0;
	//for (i=0;i<n;i++)
	while((i<n)&&(w[i]<=w_b))
{
//	if (w[i]<=w_b)
	{   B->setbit(i+1,1);//нумерация с 1
		w_b-=w[i];
		w_loc+=w[i];
		c_loc+=c[i];
//		str[i]='1';
		if(w_max<w[i]) w_max=w[i];
	}
//	else {str[i]='0';B->setbit(i+1,0);}
	i++;
}
//str[n]=0;
printf("BOTTOM: w=%i c=%i (%i,%i)\n", w_loc,c_loc,W-w_loc, W-w_loc+w_max);
//printf("Bottom \t %s \n",str);
//T->print("bc_T");B->print("bc_B");
}



// подряд идущие, c пропусками
void branch_cut(int n, int *w, int *c, int W, int &k_t, int &k_b)
{   int i,k_tmp=0;
	int w_t=W;
	int w_b=W;
	int w_loc=0;
	int c_loc=0;
	k_t=n;
	k_b=0;
	for (i=0;i<n;i++)
	{
		if (w[i]<=w_b)
		{//   printf("%i",1);
			w_b-=w[i];
			w_loc+=w[i];
			c_loc+=c[i];
			k_b++;

		}else
		{
		//	printf("%i",0);
			if (k_tmp==0) k_tmp=k_b;
		}

	}
    printf("TOP: w=%i c=%i \n", w_loc,c_loc);
    w_loc=0;c_loc=0;
	k_b=(k_tmp==0)?k_b:k_tmp+1;
	i=1; k_tmp=0;
//	puts("Mirrored");
	for (i=1;i<n+1;i++)
		{
		if(w[n-i]<=w_t)
				{ //printf("%i",1);
					w_t-=w[n-i];
					w_loc+=w[n-i];
					c_loc+=c[n-i];
					k_t--;
				}
		else
				{
					//printf("%i",0);
					if (k_tmp==0) k_tmp=k_t;
				}
	}
	printf("BOTTOM: w=%i c=%i \n", w_loc,c_loc);
		k_t=(k_tmp==0)?k_t:k_tmp;
		printf("   %i:%i %i ",k_b,k_t ,w_t);
   	 puts("");

}

void sortdown(int *first,int *second, unsigned int size)
{	int tmp;
	for(unsigned int i=0;i<size;i++)
		for(unsigned int j=0;j<size-i-1;j++)
		 if (first[j]<first[j+1])
		 {
			tmp=first[j+1];
			first[j+1]=first[j];
			first[j]=tmp;

			tmp=second[j+1];
			second[j+1]=second[j];
			second[j]=tmp;
		 }
}

void sortdown_udel(int *first,int *second, unsigned int size)
{	int tmp;
    float ud[size];
    for(unsigned int i=0;i<size;i++) ud[i]=(float)first[i]/second[i];
	for(unsigned int i=0;i<size-1;i++)
		for(unsigned int j=0;j<size-i;j++)
		 if (ud[j]<ud[j+1])
		 {
			 tmp=ud[j+1];
			 ud[j+1]=ud[j];
			 ud[j]=tmp;

			tmp=first[j+1];
			first[j+1]=first[j];
			first[j]=tmp;

			tmp=second[j+1];
			second[j+1]=second[j];
			second[j]=tmp;
		 }
}

void sortup_udel(int *first,int *second, unsigned int size)
{	int tmp;
    float ud[size];
    for(unsigned int i=0;i<size;i++) ud[i]=((float)first[i])/second[i];
	for(unsigned int i=1;i<size-1;i++)
		for(unsigned int j=0;j<size-i;j++)
		 if (ud[j]>ud[j+1])
		 {
			 tmp=ud[j+1];
			 ud[j+1]=ud[j];
			 ud[j]=tmp;

			tmp=first[j+1];
			first[j+1]=first[j];
			first[j]=tmp;

			tmp=second[j+1];
			second[j+1]=second[j];
			second[j]=tmp;
		 }
	print_array("Ud", ud,size);
}
void sortup(int *first,int *second, unsigned int size)
{	int tmp;
	for(unsigned int i=0;i<size-1;i++)
		for(unsigned int j=0;j<size-i;j++)
		 if (first[j]>first[j+1])
		 {
			tmp=first[j+1];
			first[j+1]=first[j];
			first[j]=tmp;

			tmp=second[j+1];
			second[j+1]=second[j];
			second[j]=tmp;
		 }
}

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

unsigned long long int complete_search(Table T, Slice ST,Slice SB)
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
//	printf("lth=%i ",lth);
    T.GetRow(&SB1,lth);
// puts("Row last");
    err = cudaGetLastError();
             if (err>0) printf("errors after GetRow lth %d:%s\n",err, cudaGetErrorString(err));
    l=SB1.FND();
// puts("SB1.FND");
//    printf("l=%i \n",l);
//    SB1.print("SB1");
    SN.CLR();
    if (l>1) SN.setbit(l-1,1);
    else
    {
//    	printf("перебор за одну итерацию l=%d \n",l);
    }
	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();
	    ST1.ASSIGN(&ST);
	    SB1.NOT();
	    ST1.AND(&SB1);
//ST.print("ST before ADDC1"); ST1.print("ST corrected ");
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
	    return j;
}

void knapsack_optim_new(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{  unsigned int sz=T.size,lth=T.length;
//unsigned long long int w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
unsigned long long int i,j,k=0;
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
    branch_cut(sz, w,c,W, &ST,&SB);
//    ST.print("ST");
//    SB.print("SB");
//////////////////////////////
    ST.ASSIGN(&SB);
    ST.setbit(1,0);
    t_lmax.ASSIGN(&SB);
/////////////////////////////////

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
	//	    	printf("перебор за одну итерацию l=%d \n",l);
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
	        GREAT(&WT,&Z3,&w_max,&Z);Z.NOT();
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
	          k++;
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
	    w_res=w_lmax.ToDigit();
	    c_res=c_lmax.ToDigit();
	    res.ASSIGN(&t_lmax);
	    printf("j=%llu k=%llu\n",j,k);
}
///////////////////////////////////////
void knapsack_optim_new1(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{  unsigned int sz=T.size,lth=T.length;
//unsigned long long int w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
unsigned long long int i,j,k=0;
unsigned long long int cloc,wloc;//,wlast,wlmin;
int cost_max=0;
Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

Table WW(sz,N_K),CC(sz,N_K);
unsigned long long int nj,ns;//tmp,dig,
Table WT(lth,N_K), CT(lth,N_K);
	Table WLast(lth,N_K), CLast(lth,N_K);
	unsigned int klst,l;
// bool change;

Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
cudaError_t err;
    branch_cut(sz, w,c,W, &ST,&SB,cost_max);
//    ST.print("ST");
    ST.ASSIGN(&SB);
    ST.setbit(1,0);
    t_lmax.ASSIGN(&SB);
    c_lmax.FromDigit(cost_max);

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
	//	    	printf("перебор за одну итерацию l=%d \n",l);
		    }

	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();

	    ST1.ASSIGN(&ST);
	    SB1.NOT();
	    ST1.AND(&SB1);
	    nj=ST1.ToDigit();
//	    printf("count  %llu : j<%llu \n",SB.ToDigit()-nj,(SB.ToDigit()-nj)/T.length);
//	    ST1.print("ST");
//	    SB.print("SB");

	    ADDC1(&T,&ST1,&Z3);

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
	    MAX(&CLast,&Y,&X);
	    i=X.FND();
	    CLast.GetRow(&u,i);

	    int cost_teal=u.ToDigit();
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
	  if((wloc<=W)&&((cloc+cost_teal)>cost_max))
		{
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
	        GREAT(&WT,&Z3,&w_max,&Z); Z.NOT();
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
	          cost_max=c_lmax.ToDigit();
	          k++;
//	          t_lmax.print("local");
//	          printf("j=%i i=%i C=%llu W=%llu\n",j,i,c_lmax.ToDigit(),w_lmax.ToDigit());
	       }

	       j++;
		}
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
	    w_res=w_lmax.ToDigit();
	    c_res=c_lmax.ToDigit();
	    res.ASSIGN(&t_lmax);
	    printf("j=%llu k=%llu \n",j,k);
}

/////////////////////////////////////////////////////////////////////////////////
void knapsack_optim_new2(int *w,int *c,int W, Table T,int &w_res, int &c_res, Slice res)
{  unsigned int sz=T.size,lth=T.length;
//unsigned long long int w0,w01,nn;//,W_t=0,W_b=0,k_b,k_t;
unsigned long long int i,j,k=0;
unsigned long long int cloc,wloc;//,wlast,wlmin;
int cost_max=0;
Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K),c_lmax(N_K),w_lmax(N_K),t_lmax(sz);

Table WW(sz,N_K),CC(sz,N_K);
unsigned long long int nj,ns;//tmp,dig,
Table WT(lth,N_K), CT(lth,N_K);
	Table WLast(lth,N_K), CLast(lth,N_K);
	unsigned int klst,l;
// bool change;

Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);
cudaError_t err;
    branch_cut(sz, w,c,W, &ST,&SB,cost_max);
//    ST.print("ST");
    ST.ASSIGN(&SB);
    ST.setbit(1,0);
    t_lmax.ASSIGN(&SB);
    c_lmax.FromDigit(cost_max);

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
	//	    	printf("перебор за одну итерацию l=%d \n",l);
		    }

	    Z1.CLR();
	    Z2.SET();
	    Z3.SET();

	    ST1.ASSIGN(&ST);
	    SB1.NOT();
	    ST1.AND(&SB1);
	    nj=ST1.ToDigit();
//	    printf("count  %llu : j<%llu \n",SB.ToDigit()-nj,(SB.ToDigit()-nj)/T.length);
//	    ST1.print("ST");
//	    SB.print("SB");

	    ADDC1(&T,&ST1,&Z3);

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
	    MAX(&CLast,&Y,&X);
	    i=X.FND();
	    CLast.GetRow(&u,i);

	    int cost_teal=u.ToDigit();
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

//	        i=TMP.STEP();
//	        while ((i>0)&&(i<i1))
	        for(i=1;i<i1;i++)
	        {
//printf(" %i ",i);
	        	if(TMP.getbit(i)==1){
	        	cloc+=c[i-1];
	        	wloc+=w[i-1];}
//  	        	printf("<%i,%i> %d->%d %d->%d",j,i,c[i-1],cloc,w[i-1],wloc);
	        //	i=TMP.STEP();
	        }
	  if((wloc<=W)&&((cloc+cost_teal)>cost_max))
		{
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
	        GREAT(&WT,&Z3,&w_max,&Z); Z.NOT();
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
	          cost_max=c_lmax.ToDigit();
	          k++;
//	          t_lmax.print("local");
//	          printf("j=%i i=%i C=%llu W=%llu\n",j,i,cost_max,w_lmax.ToDigit());
	       }

	       j++;
		}
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
	    w_res=w_lmax.ToDigit();
	    c_res=c_lmax.ToDigit();
	    res.ASSIGN(&t_lmax);
	    printf("j=%llu k=%llu \n",j,k);
}
/////////////////////////////////////////////////////////////////
void knapsack_partitial(int *w,int *c,int W,unsigned int M1,unsigned int lth, int &w_res, int &c_res, Slice res)
{
	Table T1(lth,M-M1),T2(lth,M1);
	init_stable(T1);init_stable(T2);
}
////////////////////////////////////////////////////////////////////////////////
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
    branch_cut(sz, w,c,W, &ST,&SB);
 //   ST.print("sorted ST");
 //   SB.print("sorted SB");
    nj=ST.ToDigit();
//   printf("count  %llu : j<%llu \n",SB.ToDigit()-nj,(SB.ToDigit()-nj)/T.length);

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
	//	    	printf("перебор за одну итерацию l=%d \n",l);
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
	    branch_cut(sz, w,c,W, &ST,&SB);
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
//printf("*********************l=%i \n",klst);
/* сдвигаем диапазон, чтобы таблица делилась только на две части*/
	ST1.ASSIGN(&ST);
	SB1.NOT();
	ST1.AND(&SB1);
//	ST1.print("ST corrected");

	SN.CLR();  if (klst>1) SN.setbit(klst-1,1);
//	SN.print("SN");
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
    	   printf("ADDROW CLast i=%llu j=%llu %d , %s \n",j,i,err,cudaGetErrorString(err));
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
	c_lmax.print("Cmax");
	w_lmax.print("Wmax");
	c_res=c_lmax.ToDigit();
}

void print_array(const char * name, int *ar, int sz)
{
  printf("%s={%i(1)",name,ar[0]);
  for (int i=1;i<sz;i++) printf(", %i(%i)",ar[i],i+1);
  printf("}\n");
}
void print_array(const char * name, float *ar, int sz)
{
  printf("%s={%0.2f(1)",name,ar[0]);
  for (int i=1;i<sz;i++) printf(", %0.2f(%i)",ar[i],i+1);
  printf("}\n");
}

void knapsack_experiment1()
{
	int W;//,j,w_res,c_res;
//	if (InitAuxSlices(T.NN)>0){puts(" InitAuxSlices error");}
//	double tt;
//	struct timeval tv1,tv2;
	//unsigned long long int iter_count;

	FILE *f = fopen("res/time.txt", "w");
//	fprintf(f,"\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
//	printf("\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
//	fprintf(f,"count only \t optim_new \t optim \n");

	int w[M],c[M];
	problem_generate(M, w, c,W);
	W=W*0.5;
	Slice ST(M),SB(M);
	if (InitAuxSlices(ST.NN)>0){puts(" InitAuxSlices error");}
//	sortdown_udel(c,w, M);
	sortup_udel(c,w, M);
//	print_array("c", c,M);
//	print_array("w", w,M);
	printf("\nM=%i W=%i \n",M,W);
	branch_cut1(M, w,c,W, &ST,&SB);
	ST.print("Top");
//	SB.print("Bottom");
	// проверка границ
//	ST.ASSIGN(&SB);
//	ST.NOT();
//	ST.print("NOT");
	int low_bound=ST.FND();
	if (low_bound>0) low_bound-=1;
	printf("table to %i  (%i) : %i x %f\n",low_bound+1,M,low_bound,pow(2,low_bound));
//	sortdown(&w[0],&c[0],low_bound);
//	print_array("c", c,M);
//	print_array("w", w,M);
	branch_cut(M, w,c,W, &ST,&SB);
	ST.print("TOP2");
	low_bound=ST.FND();
	printf("table from %i \n",low_bound);
}

void knapsack_experiment(unsigned int sz, unsigned int lth,unsigned int M1)
{
	int j,	W,w_res,c_res;
	Slice res(sz);
	Table T(lth,sz);
	Slice X(lth);
	Slice ST(sz),SB(sz);
//	ST.FromDigit(371617);SB.SET();
	if (InitAuxSlices(X.NN)>0){puts(" InitAuxSlices error");}
	double tt;
	struct timeval tv1,tv2;
//	unsigned long long int iter_count;

	FILE *f = fopen("res/time.txt", "w");
	fprintf(f,"\n complete_search:M=%i blocks=%d it=%d %dx%d,NN=%d \n",M, X.blocks,X.IT, sz,lth,X.NN);
	printf("\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",X.blocks,X.IT, sz,lth,X.NN);
	fprintf(f,"count only \t optim_new \t optim \n");
/*
	int w[sz],c[sz];
	problem_generate(sz, w, c,W);
	W=W*0.5;
	//W=rand()%(WMAX*sz/2)+1;
*/

//	int w[sz]{63, 86, 33, 65, 75, 64, 49, 49, 35, 32, 29, 35, 95, 95, 48, 49, 2, 85, 23, 89, 1, 10, 69, 52, 97, 74, 40, 40, 51, 51, 77, 60};
//	int c[sz]{359, 724, 37, 504, 600, 62, 666, 403, 740, 719, 105, 263, 844, 566, 692, 610, 34, 334, 602, 397, 152, 549, 758, 555, 483, 578, 8, 234, 834, 751, 351, 630};
//	 W=137;

	//M=32
	//int w[sz]={82, 90, 34, 24, 19, 61, 43, 53, 96, 42, 2, 63, 37, 40, 83, 56, 17, 46, 51, 89, 45, 76, 43, 87, 49, 21, 33, 68, 10, 13, 65, 59};
	//int c[sz]={111, 531, 355, 188, 25, 955, 143, 201, 670, 680, 702, 466, 227, 292, 707, 916, 97, 902, 969, 669, 401, 187, 880, 490, 380, 603, 384, 170, 959, 344, 368, 833};
	//W=416;
	//M=33
	int c[M]={384, 977, 641, 626, 978, 830, 710, 426, 169, 655, 940, 953, 364, 340, 867, 581, 268, 691, 426, 439, 618, 739, 170, 261, 509, 161, 507, 227, 361, 166, 68, 103, 81};
	int w[M]={81, 5, 5, 10, 18, 16, 23, 22, 9, 36, 51, 56, 23, 25, 66, 46, 22, 65, 45, 54, 74, 83, 27, 42, 89, 30, 100, 49, 98, 73, 46, 83, 98};
	W=785;
	print_array("c bS", c,sz);
	print_array("w bS", w,sz);
	sortdown_udel(c,w, sz);
	print_array("c aS", c,sz);
	print_array("w aS", w,sz);

	printf("\n W=%i \n",W);
	branch_cut(sz, w,c,W, &ST,&SB);
	ST.print("Top");
	SB.print("Bottom");
	int low_bound=ST.FND();
	printf("low=%i\n",low_bound);
	low_bound-=1;
/*	sortdown(&w[0],&c[0],low_bound);
	print_array("c", c,low_bound);
	print_array("w", w,low_bound);
	branch_cut(sz, w,c,W, &ST,&SB);

	print_array("c", c,M);
	print_array("w", w,M);
	ST.print("Top1");
	//SB.print("Bottom");
/*
	printf("\n complete_search:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	gettimeofday(&tv1,NULL);
//	knapsack_optim(w,c,W,T,w_res,c_res,res);
	for(j=0;j<run_count;j++)
	  iter_count=complete_search(T, ST, SB);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	tt=tt/run_count;
	fprintf(f,"%.5f \t", tt);
	printf(" t=%.5f \n  iter_count=%llu \n",tt,iter_count);
*/
/*	printf("knapsack_optim_new:blocks=%d it=%d %dx%d,NN=%d \n",X.blocks,X.IT, sz,lth,X.NN);
	gettimeofday(&tv1,NULL);
	for(j=0;j<run_count;j++)
	    knapsack_optim_new(w,c,W,T,w_res,c_res,res);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	tt=tt/run_count;
	fprintf(f,"%.5f \t", tt);
	printf(" t=%.5f w_res=%i c_res=%i\n",tt,w_res,c_res);
	res.print("result");


	printf("knapsack_optim_new:blocks=%d it=%d %dx%d,NN=%d \n",X.blocks,X.IT, sz,lth,X.NN);
		gettimeofday(&tv1,NULL);
		for(j=0;j<run_count;j++)
		    knapsack_optim_new1(w,c,W,T,w_res,c_res,res);
		gettimeofday(&tv2,NULL);
		tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
		tt=tt/run_count;
		fprintf(f,"%.5f \t", tt);
		printf(" t=%.5f w_res=%i c_res=%i\n",tt,w_res,c_res);
		res.print("result");

		printf("knapsack_optim_new:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
			gettimeofday(&tv1,NULL);
			for(j=0;j<run_count;j++)
			    knapsack_optim_new2(w,c,W,T,w_res,c_res,res);
			gettimeofday(&tv2,NULL);
			tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
			tt=tt/run_count;
			fprintf(f,"%.5f \t", tt);
			printf(" t=%.5f w_res=%i c_res=%i\n",tt,w_res,c_res);
			res.print("result");
*/
			knapsack_partitial(w,c,W,M1,lth,w_res,c_res,res);
/*
	res.CLR();
	printf("knapsack_optim:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
	gettimeofday(&tv1,NULL);
	for(j=0;j<run_count;j++)
	   knapsack_optim(w,c,W,T,w_res,c_res,res);
	gettimeofday(&tv2,NULL);
	tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	tt=tt/run_count;
	fprintf(f,"%.5f \t", tt);
	printf(" t=%.5f \n",tt);
	res.print("result");

	res.CLR();
		printf("knapsack_bound:blocks=%d it=%d %dx%d,NN=%d \n",T.blocks,T.IT, sz,lth,T.NN);
		gettimeofday(&tv1,NULL);
		for(j=0;j<run_count;j++)
			knapsack_bound(w,c,W,T,w_res,c_res,res);
		gettimeofday(&tv2,NULL);
		tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
		tt=tt/run_count;
		fprintf(f,"%.5f \t", tt);
		printf(" t=%.5f \n",tt);
		res.print("result");

*/
	fclose(f);
}
