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
{   srand( time(0));
	int i;
	for (i=0;i<n; i++)
	{
		w[i]=rand()%WMAX+1;
		c[i]=rand()%CMAX+1;
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
//количество предметов
#define M 32
//количество бит для максимального веса и цены
#define N_K 16

void knapsack_exp()
{   puts("knapsack 0");
	unsigned int N1,sz=M,lth;
//	lth=powl(2,sz);
	lth=64*2048;//powl(2,sz);
	int
//unsorted
/*	w[M]{80,65,83,53,41,48,98,17,81,64,22,91,54,32,63,59,2,70,97,68,67,70,9,62,79,
		73,69,47,46,78,9,54},
	c[M]{972,276,822,732,793,582,942,346,642,33,592,96,931,356,731,995,424,736,75,389,
			867,117,586,341,677,552,863,622,679,60,88,462}
*/
//sorted
	w[M]{98,97,	91,	83,81,80,79,78,73,70,70,69,68,67,65,64,63,62,59,54,54,53,48,47,46,
		41,32,22,17,9,9,2},
	c[M]{942,75,96,822,642,972,677,60,552,736,117,863,389,867,276,33,731,341,995,931,
		462,732,582,622,679,793,356,592,346,586,88,424}

	,W{1511},k_b,k_t,i,j,W_t=0,W_b=0;

	//problem_generate(M, w, c);
//	W=rand()%(WMAX*M/2)+1;

	puts("knapsack 1");
	Slice ST(sz),SB(sz),ST1(sz),SB1(sz),X(lth),w_max(N_K);


	Table WW(sz,N_K),CC(sz,N_K);
	puts("WT new");
	puts("Var");

// ST Верхняя строка проверяемого
// SB Нижняя строка проверяемого

//	Для проверки границ
	// ST1 Верхняя строка проверяемого
	// SB1 Нижняя строка проверяемого
//

	branch_cut(M, w,W,k_t,k_b);
	branch_cut(M, w,W, &ST,&SB);
	ST.print("st");
	SB.print("sb");

 //   ST->FromDigit(2569701075);// top bound
  //  (1463011463)
     //(264241151); //самый большой до падения на 3164 шаге

	unsigned long long int tmp,dig;
// print results branch_cut
	for (i=0; i<sz;i++){
//	  tmp=c[i]<<(64-32);
//	  dig=__brevll(tmp); //переворот в правильную сторону для суммирования
//		printf("<%i;%i> \n ",w[i],c[i]);
		w_max.FromDigit(w[i]);
		WW.SetRow(&w_max,i+1);
		w_max.FromDigit(c[i]);
		CC.SetRow(&w_max,i+1);
	}
     	w_max.FromDigit(W);
//		w_max->print("w_max",0);
	printf("\n W=%i <>%i\n",W,w_max.ToDigit());

	Table T(lth,sz);
//	unsigned long long int hostT[M][NN1];

	puts("Init T");
	printf("Размеры: size=%u length =%u, NN=%u IT=%u\n",T.size,T.length,T.NN,T.IT);
	// инициализация таблицы перебора
	// появление в ней sb - условие окончания перебора
	init_stable<<<sz,1>>>(T.get_device_pointer(),T.NN,sz);

//	T->writeToFile("log/init");
	cudaError_t err = cudaGetLastError();
	 printf("errors after init_stable %d\n",err);

	double tt;
    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);
	Table WT(lth,N_K), CT(lth,N_K);


	puts("WT new");

	Slice  Y(lth),Z(lth),Z1(lth),Z2(lth),Z3(lth), SN(sz),TMP(sz),u(N_K),v(N_K);


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
//    SN.print("будем складывать");
    Z1.CLR();
    Z2.SET();
    Z3.SET();
    ADDC1(&T,&ST,&Z3);
 //   T.GetRow(&TMP,1);
//    TMP.print("верхняя строка");
 //   T.GetRow(&TMP,lth);
 //   TMP.print("нижняя строка");
    Z2.setbit(1,0);
//    TMP->FromDigit(2533359615);//last maximum
//    T->SetRow(TMP,1);

    ST.print("log/ST");
    SN.print("log/SN");
    SB.print("log/SB");
    printf("SN=%llu, ST=%llu, SB=%llu \n",SN.ToDigit(),ST.ToDigit(),SB.ToDigit());
    //вставить проверку, что ST<>SB, иначе тривиально.


    printf("Начинаем перебор \n");
    j=0;

//Z1->print("Z1",0);
  //  while (Z1.ZERO())
    {

  // проверка диапазона
    	if ((j>3000))  printf("проверка диапазона %i\n",j);
        MATCH(&T,&Z3,&SB,&Z1);// тут должен быть h=H1
 //       Z1.print("Pos_SB");
 //       printf("%i \n",Z1->FND1());
 //       puts("MATCH");
        if ((j>3000))       puts("CLEAR(WT)");
    	CLEAR(&WT); // тут должен быть h=N_K
    	if ((j>3000))   	puts("CLEAR(CT)");
    	CLEAR(&CT); // тут должен быть h=N_K
//       puts("Clear WT and CT");
       for(i=1;i<=sz;i++)
    	{
          WW.GetRow(&v,i);
          if ((j>3000)) printf(" W.GetRow[%i]\n ",i);
          CC.GetRow(&u,i);
          if ((j>3000)) printf(" C.GetRow[%i]\n ",i);
 //         u.print("u");
 //         printf(" i=%i w=%i c=%i  \n",i,v.ToDigit(),u.ToDigit());
          T.GetCol(&Y,i);
  //        Y.print("Y");
          if ((j>3000)) printf(" T.GetCOL[%i]\n ",i);
          ADDC1(&WT,&v,&Y);
          if ((j>3000)) printf(" ADDC W\n ");
          ADDC1(&CT,&u,&Y);
          if ((j>3000)) printf(" ADDC C\n ");

          CT.GetRow(&u,2);
          WT.GetRow(&v,2);
 //         printf("2:j=%i weight sum= %i, another summ = %i \n",i, v.ToDigit(),u.ToDigit());
    	}

       LESS(&WT,&Z3,&w_max,&Z);
 //      Z.print("LESS");
       if (j>3000)      puts("Less");
 //      CT->writeToFile("\log\CT.dat");
       MAX(&CT,&Z,&X);
//       X.print("MAX_X");
       if ((j>3000))      puts("MAX");
       i=X.FND();
//       printf("max=%i\n",i);
       if (i>1)// смена максимума
       {

          T.GetRow(&TMP,i);
          T.SetRow(&TMP,1);
    //      row(1,CT)=row(i,CT);//необязательно, посчитается на следующем шаге
     //     row(1,WT)=row(i,WT);
          CT.GetRow(&u,1);
          printf("max changes %i %i last c=%i ",j,i,u.ToDigit());
          CT.GetRow(&u,i);
          CT.SetRow(&u,1);
          WT.GetRow(&v,i);
          WT.SetRow(&v,1);
   //       CT->GetRow(u,i);
   //       u->print("max",0);
          printf(" w=%i c=%i %llu \n",v.ToDigit(),u.ToDigit(),TMP.ToDigit());
          TMP.print("change");
       }

 //      puts("max changes");

       if ((j%100==0)|| (j>3000)){

           T.GetRow(&TMP,lth);
           printf(" %i: TMP=%llu \n ",j,TMP.ToDigit());

           gettimeofday(&tv2,NULL);
          	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
          	printf("time of all work %f sec \n", tt);
          	 tv1=tv2;
 //          TMP->print("tmp",0);
         }

 //     if (Z1.ZERO())
       {
  //    puts("next step");
    	   ADDC1(&T,&SN,&Z2); // к первой строке не добавляется, там максимум
    	   if (j>3000) puts("next step");
       }

       j++;

    }

     err = cudaGetLastError();
     printf("after init search table %d , %s \n",err,cudaGetErrorString(err));

     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);

	 T.GetRow(&ST1,2);
	 ST1.print("nextST");
	 T.GetRow(&SB1,lth);
	 SB1.print("nextSB");
	 T.GetRow(&ST1,1);
	 ST1.print("MAX");
	 CT.GetRow(&u,1);
	 WT.GetRow(&v,1);
	 printf("Max=%llu, NextSB=%llu \n",ST1.ToDigit(),SB1.ToDigit());
//	 printf("j=%i weight sum= %i, cost summ = %i \n",j, v.ToDigit(),u.ToDigit());


	 printf("time of all work seq %f sec \n", tt);

//	sprintf(str, "res/NP/test%d.txt",j);
//	T->writeToFile("res/NP/test.txt");
}
