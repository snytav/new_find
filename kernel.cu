
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

__global__ void test()
{
      cuPrintf("test");
}


int main()
{

    cudaPrintfInit();
    test << <1, 10 >> > ();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();



    return 0;
}
