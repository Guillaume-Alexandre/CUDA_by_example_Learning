//main

//####################################################################################################
// DOT PRODUCT USING ZEO MEMORY
//####################################################################################################

#ifndef ALL
#define ALL
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#endif // !ALL

#ifndef LIB
#define LIB
#include "lib\book.h"
#include "lib\cpu_anim.h"
#endif // !LIB

#include "kernel.cuh"
#include "functions.cuh"

int main(void) {
	//check device properties
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (prop.canMapHostMemory != 1) {
		printf("Device cannot map memory.\n");
		return 0;
	}
	//tell the GPU to make Map memory
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
	//copy memory from CPU to GPU
	float elapsedTime = malloc_test(N);
	printf("Time using cudaMalloc: %3.1f ms\n",
		elapsedTime);
	//Zerocopy memory
	elapsedTime = cuda_host_alloc_test(N);
	printf("Time using cudaHostAlloc: %3.1f ms\n",
		elapsedTime);
	return 0;
}
