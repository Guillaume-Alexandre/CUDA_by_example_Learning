//main

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
	float           elapsedTime;
	float           MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;


	// try it with cudaMalloc
	elapsedTime = cuda_malloc_test(SIZE, true);
	printf("Time using cudaMalloc:  %3.1f ms\n",
		elapsedTime);
	printf("\tMB/s during copy up:  %3.1f\n",
		MB / (elapsedTime / 1000));

	elapsedTime = cuda_malloc_test(SIZE, false);
	printf("Time using cudaMalloc:  %3.1f ms\n",
		elapsedTime);
	printf("\tMB/s during copy down:  %3.1f\n",
		MB / (elapsedTime / 1000));

	// now try it with cudaHostAlloc
	elapsedTime = cuda_host_alloc_test(SIZE, true);
	printf("Time using cudaHostAlloc:  %3.1f ms\n",
		elapsedTime);
	printf("\tMB/s during copy up:  %3.1f\n",
		MB / (elapsedTime / 1000));

	elapsedTime = cuda_host_alloc_test(SIZE, false);
	printf("Time using cudaHostAlloc:  %3.1f ms\n",
		elapsedTime);
	printf("\tMB/s during copy down:  %3.1f\n",
		MB / (elapsedTime / 1000));

	scanf("");
}
