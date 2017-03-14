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
#endif // !LIB

#include "kernel.cuh"
#include "functions.cuh"

int main(void) {
	cudaDeviceProp prop;
	int whichDevice;

	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	for (int i = 0; i < FULL_DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	for (int i = 0; i < FULL_DATA_SIZE; i += N) {
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

		kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time taken: %3.1f ms\n", elapsedTime);

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaStreamDestroy(stream);
	return 0;
}