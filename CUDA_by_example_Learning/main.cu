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
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no "
			"speed up from streams\n");
		return 0;
	}

	//Initialise timers
	cudaEvent_t start, stop;
	float elapsedTime;
	// start the timers
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// initialize the stream we want to use for the application
	cudaStream_t stream;
	HANDLE_ERROR(cudaStreamCreate(&stream));

	
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;
	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a,
		N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b,
		N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c,
		N * sizeof(int)));

	// allocate page-locked memory, used to stream
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	for (int i = 0; i<FULL_DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	// now loop over full data, in bite-sized chunks
	//It is important in the case where GPU has much less memory than the host
	for (int i = 0; i<FULL_DATA_SIZE; i += N) {
		// copy the locked memory to the device, async
		HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i,
			N * sizeof(int),
			cudaMemcpyHostToDevice,
			stream));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i,
			N * sizeof(int),
			cudaMemcpyHostToDevice,
			stream));
		//compute
		kernel << <N / 256, 256, 0, stream >> >(dev_a, dev_b, dev_c);
		// copy the data from device to locked memory
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c,
			N * sizeof(int),
			cudaMemcpyDeviceToHost,
			stream));
	}
	// copy result chunk from locked to full buffer
	HANDLE_ERROR(cudaStreamSynchronize(stream));

	//stop timers
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);


	// cleanup the streams and memory
	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));
	//destroy stream
	HANDLE_ERROR(cudaStreamDestroy(stream));
	return 0;
}
