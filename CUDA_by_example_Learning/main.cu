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
		printf("Device will not handle overlaps, so no"
			"speed up from streams\n");
		return 0;
	}
	//initialize timers
	cudaEvent_t start, stop;
	float elapsedTime;
	// start the timers
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// initialize the streams
	cudaStream_t stream0, stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));

	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0; //GPU buffers for stream0
	int *dev_a1, *dev_b1, *dev_c1; //GPU buffers for stream1
	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

	// allocate page-locked memory, used to stream
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

	//Fill input buffers on the host
	for (int i = 0; i<FULL_DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}
	
	// now loop over full data, in bite-sized chunks
	for (int i = 0; i<FULL_DATA_SIZE; i += N * 2) {
		// copy the locked memory to the device, async
		HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		kernel << <N / 256, 256, 0, stream0 >> >(dev_a0, dev_b0, dev_c0);
		// copy the data from device to locked memory
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
		// copy the locked memory to the device, async
		HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice,stream1));
		kernel << <N / 256, 256, 0, stream1 >> >(dev_a1, dev_b1, dev_c1);
		// copy the data from device to locked memory
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	//synchronization of the two streams to then stop the timers
	HANDLE_ERROR(cudaStreamSynchronize(stream0));
	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);


	// cleanup the streams and memory
	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));

	HANDLE_ERROR(cudaFree(dev_a0));
	HANDLE_ERROR(cudaFree(dev_b0));
	HANDLE_ERROR(cudaFree(dev_c0));
	HANDLE_ERROR(cudaFree(dev_a1));
	HANDLE_ERROR(cudaFree(dev_b1));
	HANDLE_ERROR(cudaFree(dev_c1));
	HANDLE_ERROR(cudaStreamDestroy(stream0));
	HANDLE_ERROR(cudaStreamDestroy(stream1));
	return 0;
}