//kernel

#ifndef DEFINITIONS
#define DEFINITIONS

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

#endif // !DEFINITIONS

#ifndef STRUCT
#define STRUCT

struct structure {
	int int_structure = 0;
};

#endif // !STRUCT

#ifndef HEAD_KERNEL
#define HEAD_KERNEL


__global__ void dot(int size, float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < size) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	// set the cache values
	cache[cacheIndex] = temp;
	// synchronize threads in this block
	__syncthreads();
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}



#endif // !HEAD_KERNEL