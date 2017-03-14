//kernel

#ifndef DEFINITIONS
#define DEFINITIONS

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

#endif // !DEFINITIONS

#ifndef STRUCT
#define STRUCT

#endif // !STRUCT

#ifndef HEAD_KERNEL
#define HEAD_KERNEL

__global__ void kernel(int *a, int *b, int *c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}




#endif // !HEAD_KERNEL