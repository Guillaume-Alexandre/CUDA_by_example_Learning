//kernel

#ifndef DEFINITIONS
#define DEFINITIONS

#define DIM 800
#define PI 3.14

#endif // !DEFINITIONS

#ifndef STRUCT
#define STRUCT

struct structure {
	int int_structure = 0;
};

#endif // !STRUCT

#ifndef HEAD_KERNEL
#define HEAD_KERNEL


__global__ void kernel(void);

__global__ void kernel(void) {
	printf("Hello World!");
}





#endif // !HEAD_KERNEL