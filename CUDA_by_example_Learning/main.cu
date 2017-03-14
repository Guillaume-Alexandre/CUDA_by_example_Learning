//kernel
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
	structure thing;

	function(thing.int_structure);

	return 0;

}
