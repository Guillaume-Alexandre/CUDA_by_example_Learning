//functions

#ifndef HEAD_FUNCTIONS
#define HEAD_FUNCTIONS

void function(int arg);

void function(int arg) {
	//init
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	//kernel call
	kernel << <blocks, threads >> > ();
}




#endif // !HEAD_FUNCTIONS