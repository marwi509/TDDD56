#include <stdio.h>

#define N 10000

__global__ void add(int *a,int *b,int *c)
{
	int tid = blockIdx.x;
	if(tid < N)
		c[tid] = tid; // a[tid] + b[tid];
}	



int main(void)
{
	int i;
	// The three arrays on the CPU
	int a[N], b[N], c[N];
	
	// Pointers that will be allocated on the GPU (device)
	int *dev_a, *dev_b, *dev_c;
	
	// Allocate memory on the device, the size of the whole array
	if(cudaMalloc(&dev_a, N*sizeof(int)) != cudaSuccess)
		printf("Error!");
	if(cudaMalloc(&dev_b, N*sizeof(int)) != cudaSuccess)
		printf("Error!");
	if(cudaMalloc(&dev_c, N*sizeof(int)) != cudaSuccess)
		printf("Error!");
		
	for(i = 0;i<N;i++)
	{
		a[i] = 1;
		b[i] = i*i;
	}
	
	// Copy the arrays to the device
	cudaMemcpy(dev_a, a, N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N*sizeof(int),cudaMemcpyHostToDevice);
	
	add<<<N,1>>>(dev_a,dev_b,dev_c);
	
	// Copy the array back to the host
	if(cudaMemcpy(c, dev_c, N*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error!");
	
	for(i = 0; i<N;i++)
		printf("%d\n",c[i]);
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
