
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>



__global__ void kernel(double* arr, const double* a1, const double* a2, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < n) {
		if (a1[idx] < a2[idx])
			arr[idx] = a1[idx];
		else arr[idx] = a2[idx];
		idx += offset;
	}
}


int main() {
	int i, n = 0;
	std::ios_base::sync_with_stdio(false);
	std::cin >> n;

	double* a1 = (double*)malloc(sizeof(double) * n);
	for (i = 0; i < n; i++) std::cin >> a1[i];
	double* dev_a1 = 0;
	cudaMalloc((void**)&dev_a1, sizeof(double) * n);
	cudaMemcpy(dev_a1, a1, sizeof(double) * n, cudaMemcpyHostToDevice);
	double* a2 = (double*)malloc(sizeof(double) * n);
	for (i = 0; i < n; i++) std::cin >> a2[i];
	double* dev_a2 = 0;
	cudaMalloc((void**)&dev_a2, sizeof(double) * n);
	cudaMemcpy(dev_a2, a2, sizeof(double) * n, cudaMemcpyHostToDevice);
	double* dev_arr = 0;
	cudaMalloc((void**)&dev_arr, sizeof(double) * n);
	kernel << < 256, 256 >> > (dev_arr, dev_a1, dev_a2, n);
	double* arr = (double*)malloc(sizeof(double) * n);
	cudaDeviceSynchronize();
	cudaGetLastError();
	cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost);
	for (i = 0; i < n; i++)
		std::cout << std::setprecision(10) << std::scientific << arr[i] << "  ";
	cudaFree(dev_a1);
	cudaFree(dev_a2);
	cudaFree(dev_arr);
	free(a1);
	free(a2);
	free(arr);
	return 0;
}
