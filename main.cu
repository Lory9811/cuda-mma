#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <random>
#include <string>

#include <cuda.h>
#include <mma.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace nvcuda;

// mm with tensor cores
__global__ void wmma16x16(half* a, half* b, half* c) {
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
	
	wmma::load_matrix_sync(a_frag, a, 16);
	wmma::load_matrix_sync(b_frag, b, 16);

	wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

	wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_row_major);
}

// mm without tensor cores
__global__ void mma16x16(half* a, half* b, half* c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 16 || j >= 16) {
		return;
	}

	half sum = c[i + j * 16];
	for (int k = 0; k < 16; k++) {
		sum += a[j * 16 + k] * b[i + k * 16];
	}

	c[i + j * 16] = sum;
}

std::vector<half> randomMatrix(std::uniform_real_distribution<float> distribution, int size) {
	std::random_device rd;
	std::default_random_engine engine(rd());
	std::vector<half> matrix;
	
	for (int i = 0; i != size; i++) {
		matrix.emplace_back(__float2half(distribution(engine)));
	}

	return matrix;
}

void checkCudaError(cudaError_t error, int line) {
	if (error != cudaSuccess) {
		std::cout << "(" << line << ") Cuda error: " << cudaGetErrorString(error) << std::endl;
	}
}

#define checkCudaError(x) checkCudaError(x, __LINE__)

std::vector<half> runWmmaDemo(std::vector<half> a, std::vector<half> b, float* time) {
	std::vector<half> c(16 * 16, 10.0);

	half* device_a;
	half* device_b;
	half* device_c;

	checkCudaError(cudaMallocManaged(&device_a, sizeof(half) * 16 * 16));
	checkCudaError(cudaMallocManaged(&device_b, sizeof(half) * 16 * 16));
	checkCudaError(cudaMallocManaged(&device_c, sizeof(half) * 16 * 16));

	checkCudaError(cudaMemcpy(device_a, a.data(), sizeof(half) * a.size(), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(device_b, b.data(), sizeof(half) * b.size(), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	checkCudaError(cudaEventCreate(&start));
	checkCudaError(cudaEventCreate(&stop));

	checkCudaError(cudaEventRecord(start));				// Start performance measurement

	// TODO: set proper block/grid size
	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks(1);
	wmma16x16<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c);
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	checkCudaError(cudaEventRecord(stop));				// Stop measurement
	checkCudaError(cudaEventSynchronize(stop));
		
	checkCudaError(cudaEventElapsedTime(time, start, stop));

	checkCudaError(cudaMemcpy(c.data(), device_c, sizeof(half) * c.size(), cudaMemcpyDeviceToHost));

	checkCudaError(cudaFree(device_a));
	checkCudaError(cudaFree(device_b));
	checkCudaError(cudaFree(device_c));

	return c;
}

std::vector<half> runMmaDemo(std::vector<half> a, std::vector<half> b, float* time) {
	std::vector<half> c(16 * 16, 10.0);

	half* device_a;
	half* device_b;
	half* device_c;

	checkCudaError(cudaMallocManaged(&device_a, sizeof(half) * 16 * 16));
	checkCudaError(cudaMallocManaged(&device_b, sizeof(half) * 16 * 16));
	checkCudaError(cudaMallocManaged(&device_c, sizeof(half) * 16 * 16));

	checkCudaError(cudaMemcpy(device_a, a.data(), sizeof(half) * a.size(), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(device_b, b.data(), sizeof(half) * b.size(), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks(1);
	cudaEvent_t start, stop;
	checkCudaError(cudaEventCreate(&start));
	checkCudaError(cudaEventCreate(&stop));

	checkCudaError(cudaEventRecord(start));				// Start performance measurement

	mma16x16<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	checkCudaError(cudaEventRecord(stop));				// Stop measurement
	checkCudaError(cudaEventSynchronize(stop));
		
	checkCudaError(cudaEventElapsedTime(time, start, stop));

	checkCudaError(cudaMemcpy(c.data(), device_c, sizeof(half) * c.size(), cudaMemcpyDeviceToHost));

	checkCudaError(cudaFree(device_a));
	checkCudaError(cudaFree(device_b));
	checkCudaError(cudaFree(device_c));

	return c;
}

int main() {
	std::uniform_real_distribution<float> distribution(-10, 10);
	
	std::vector<half> a = randomMatrix(distribution, 16 * 16);
	std::vector<half> b = randomMatrix(distribution, 16 * 16);

	float time_wmm = 0.0f;
	float time_mm = 0.0f;
	std::vector<half> c_wmm = runWmmaDemo(a, b, &time_wmm);
	std::vector<half> c_mm = runMmaDemo(a, b, &time_mm);
	
	std::cout << "With Tensor Cores: " <<  std::to_string(time_wmm) << " ms" << std::endl;
	std::cout << "Cuda: " <<  std::to_string(time_mm) << " ms" << std::endl;

	int mismatches = 0;
	for (int i = 0; i < 16 * 16; i++) {
		if (abs(__half2float(c_wmm[i]) - __half2float(c_mm[i])) > 0.5) {
			std::cout << __half2float(c_wmm[i]) << " " << __half2float(c_mm[i]) << std::endl;
			mismatches += 1;
		}
	}

	if (mismatches > 0) {
		std::cout << mismatches << " wrong" << std::endl;
	}

	return 0;
}
