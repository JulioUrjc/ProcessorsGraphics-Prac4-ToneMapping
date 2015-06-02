#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define BLOCK_SIZE 1024

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

/// Reduction kernel to find the min and max value of an Array
__global__
void findMinMax(const float *d_in_min, const float *d_in_max, float *d_out_min, float *d_out_max){
	/// Share memory
	__shared__ float ds_min[BLOCK_SIZE];
	__shared__ float ds_max[BLOCK_SIZE];

	/// Which thread is?
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x;
	int t2;
	float aux;

	ds_min[threadIdx.x] = d_in_min[t];
	ds_max[threadIdx.x] = d_in_max[t];

	__syncthreads();

	while (totalThreads > 1){
		int halfPoint = (totalThreads >> 1);	// divide by two

		if (threadIdx.x < halfPoint){
			t2 = threadIdx.x + halfPoint;

			// Get the shared value stored by another thread
			aux = ds_min[t2];
			if (aux < ds_min[threadIdx.x])
				ds_min[threadIdx.x] = aux;

			aux = ds_max[t2];
			if (aux > ds_max[threadIdx.x])
				ds_max[threadIdx.x] = aux;
		}
		__syncthreads();

		// Reducing the binary tree size by two:
		totalThreads = halfPoint;
	}
	//Save the min and max at the block
	if (threadIdx.x == 0){
		d_out_min[blockIdx.x] = ds_min[0];
		d_out_max[blockIdx.x] = ds_max[0];
	}
}

/// Kernel to compute the histogram
__global__
void computeHistogram(const float* lumi, int* histogram, const float min, const float range, const int bins){
	
	/// which thread is?
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	//Compute bin
	int bin = ((lumi[t] - min) / range)*bins;

	//Incremento en el bin calculado
	atomicAdd(&(histogram[bin]), 1);
}

__global__
void exclusiveScan(unsigned int* numberArray, int texSize){
	__shared__ int tempArray[BLOCK_SIZE * 2];
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int threadId = threadIdx.x;
	int offset = 1;
	int temp;
	int ai = threadId;
	int bi = threadId + texSize / 2;
	int i;
	//assign the shared memory
	tempArray[ai] = numberArray[id];
	tempArray[bi] = numberArray[id + texSize / 2];

	//up tree
	for (i = texSize >> 1; i > 0; i >>= 1){
		__syncthreads();
		if (threadId < i){
			ai = offset*(2 * threadId + 1) - 1;
			bi = offset*(2 * threadId + 2) - 1;
			tempArray[bi] += tempArray[ai];
		}
		offset <<= 1;
	}

	//put the last one 0
	if (threadId == 0)
		tempArray[texSize - 1] = 0;

	//down tree
	for (i = 1; i < texSize; i <<= 1){//traverse down tree & build scan
		offset >>= 1;
		__syncthreads();

		if (threadId < i){
			ai = offset*(2 * threadId + 1) - 1;
			bi = offset*(2 * threadId + 2) - 1;
			temp = tempArray[ai];
			tempArray[ai] = tempArray[bi];
			tempArray[bi] += temp;
		}
	}
	__syncthreads();

	numberArray[id] = tempArray[threadId];
	numberArray[id + texSize / 2] = tempArray[threadId + texSize / 2];
}

void calculate_cdf(const float* const d_logLuminance, unsigned int* const d_cdf, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols, const size_t numBins)
{
	/// Calcule the size of grid and block
	int blockSize = BLOCK_SIZE;
	int gridSize = ceil((float)(numRows*numCols) / (float) BLOCK_SIZE); // Upper round

	///	1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance

	/// Declare and allocate variables to calculate the max and min values and allocate the memory for it.
	float *d_min, *d_max, *d_min_aux, *d_max_aux;
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float) * BLOCK_SIZE));
	checkCudaErrors(cudaMalloc(&d_max, sizeof(float) * BLOCK_SIZE));

	/// launch the kernel which compute the max and min in each block
	findMinMax << <gridSize, blockSize >> >(d_logLuminance, d_logLuminance, d_min, d_max);

	/// Bucle until reduce all the blocks
	while (gridSize>1)
	{
		//Se reduce el tamaño del grid BLOCKSIZE veces
		gridSize = ceil((float)gridSize / (float)BLOCK_SIZE);
		
		//Iteración Minimo
		checkCudaErrors(cudaMalloc(&d_min_aux, sizeof(float)*gridSize));
		checkCudaErrors(cudaMalloc(&d_max_aux, sizeof(float)*gridSize));
		
		findMinMax << <gridSize, blockSize >> >(d_min, d_max, d_min_aux, d_max_aux);

		d_min = d_min_aux;
		d_max = d_max_aux;
	}

	/// Copy to Host the results of max and min and clean memory
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	///	2) Obtener el rango a representar
	float range = max_logLum - min_logLum;

	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//		bin = (Lum [i] - lumMin) / lumRange * numBins

	/// Recalcule the size of grid and block
	gridSize = ceil((float)(numRows*numCols) / (float)BLOCK_SIZE);
	blockSize = BLOCK_SIZE;

	/// Declare, allocate and zero histogram variable
	int* d_histogram;
	cudaMalloc(&d_histogram, sizeof(int)*numBins);
	cudaMemset(d_histogram, 0, sizeof(int)*numBins);

	/// Launch the kernel for compute the histogram
	computeHistogram << <gridSize, blockSize >> >(d_logLuminance, (int*)d_cdf, min_logLum, range, numBins);

	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	exclusiveScan << < 1, (numBins / 2) >> >(d_cdf, numBins);

	/// Free memory
	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_histogram));
}
