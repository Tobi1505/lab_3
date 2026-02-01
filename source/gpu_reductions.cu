#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
//Kernel für blockweise Reduktion
__global__ void max_reduction_kernel(const double* __restrict__ input, double* __restrict__ block_maxima, uint32_t n) {
    extern __shared__ double sdata[];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    //Initialisierung des Shared Memory
    double thread_max = -1.0; //Pixelwerte >= 0
    if (i < n) {
        thread_max = input[i];
    }
    sdata[tid] = thread_max;
    __syncthreads();

    //Reduktion im Shared Memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    //Maximum des Blockes in globalen Speicher schreiben
    if (tid == 0) {
        block_maxima[blockIdx.x] = sdata[0];
    }
}
double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width){
    uint32_t n = source_image_width * source_image_height;
    if (n == 0) return 0.0;

    const uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    double *d_input = static_cast<double*>(d_source_image);
    double *d_block_maxima;

    //Speicher allokieren
    cudaMalloc(&d_block_maxima, blocks_per_grid * sizeof(double));

    //1. Durchgang: Maxima pro Block finden
    max_reduction_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(double)>>>(d_block_maxima, d_final_max, blocks_per_grid);

    //2. Durchgang: block_maxima auf ein Maximum reduzieren
    double *d_final_max;
    cudaMalloc(&d_final_max, sizeof(double));

    max_reduction_kernel<<<1, threads_per_block, threads_per_block * sizeof(double)>>>(d_block_maxima, d_final_max, blocks_per_grid);

    double h_max = 0.0;
    //finaler Wert zum Host übertragen
    cudaMemcpy(&h_max, d_final_max, sizeof(double), cudaMemcpyDeviceToHost);

    //Speicher freigeben
    cudaFree(d_block_maxima);
    cudaFree(d_final_max);

    return h_max;
}