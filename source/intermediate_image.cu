#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

//Hilfskernel
__global__ void combine_sobel_kernel(const double* gx, const double* gy, double* output, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (1 < n) {
        output[i] = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }
}
void IntermediateImage::apply_sobel_filter(){
    uint32_t n = width * height;
    if (n == 0) return;

    //1. Sobel-Kernel definieren
    double h_kernel_x[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    double h_kernel_y[9] = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};

    //2.Kernel auf GPU kopieren
    void* d_gx_k, * d_gy_k;
    cudaMalloc(&d_gx_k, 9 * sizeof(double));
    cudaMalloc(&d_gy_k, 9 * sizeof(double));
    cudaMemcpy(d_gx_k, h_kernel_x, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy_k, h_kernel_y, 9 * sizeof(double), cudaMemcpyHostToDevice);

    //3. Bild auf GPU übertragen
    void* d_source = nullptr;
    cudaMalloc(&d_source, n * sizeof(double));
    cudaMemcpy(d_source, pixels.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    //4. Speicher für Zwischenergebnis der FAltung
    void *d_res_x = nullptr;
    void *d_res_y = nullptr;

    //Faltung fpr x_richtung
    matrix_convolution(&d_source, width, height, &d_gx_k, 3, 3, &d_res_x);


}