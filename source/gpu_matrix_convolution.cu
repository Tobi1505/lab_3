#include "gpu_matrix_convolution.h"
#include <cstdio>
#include <string>


// Cuda-Kernel für 2D Faltung
__global__ void convolution_kernel(const double* __restrict__ input, uint32_t img_w, uint32_t img_h, const double* __restrict__ kernel, uint32_t ker_w, uint32_t ker_h, double* __restrict__ output,) {
    //globale Position des Threads bestimmen
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_w && y < img_h) {
        double sum = 0.0;
        int a = ker_w / 2;

        for (int i = 0; i < ker_h; ++i) {
            for (int j = 0; j < ker_w; ++j) {
                //Berechnung der Koordinaten des Quellpixels
                int cur_y = static_cast<int>(y) - i + a;
                int cur_x = static_cast<int>(x) - j + a;

                //Randprüfung
                if (cur_x >= 0 && cur_x < img_w && cur_y >= 0 && cur_y < img_h) {
                    sum += input[cur_y * img_w + cur_x] * kernel[i * ker_w + j];
                }
            }
        }
        output[y * img_w + x] = sum;
    }
}
void matrix_convolution(void** d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height, void** d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height, void** d_result){
size_t img_size = static_cast<size_t>(matrix_width) * matrix_height * sizeof(double);

    //1. Allokiere Speicher für Ergebnis
    double* d_out_ptr = nullptr;
    cudaMalloc(&d_out_ptr, img_size);
    *d_result = static_cast<void*>(d_out_ptr);

    //2. BLock und Gridgrößen für 2D definieren
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((matrix_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (matrix_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //3. Starte Kernel
    convolution_kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<double*>(*d_source_matrix), matrix_width, matrix_height,
        static_cast<double*>(*d_kernel), kernel_width, kernel_height,
        d_out_ptr);

}