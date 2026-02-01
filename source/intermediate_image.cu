#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

//Hilfskernel
__global__ void combine_sobel_kernel(const double* gx, const double* gy, double* output, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }
}
void IntermediateImage::apply_sobel_filter(){
    uint32_t n = width * height;
    if (n == 0) return;

    // --- ZEITMESSUNG VORBEREITEN
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); //Start Flagge

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
    //Faltung für y_Richtung
    matrix_convolution(&d_source, width, height, &d_gy_k, 3, 3, &d_res_y);

    //5. Ergebnisse kombinieren
    void* d_final = nullptr;
    cudaMalloc(&d_final, n * sizeof(double));

    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    combine_sobel_kernel<<<blocks, threads>>>(
        static_cast<double*>(d_res_x),
        static_cast<double*>(d_res_y),
        static_cast<double*>(d_final),
        n
        );
    //6. Ergebnis zurück in CPU-Vektor kopieren
    cudaMemcpy(pixels.data(), d_final, n * sizeof(double), cudaMemcpyDeviceToHost);

    //Cuda freigeben
    cudaFree(d_gx_k);
    cudaFree(d_gy_k);
    cudaFree(d_source);
    cudaFree(d_res_x);
    cudaFree(d_res_y);
    cudaFree(d_final);

    // ---ZEITMESSUNG ABSCHLIEßEN
    cudaEventRecord(stop); // Stop-Flagge
    cudaEventSynchronize(stop); //warten bis GPU fertig ist

    float millisecond = 0;
    cudaEventElapsedTime(&millisecond, start, stop);

    //Zeit in Konsole ausgeben
    printf("Bildgröße &d x &d | GPU Zeit: %f ms\n", widht, height, millisecond);

    //Events löschen
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}