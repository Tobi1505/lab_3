#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    return image_width *2;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    return image_height *2;
}


__global__ void LIN_upscaling_kernel(double* d_source, std::uint32_t s_height, std::uint32_t s_width, double* d_result, std::uint32_t r_height, std::uint32_t r_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < r_width && y < r_height) {
        std::uint32_t s_x = x / 2.0;
        std::uint32_t s_y = y / 2.0;

        double s_x = (double)x / 2.0;
        double s_y = (double)y / 2.0;

        std::uint32_t x_l = (uint32_t)s_x;
        std::uint32_t y_l = (uint32_t)s_y;
        std::uint32_t x_h = (x_l + 1 < s_width) ? x_l + 1 : x_l;
        std::uint32_t y_h = (y_l + 1 < s_height) ? y_l + 1 : y_l;

        double dx = s_x - x_l;
        double dy = s_y - y_l;

        double p1 = d_source[y_l * s_width + x_l];
        double p2 = d_source[y_l * s_width + x_h];
        double p3 = d_source[y_h * s_width + x_l];
        double p4 = d_source[y_h * s_width + x_h];

        d_result[y* r_width + x] = val;
    }
}
void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::unit32_t r_height = s_height * 2;
    std::unit32_t r_width = s_width * 2;
    cudaMalloc(d_result, r_height * r_width * sizeof(double));
    dim3 blockSize(16, 16);
    dim3 gridSize((r_width + blockSize.x - 1) / blockSize.x, (r_height + blockSize.y - 1) / blockSize.y);
    LIN_upscaling_kernel<<<gridSize, blockSize>>>((double*)(*d_source_image), s_height, s_width, (double*)(*d_result), r_height, r_width);
    cudaDeviceSynchronize();
}