#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width * 2;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height * 2;
}
__global__ void NN_upscaling_kernel(double* d_source, std::uint32_t s_height, std::uint32_t s_width, double* d_result, std::uint32_t r_height, std::uint32_t r_width){

    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < r_width && y < r_height) {
        std::uint32_t s_x = x / 2;
        std::uint32_t s_y = y / 2;

        d_result[y * r_width + x] = d_source[s_y * s_width + s_x];
    }
}

void NN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t r_height = get_NN_upscaled_height(s_height);
    std::uint32_t r_width = get_NN_upscaled_width(s_width);

    cudaMalloc(d_result, r_height * r_width * sizeof(double));

    dim3 blockSize(16, 16);
    dim3 gridSize((r_width + blockSize.x - 1) / blockSize.x, (r_height + blockSize.y - 1) / blockSize.y);

    cudaDeviceSynchronize();
}