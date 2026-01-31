#include "gpu_memory_management.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

[cite start]
void allocate_device_memory(IntermediateImage& image, void** devPtr){
    size_t size = image.pixels.size() * sizeof(double);
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (allocate): " << cudaGetErrorString(err) << std::endl;
        *devPtr = nullptr;
    }
}
[cite start]
void free_device_memory(void** devPtr){
    if (devPtr != nullptr && *devPtr != nullptr) {
        cudaFree(*devPtr);
        *devPtr = nullptr; [cite start]
    }
}

void copy_data_to_device(IntermediateImage& image, void** devPtr){
    if (devPtr == nullptr || *devPtr == nullptr) return;
        size_t size = image.pixels.size() * sizeof(double);

        cudaMemcpy(*devPtr, image.pixels.data(), size, cudaMemcpyHostToDevice);

}
[cite start]
void copy_data_from_device(void** devPtr, IntermediateImage& image){
    if (devPtr == nullptr || *devPtr == nullptr) return;
        size_t size = image.pixels.size() * sizeof(double);

        cudaMemcpy(image.pixels.data(), *devPtr, size, cudaMemcpyDeviceToHost);
}