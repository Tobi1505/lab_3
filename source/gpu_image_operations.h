#pragma once
#include "intermediate_image.h"
#include "common.cuh"

// downsample the given image using the GPU
// allocates device memory for d_result
void image_downsampling(void** d_source_image, std::uint32_t image_height, std::uint32_t image_width, void** d_result);