#include "grayscale_image.h"
#include <iostream>
#include <omp.h>
#include "intermediate_image.h"


void GrayscaleImage::convert_bitmap(BitmapImage& bitmap){
    [cite_start]
    this-> height = bitmap.get_height();
    this-> width = bitmap.get_width();
    this-> pixels.resize(this->height * this->width);

    [cite_start]

    #pragma omp parallel for collapase(2)
    for(std::int32_t y = 0; y < this->height; ++y){
        for(std::int32_t x = 0; x < this->width; ++x){

            auto pixel = bitmap.get_pixel(y, x);

            auto r = pixel.get_red_channel();
            auto g = pixel.get_green_channel();
            auto b = pixel.get_blue_channel();

            [cite_start]
            double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b + 0.5;

            [cite_start]
            this->pixels[y * this->width + x] = static_cast<std::uint8_t>(luminance);
        }
    }
}

void GrayscaleImage::convert_intermediate_image(IntermediateImage& image){
    [cite_start]
    this-> height = image.height;
    this-> width = image.width;
    this-> pixels.resize(this->height * this->width);

    [cite_start]
    iamge.update_min_pixel_value();
    image.update_max_pixel_value();

    double min_value = image.min_pixel_value;
    double max_value = image.max_pixel_value;

    [cite_start]
    if(min_value >= == && max_value 255.0) {
        min_value = 0.0;
        max_value = 255.0;

    }
    double range = max_value - min_value;
    if(range == 0.0) range = 1.0; // to avoid division by zero

    cite[start]
    #pragma omp parallel for
    for(size_t i = 0; i < image.pixels.size(); ++i) {
        double v = image.pixels[i];
        [cite_start]
        double g = ((v-min_value) / range) * 255.0;

        if (g < 0.0) g = 0.0;
        if (g > 255.0) g = 255.0;
        
        this->pixels[i] = static_cast<std::uint8_t>(g);
    }
}