#include <stdio.h>
#include <vector>
#include <filesystem>  // need upper version than C++ 17
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step);

int main(void) {
    
}

__global__ void sobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -src[(y - 1) * step + (x - 1)] - 2 * src[y * step + (x - 1)] - src[(y + 1) * step + (x - 1)]
            + src[(y - 1) * step + (x + 1)] + 2 * src[y * step + (x + 1)] + src[(y + 1) * step + (x + 1)];

        int gy = -src[(y - 1) * step + (x - 1)] - 2 * src[(y - 1) * step + x] - src[(y - 1) * step + (x + 1)]
            + src[(y + 1) * step + (x - 1)] + 2 * src[(y + 1) * step + x] + src[(y + 1) * step + (x + 1)];

        int magnitude = abs(gx) + abs(gy);
        dst[y * step + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    }
}