#include <stdio.h>
#include <vector>
#include <filesystem>  // need upper version than C++ 17
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step);
__global__ void meanFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);
__global__ void medianFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);

void runSobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step);
void runMeanFilter(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);
void runMedianFilter(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);

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

__global__ void meanFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfSize = kernelSize / 2;

    if (x >= halfSize && x < width - halfSize && y >= halfSize && y < height - halfSize) {
        int sum = 0;
        int count = 0;

        for (int ky = -halfSize; ky <= halfSize; ky++) {
            for (int kx = -halfSize; kx <= halfSize; kx++) {
                sum += src[(y + ky) * step + (x + kx)];
                count++;
            }
        }

        dst[y * step + x] = sum / count;
    }
}

//__global__ void medianFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int halfSize = kernelSize / 2;
//    int windowSize = kernelSize * kernelSize;
//
//    if (x >= halfSize && x < width - halfSize && y >= halfSize && y < height - halfSize) {
//        unsigned char* window = new unsigned char[windowSize];
//        int idx = 0;
//
//        for (int ky = -halfSize; ky <= halfSize; ky++) {
//            for (int kx = -halfSize; kx <= halfSize; kx++) {
//                window[idx++] = src[(y + ky) * step + (x + kx)];
//            }
//        }
//
//        // 버블 소트 (window 크기 작을 때 유리)
//        for (int i = 0; i < windowSize - 1; i++) {
//            for (int j = 0; j < windowSize - i - 1; j++) {
//                if (window[j] > window[j + 1]) {
//                    unsigned char tmp = window[j];
//                    window[j] = window[j + 1];
//                    window[j + 1] = tmp;
//                }
//            }
//        }
//
//        dst[y * step + x] = window[windowSize / 2];
//        delete[] window;
//    }
//}