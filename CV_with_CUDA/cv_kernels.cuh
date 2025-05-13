#pragma once
#include <cuda_runtime.h>

__global__ void sobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step);
__global__ void meanFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);
__global__ void medianFilterKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);

void runSobelKernel(const unsigned char* src, unsigned char* dst, int width, int height, int step);
void runMeanFilter(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);
void runMedianFilter(const unsigned char* src, unsigned char* dst, int width, int height, int step, int kernelSize);